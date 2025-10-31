# hafix_agent/agents/hafix_agent.py
"""
HAFixAgent extending mini-swe-agent's DefaultAgent for Defects4J repair.
"""

import re
from typing import Dict, Any, Optional
from dataclasses import dataclass
from minisweagent.agents.default import DefaultAgent, AgentConfig, FormatError, Submitted
import litellm
from litellm.exceptions import ContextWindowExceededError
from ..utils.model_specs import get_context_limit


@dataclass
class HAFixAgentConfig(AgentConfig):
    """Configuration for HAFixAgent. All values can be overridden by YAML config."""
    # Default limits for Defects4J (overridden by YAML)
    step_limit: int = 50
    cost_limit: float = 1.0

    # Default templates (overridden by YAML config)
    system_template: str = "You are an expert Java developer fixing bugs."
    instance_template: str = "Fix the bug: {{task}}"
    action_observation_template: str = "Observation: {{output}}"
    format_error_template: str = "Please provide exactly one action in triple backticks."
    timeout_template: str = "Command timed out. Try a simpler approach."


class HAFixAgent(DefaultAgent):
    """
    HAFixAgent for Defects4J with perfect fault localization and history integration.
    Extends DefaultAgent with bug context, dual-repo architecture, and enhanced prompting.
    """

    def __init__(self, model, env, logger=None, **kwargs):
        """Initialize with HAFixAgentConfig by default."""
        # If config values are passed directly (from YAML), they override HAFixAgentConfig defaults
        super().__init__(model, env, config_class=HAFixAgentConfig, **kwargs)

        # Defects4J specific context
        self.bug_info = None
        self.blame_info = None
        self.history_category = None

        # Enhanced execution tracking
        self.test_failures_count = 0
        self.compilation_failures = 0

        # Logger for capturing console output to file
        self.logger = logger

        # Adaptive context loading (RQ2)
        self.blame_context_cache = {}  # Cache extracted contexts by heuristic name
        self.requested_heuristics = []  # Track which heuristics were requested and when
        self.model_config = None  # Model config for LLM judge selector (set during initialization)
        self._blame_extraction_cache = None  # Cache blame result to ensure all heuristics use same commits

    def set_bug_context(self, bug_info: Dict[str, Any], blame_info: Optional[Dict] = None, history_category: str = "baseline") -> None:
        """
        Set Defects4J bug context with optional history augmentation.

        Args:
            bug_info: Dictionary containing:
                - bug_id: str (e.g., "Lang_1")
                - fault_locations: List[Dict] with file, start_line, end_line, buggy_code
                - failing_tests: List[str] of test names
                - description: str (optional)
            blame_info: Optional blame commit information
            history_category: History augmentation category (baseline, function_code_pair, etc.)
        """
        self.bug_info = bug_info
        self.blame_info = blame_info
        self.history_category = history_category

        # Add to extra template variables for Jinja2 templates
        self.extra_template_vars.update({
            "bug_id": bug_info.get("bug_id", ""),
            "fault_locations": bug_info.get("fault_locations", []),
            "failing_tests": bug_info.get("failing_tests", []),
            "is_multi_hunk": len(bug_info.get("fault_locations", [])) > 1,
            "description": bug_info.get("description", ""),
            "history_category": history_category,
            "has_blame_info": blame_info is not None,
            # Prefer dataset-provided checkout path inside container; fallback to /workspace for compatibility
            "repo_path": bug_info.get("repo_path", "/workspace"),
            # Initialize execution tracking variables
            "current_failing_tests": [],
            "tests_fixed": False,
            "test_failures_count": 0,
            "compilation_failures": 0,
            "compilation_status": None,
        })

        # Also set the environment working directory so generic edits run in the repo by default
        try:
            repo_path = self.extra_template_vars.get("repo_path", "/workspace")
            if hasattr(self.env, "config") and getattr(self.env.config, "cwd", None) != repo_path:
                self.env.config.cwd = repo_path
        except Exception:
            # Be resilient; command prefixing still ensures correctness
            pass

    def _get_repo_path(self) -> str:
        """Return the repository working directory inside the container.

        Priority order:
        - bug_info.repo_path (set by dataset-specific extractor)
        - extra_template_vars.repo_path
        - fallback to '/workspace'
        """
        if self.bug_info and isinstance(self.bug_info, dict):
            rp = self.bug_info.get("repo_path")
            if rp:
                return rp
        return self.extra_template_vars.get("repo_path", "/workspace")

    def _log(self, message: str, level: str = "INFO"):
        """Helper to print to console and log to file."""
        print(message)
        if self.logger:
            self.logger.console_print(message, level)

    def _check_context_limit(self) -> tuple[bool, str]:
        """
        Check if current messages would exceed the model's context limit.

        Returns:
            Tuple of (is_safe, error_message)
        """
        try:
            # Get model name from model instance
            model_name = getattr(self.model, 'model_name', 'unknown')
            context_limit = get_context_limit(model_name)

            # Get max output tokens from model config (default to 8192)
            max_output_tokens = 8192
            if hasattr(self.model, 'model_kwargs') and 'max_tokens' in self.model.model_kwargs:
                max_output_tokens = self.model.model_kwargs['max_tokens']

            # Safety margin to account for template rendering variations
            safety_margin = 500

            # Count tokens in current message history
            try:
                estimated_tokens = litellm.token_counter(model=model_name, messages=self.messages)
            except Exception as e:
                # Fallback: rough estimation if litellm fails
                print(f"Warning: Token counting failed ({e}), using rough estimation")
                total_chars = sum(len(str(msg)) for msg in self.messages)
                estimated_tokens = total_chars // 4  # ~4 chars per token

            # Calculate required space
            required_tokens = estimated_tokens + max_output_tokens + safety_margin

            # Check if we would exceed the limit
            if required_tokens > context_limit:
                error_msg = (
                    f"Context limit would be exceeded: {estimated_tokens} input tokens + "
                    f"{max_output_tokens} max output + {safety_margin} safety margin = "
                    f"{required_tokens} tokens, but model limit is {context_limit}"
                )
                return False, error_msg

            return True, ""

        except Exception as e:
            # If context checking fails, log but continue (fail-safe)
            print(f"Warning: Context limit checking failed: {e}")
            return True, ""

    def parse_action(self, response: dict) -> dict:
        """
        Override to handle both ```bash and ``` blocks for flexibility.
        Also handles Defects4J command shortcuts.
        """
        content = response["content"]

        # Look for bash code blocks first (flexible whitespace)
        actions = re.findall(r"```bash\s*(.*?)\s*```", content, re.DOTALL)

        # If no bash blocks, look for general code blocks
        if not actions:
            actions = re.findall(r"```(?!bash)\s*(.*?)\s*```", content, re.DOTALL)

        # Fallback: handle any triple backtick blocks with optional language
        if not actions:
            actions = re.findall(r"```.*?\n(.*?)\n```", content, re.DOTALL)

        if len(actions) >= 1:
            # Handle single or multiple commands
            processed_commands = []
            repo_path = self._get_repo_path()

            for action in actions:
                action = action.strip()
                if not action:
                    continue

                # Convert high-level commands to Defects4J commands
                if action == "compile":
                    action = f"cd {repo_path} && defects4j compile"
                elif action == "run_tests":
                    action = f"cd {repo_path} && defects4j test -r"
                elif action == "run_failing_tests":
                    action = f"cd {repo_path} && defects4j test -r"
                elif action.startswith("test "):
                    action = f"cd {repo_path} && defects4j test -r"

                processed_commands.append(action)

            if len(processed_commands) == 0:
                self._log(f"   ➤ [No valid commands found]")
                raise FormatError(self.render_template(self.config.format_error_template, actions=actions))
            elif len(processed_commands) == 1:
                final_action = processed_commands[0]
                # Don't log here - will be logged in execute_action
            else:
                # Chain multiple commands with &&
                final_action = " && ".join(processed_commands)
                # Don't log here - will be logged in execute_action

            return {"action": final_action, **response}

        # Handle empty actions
        self._log(f"   ➤ [No bash command found in response]")
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        """
        Ensure ALL commands run from the correct repo inside the container.
        Intercepts hafix-context commands for adaptive context loading (RQ2).
        """
        cmd = action.get("action", "")

        # Log the action command to file
        if cmd:
            self._log(f"   ➤ {cmd}")

        # Intercept hafix-context commands (standalone or in chains)
        # Handle both: "hafix-context fn_all" and "cmd1 && hafix-context fn_all && cmd2"
        if "hafix-context" in cmd:
            pass
        # Prefix any command that doesn't set its own cwd
        if cmd and not cmd.lstrip().startswith("cd "):
            repo_path = self._get_repo_path()
            action["action"] = f"cd {repo_path} && {cmd}"

        # Call parent's execute_action
        return super().execute_action(action)

    def step(self) -> dict:
        """Override step to add progress logging and context limit checking."""
        step_num = self.model.n_calls + 1

        # Show initialization step for clarity
        if step_num == 1:
            self._log(f"\n🔄 Agent step 1: [Initialization - receiving task and context]")
        else:
            self._log(f"\n🔄 Agent step {step_num}...")

        # Check context limit before proceeding
        is_safe, error_msg = self._check_context_limit()
        if not is_safe:
            self._log(f"   ⚠️  {error_msg}", "WARNING")
            # Raise a ContextWindowExceededError that will be caught by mini-swe-agent
            model_name = getattr(self.model, 'model_name', 'unknown')
            try:
                _, llm_provider, _, _ = litellm.get_llm_provider(model_name)
            except Exception:
                llm_provider = 'unknown'
            raise ContextWindowExceededError(
                message=error_msg,
                model=model_name,
                llm_provider=llm_provider
            )

        try:
            result = super().step()
            return result
        except Submitted as e:
            self._log(f"   ✅ Task completed successfully!")
            raise
        except Exception as e:
            self._log(f"   ❌ Step failed: {e}", "ERROR")
            raise


    def get_observation(self, response: dict) -> dict:
        """
        Override to enhance observations with Defects4J context and execution tracking.
        """
        output = super().get_observation(response)
        output_str = str(output.get("output", ""))

        # Compilation tracking - detect compilation failures first
        if ("Running ant (compile)" in output_str or "Running ant (compile.tests)" in output_str):
            if ("FAIL" in output_str and ("BUILD FAILED" in output_str or "Cannot compile" in output_str)):
                self.compilation_failures += 1
                self.extra_template_vars["compilation_status"] = "failed"
                return output  # Don't process test results if compilation failed
            elif " OK" in output_str:
                self.extra_template_vars["compilation_status"] = "successful"

        # Test result analysis (only if compilation succeeded)
        if "Failing tests:" in output_str:
            if "Failing tests: 0" in output_str:
                self.extra_template_vars["tests_fixed"] = True
            else:
                self._analyze_test_output(output)
                self.test_failures_count += 1

        # Timeout handling
        if 'TIMEOUT' in output_str:
            self.test_failures_count += 1

        # Add execution statistics to context
        self.extra_template_vars.update({
            "test_failures_count": self.test_failures_count,
            "compilation_failures": self.compilation_failures
        })

        return output

    def _analyze_test_output(self, output: dict) -> None:
        """
        Analyze test output and update context.
        Helper method to parse Defects4J test results.
        """
        output_str = str(output.get("output", ""))
        lines = output_str.split('\n')

        failing_tests = []
        for line in lines:
            if line.strip().startswith('-') and '::' in line:
                # Parse test name from Defects4J output format
                test_name = line.strip()[1:].strip()
                failing_tests.append(test_name)

        # Update context with current failing tests
        if failing_tests:
            self.extra_template_vars["current_failing_tests"] = failing_tests
            self.extra_template_vars["tests_fixed"] = False
        else:
            self.extra_template_vars["tests_fixed"] = True

    def run(self, **kwargs) -> tuple[str, str]:
        """
        Run the agent with template-based context.

        Args:
            **kwargs: Additional context passed to templates

        Returns:
            Tuple of (exit_status, message)
        """
        # Template variables are already set via agent.extra_template_vars
        # All rendering happens in super().run() when creating system/user messages
        return super().run(task="", **kwargs)

    def has_finished(self, output: dict[str, str]):
        """
        Override to add SWE-Bench/Defects4J-specific completion conditions.

        The agent finishes when:
        1. Standard completion markers are found (parent's logic)
        2. All tests pass and bug is fixed (if we have test info)
        """
        # Check parent's completion conditions first
        parent_result = super().has_finished(output)
        if parent_result:
            return parent_result

        # Additional completion checks for bug fixing
        output_str = output.get("output", "")

        # For Defects4J: Check if all tests are passing
        if "Failing tests: 0" in output_str and self.bug_info:
            # For multi-hunk bugs, we should verify all locations were addressed
            # But for now, trust that passing tests means success
            if self.extra_template_vars.get("is_multi_hunk"):
                # Could add additional verification here
                pass

            # Note: We don't auto-submit here, let the agent explicitly submit
            # This gives the agent a chance to verify the fix

        return False

    # ========================================================================
    # Adaptive Context Loading (RQ2)
    # ========================================================================

    def _extract_blame_for_heuristic(
        self,
        selector_type: str = "llm_judge",
        n_lines: int = 1
    ) -> Dict:
        """
        Reuse existing extraction code to get blame context.
        Calls hafix_agent/blame/core.py:extract_blame_context()

        All heuristics use the same blame commits for consistency.

        Args:
            selector_type: Selector type for blame line selection (default: llm_judge)
            n_lines: Number of lines to select per hunk (default: 1)
        """
        # Return cached result if available (ensures all heuristics use same commits)
        if self._blame_extraction_cache is not None:
            return self._blame_extraction_cache

        from hafix_agent.blame.core import extract_blame_context
        from hafix_agent.blame.patch_parser import PatchFormat

        # Validate we have necessary bug info
        if not self.bug_info:
            raise ValueError("Bug info not available for context extraction")

        # Use golden_patch (forward patch: buggy → fixed)
        golden_patch = self.bug_info.get('golden_patch')
        if not golden_patch:
            raise ValueError("Golden patch not available in bug_info. This field is required for adaptive context loading.")

        # Get buggy files from fault locations
        buggy_files = [loc['file'] for loc in self.bug_info.get('fault_locations', [])]
        if not buggy_files:
            raise ValueError("No fault locations found in bug_info")

        # Extract blame context using existing core function
        # This reuses all the sophisticated blame analysis logic
        blame_result = extract_blame_context(
            patch_content=golden_patch,  # Use golden_patch (forward patch)
            docker_env=self.env,  # Agent has access to Docker environment
            work_dir=self.bug_info.get('repo_path', '/workspace'),
            patch_format=PatchFormat.FORWARD,  # golden_patch is forward patch (buggy → fixed)
            selector_type=selector_type,  # Configurable selector (default: llm_judge for RQ2)
            n_lines=n_lines,  # Configurable n_lines (default: 1 for RQ2)
            buggy_files=buggy_files,
            bug_info=self.bug_info,
            include_blameless=getattr(self, 'adaptive_include_blameless', True),
            model_config=self.model_config  # Pass model config for LLM judge selector
        )

        if 'error' in blame_result:
            raise ValueError(f"Blame extraction failed: {blame_result['error']}")

        # Cache the result for subsequent heuristic requests
        self._blame_extraction_cache = blame_result

        return blame_result

    def _format_blame_context(self, heuristic_name: str, blame_info: Dict) -> str:
        """
        Format blame info into readable context for agent.
        Reuses hafix_agent/prompts/prompt_builder.py:build_history_augmentation()
        """
        from hafix_agent.prompts.prompt_builder import build_history_augmentation
        from hafix_agent.blame.core import history_name_to_category

        category = history_name_to_category[heuristic_name]

        # Reuse existing prompt builder logic
        augmentation = build_history_augmentation(category, blame_info)

        return augmentation
