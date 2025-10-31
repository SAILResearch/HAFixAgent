"""
Blame line(s) selection strategies.

This module contains basic strategies for selecting which lines to blame
from a set of candidate lines in a patch.
"""

import random
import re
import yaml
from abc import ABC, abstractmethod
from typing import List

from .patch_parser import PatchLine


class BlameLineSelector(ABC):
    """Base class for selecting which lines to blame."""
    
    @abstractmethod
    def select(self, blamable_lines: List[PatchLine], n: int = 1) -> List[PatchLine]:
        """
        Select the top N most valuable lines to blame.
        
        Args:
            blamable_lines: List of lines that can be blamed ('+' lines in Defects4J reverse patches)
            n: Number of lines to select (default 1)
            
        Returns:
            List of selected PatchLine objects (up to n lines)
        """
        pass


class RandomSelector(BlameLineSelector):
    """Randomly select lines to blame."""
    
    def __init__(self, seed: int = None):
        """
        Initialize random selector.
        
        Args:
            seed: Random seed for reproducible results (optional)
        """
        if seed is not None:
            random.seed(seed)
    
    def select(self, blamable_lines: List[PatchLine], n: int = 1) -> List[PatchLine]:
        """Randomly select n lines from blamable lines."""
        if not blamable_lines:
            return []
        
        # Select up to n lines (or all if fewer than n available)
        k = min(n, len(blamable_lines))
        return random.sample(blamable_lines, k)


class FirstSelector(BlameLineSelector):
    """Select the first N lines (current default behavior)."""
    
    def select(self, blamable_lines: List[PatchLine], n: int = 1) -> List[PatchLine]:
        """Select the first n lines from blamable lines."""
        return blamable_lines[:n]


class LLMJudgeSelector(BlameLineSelector):
    """Use LLM to judge which lines would provide the most valuable blame context."""
    
    def __init__(self, model_config: dict = None):
        """
        Initialize LLM Judge selector with LiteLLM configuration.

        Args:
            model_config: Model configuration dictionary from config file
        """
        if model_config is None:
            raise ValueError("model_config is required for LLMJudgeSelector")
        self._model_config = model_config
    
    def select(self, blamable_lines: List[PatchLine], n: int = 1) -> List[PatchLine]:
        """Use LLM to select the most valuable lines to blame."""
        if not blamable_lines:
            return []
        
        if len(blamable_lines) <= n:
            return blamable_lines
        
        # Create prompt for LLM
        prompt = self._create_judge_prompt(blamable_lines, n)
        
        try:
            # Query LLM for line selection
            selected_line_numbers = self._query_llm(prompt)
            
            # Map line numbers back to PatchLine objects
            selected_lines = []
            for line_num in selected_line_numbers:
                for line in blamable_lines:
                    if line.line_number == line_num:
                        selected_lines.append(line)
                        break
            
            # Fallback to first N lines if LLM selection failed
            if not selected_lines:
                print("LLM selection failed, falling back to first N lines")
                return blamable_lines[:n]
            
            return selected_lines[:n]  # Ensure we don't exceed n
            
        except Exception as e:
            print(f"Error in LLM judge selection: {e}")
            print("Falling back to first N lines")
            return blamable_lines[:n]
    
    def _create_judge_prompt(self, blamable_lines: List[PatchLine], n: int) -> str:
        """Create prompt for LLM to judge which lines to blame."""

        # Format lines for display using actual line numbers
        lines_text = ""
        for line in blamable_lines:
            lines_text += f"Line {line.line_number} in {line.file_path}: {line.content.strip()}\n"

        # Create specific prompt based on n
        if n == 1:
            instruction = "which single line should I run git blame on"
            example = '"574"'
            choice_text = "Choose exactly ONE line number that provides the most valuable blame history."
        else:
            instruction = f"which {n} lines should I run git blame on"
            example = '"574, 613, 625"'
            choice_text = f"Choose exactly {n} line numbers that provide the most valuable blame history."

        prompt = f"""Given this patch showing deleted lines from a bug fix, {instruction} to get the most helpful historical context for understanding this bug?

Consider:
- Lines that modify core logic are more valuable than cosmetic changes
- Function signatures and key algorithms provide better blame context
- Complex expressions may have interesting evolution history
- Variable assignments and method calls often reveal important changes

Deleted lines that can be blamed:
{lines_text.strip()}

Please respond with just the actual line numbers that would provide the most valuable blame history (e.g., {example}). {choice_text}

Response:"""

        return prompt
    
    
    def _query_llm(self, prompt: str) -> List[int]:
        """
        Query LLM with the prompt and parse response.
        
        Args:
            prompt: The prompt to send to LLM
            
        Returns:
            List of line numbers selected by LLM
        """
        # Check if we have a valid model config
        if not self._model_config:
            print("[LLM Query] No model config available, using mock response")
            mock_response = "42"
            print(f"[LLM Query] Mock response: {mock_response}")
            return self._parse_llm_response(mock_response)
        
        try:
            # Import LiteLLM here to avoid dependency issues if not installed
            from litellm import completion
            
            model_name = self._model_config.get('model_name')
            model_kwargs = self._model_config.get('model_kwargs', {})
            
            print(f"[LLM Query] Querying {model_name} with LiteLLM")
            print(f"[LLM Query] FULL PROMPT:")
            print("=" * 80)
            print(prompt)
            print("=" * 80)

            # Make LiteLLM API call
            response = completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **model_kwargs
            )

            llm_response = response.choices[0].message.content
            print(f"[LLM Query] FULL RESPONSE:")
            print("=" * 80)
            print(llm_response)
            print("=" * 80)
            
            # Parse the response to extract line numbers
            return self._parse_llm_response(llm_response)
            
        except ImportError:
            print("[LLM Query] LiteLLM not installed, using mock response")
            mock_response = "42"
            return self._parse_llm_response(mock_response)
            
        except Exception as e:
            print(f"[LLM Query] Error calling LiteLLM: {e}")
            print("[LLM Query] Falling back to mock response")
            mock_response = "42"
            return self._parse_llm_response(mock_response)
    
    def _parse_llm_response(self, response: str) -> List[int]:
        """
        Parse LLM response to extract line numbers.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            List of line numbers
        """
        try:
            # Extract numbers from response using word boundary regex
            numbers = re.findall(r'\b\d+\b', response.strip())
            line_numbers = [int(num) for num in numbers]
            
            if not line_numbers:
                print("No valid line numbers found in LLM response")
                return []
            
            return line_numbers
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return []


def get_selector(selector_type: str, model_config: dict = None) -> BlameLineSelector:
    """
    Factory function to get a selector by name.

    Args:
        selector_type: Type of selector ("random", "first", "llm_judge")
        model_config: Model configuration dictionary (required for llm_judge selector)

    Returns:
        BlameLineSelector instance
    """
    if selector_type.lower() == "random":
        return RandomSelector()
    elif selector_type.lower() == "first":
        return FirstSelector()
    elif selector_type.lower() == "llm_judge":
        if model_config is None:
            raise ValueError("model_config is required for llm_judge selector")
        return LLMJudgeSelector(model_config)
    else:
        raise ValueError(f"Unknown selector type: {selector_type}. Available: 'random', 'first', 'llm_judge'")