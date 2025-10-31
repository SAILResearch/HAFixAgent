"""
Defects4J blame context extraction in container and baseline bug information.

ARCHITECTURE USAGE:
- External code should use `hafix_agent.blame.context_loader` interface
- Internal methods (_extract_*) are private and should not be called directly
- Use create_context_loader('runtime') for runtime extraction
- Use create_context_loader('cached', dataset_path, category_filter) for cached data

Example correct usage:
    from hafix_agent.blame.context_loader import create_context_loader
    loader = create_context_loader('runtime')
    result = loader.get_blame_context(project, bug_id, extractor, ...)
"""
import sys
import json
import javalang
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from hafix_agent.blame.interface import BlameExtractor, BugInfoExtractor
from hafix_agent.environments.defects4j_docker import Defects4JDocker
from hafix_agent.blame.core import extract_blame_context
from hafix_agent.blame.patch_parser import PatchParser, PatchFormat
from .util import defects4j_project_name_url_map, get_defects4j_work_dir, get_chart_blame_dir, get_chart_temp_dir, ensure_defects4j_docker_container


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class Defects4JExtractor(BlameExtractor, BugInfoExtractor):

    def _extract_blame_context(
        self,
        project_name: str,
        bug_id: str,
        selector_type: str = "first",
        n_lines: int = 1,
        docker_env=None,
        image: str = "defects4j:latest",
        use_existing_container: str = None,
        cleanup_on_exit: bool = True,
        bug_info: Dict = None,
        include_blameless: bool = True,
        **kwargs
    ) -> Dict:
        """Extract blame context using shared or new containers."""
        
        try:
            # Ensure Docker container is ready
            docker_env, error_result = ensure_defects4j_docker_container(
                project_name, bug_id, docker_env, image, use_existing_container, cleanup_on_exit
            )
            if error_result:
                return error_result
            
            bug_id_int = int(bug_id)
            
            # 3. Get container work directory
            container_work_dir = get_defects4j_work_dir(project_name, bug_id)
            
            # 4. Get commit ID from active-bugs.csv
            commit_id = self._read_commit_id_from_csv(docker_env, project_name, bug_id_int)
            if not commit_id:
                return {
                    "bug_info": None, 
                    "blame_info": None,
                    "error": f"Could not find commit ID for {project_name}_{bug_id}"
                }
            
            # 5. Handle Chart project specially or switch to original commit for blame analysis
            chart_blame_dir = None  # Initialize for scope
            if project_name == "Chart":
                # For Chart, clone the original repo and checkout to correct commit
                chart_blame_dir = get_chart_blame_dir(bug_id)
                chart_repo_url = defects4j_project_name_url_map.get("Chart")
                
                # Clone Chart repository for blame analysis
                clone_result = docker_env.execute(
                    f"rm -rf {chart_blame_dir} && git clone {chart_repo_url} {chart_blame_dir}"
                )
                if clone_result['returncode'] != 0:
                    return {
                        "bug_info": None,
                        "blame_info": None,
                        "error": f"Failed to clone Chart repository: {clone_result['output']}"
                    }
                
                # Checkout to the buggy commit
                chart_checkout_result = docker_env.execute(
                    f"cd {chart_blame_dir} && git checkout {commit_id}"
                )
                if chart_checkout_result['returncode'] != 0:
                    return {
                        "bug_info": None,
                        "blame_info": None,
                        "error": f"Failed to checkout Chart commit {commit_id}: {chart_checkout_result['output']}"
                    }
                
                # Use the cloned Chart directory for blame analysis
                blame_work_dir = chart_blame_dir
            else:
                # For other projects, use the standard approach
                git_checkout_result = docker_env.execute(
                    f"cd {container_work_dir} && git checkout {commit_id}"
                )
                
                if git_checkout_result['returncode'] != 0:
                    return {
                        "bug_info": None,
                        "blame_info": None,
                        "error": f"Failed to checkout commit {commit_id}: {git_checkout_result['output']}"
                    }
                
                # Use the standard Defects4J directory
                blame_work_dir = container_work_dir
            
            # Get patch content from container for sophisticated analysis
            patch_result = docker_env.execute(f"cat /defects4j/framework/projects/{project_name}/patches/{bug_id_int}.src.patch")
            if patch_result['returncode'] != 0:
                return {
                    "bug_info": None,
                    "blame_info": None,
                    "error": f"Could not read patch file for {project_name}-{bug_id_int}"
                }
            
            patch_content = patch_result['output']

            # 6. Perform the git blame using the appropriate directory
            blame_info = extract_blame_context(
                patch_content=patch_content,
                docker_env=docker_env,
                work_dir=blame_work_dir,  # Use cloned Chart repo for Chart, standard D4J for others
                patch_format=PatchFormat.REVERSE,  # .src.patch is reverse patch (fixed → buggy)
                selector_type=selector_type,
                n_lines=n_lines,
                buggy_files=None,
                bug_info=bug_info,
                include_blameless=include_blameless,
                model_config=kwargs.get('model_config')
            )
            
            # Clean up Chart clone if it was created
            if project_name == "Chart" and chart_blame_dir:
                docker_env.execute(f"rm -rf {chart_blame_dir}")
            
            # 7. Switch back to Defects4J version after blame extraction (for later agent fixing running)
            d4j_tag = f"D4J_{project_name}_{bug_id}_BUGGY_VERSION"
            docker_env.execute(f"cd {container_work_dir} && git checkout {d4j_tag}")
            
            # Build result
            bug_info = {
                "project": project_name,
                "bug_id": f"{project_name}_{bug_id}",
                "container_ready": True
            }
            
            return {
                "bug_info": bug_info,
                "blame_info": blame_info,
                "docker_env": docker_env
            }
            
        except Exception as e:
            return {
                "bug_info": None,
                "blame_info": None, 
                "error": f"Container blame extraction failed: {str(e)}"
            }

    def _extract_bug_info_context(
        self, 
        project_name: str, 
        bug_id: str, 
        docker_env=None,
        image: str = "defects4j:latest",
        use_existing_container: str = None,
        cleanup_on_exit: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract bug information for Defects4J with optional Docker container for enhanced fault locations."""
        try:
            # Ensure Docker container is available (shared or create new)
            docker_env, error = ensure_defects4j_docker_container(
                project_name, bug_id, docker_env, image, use_existing_container, cleanup_on_exit
            )
            if error:
                return {
                    'project': project_name,
                    'bug_id': f"{project_name}_{bug_id}",
                    'repo_path': get_defects4j_work_dir(project_name, bug_id),
                    'error': error.get('error', 'Unknown error')
                }
            
            # Extract fault locations from patch files (enhanced with Docker if available)
            # For blameless cases, also consider - lines as fault locations
            is_blameless = self._is_blameless_case(project_name, bug_id)
            fault_locations = self._extract_fault_locations_from_patch(
                project_name, bug_id, project_root, docker_env=docker_env, include_blameless=is_blameless
            )

            # Load bug description from mined JSON files
            description = self._load_bug_description(project_name, bug_id, project_root)

            # Extract commit message from fixed commit
            commit_message = self._extract_commit_message(project_name, bug_id, docker_env)

            # Use commit message as fallback description if description is empty
            if not description.strip():
                description = commit_message

            # Get failing tests from active-bugs.csv (simplified)
            # Optimized: we filter the stack trace in the trigger test file, to only contain the project-related trace
            failing_tests = self._extract_failing_tests(project_name, bug_id, project_root)

            # Extract golden patch (forward patch: buggy -> fixed)
            golden_patch = self._extract_golden_patch(project_name, bug_id, docker_env)

            # Fallback: If no fault locations found and this is a blameless case, try golden patch
            if not fault_locations and self._is_blameless_case(project_name, bug_id):
                if golden_patch:
                    fault_locations = self._extract_fault_locations_from_golden_patch(
                        project_name, bug_id, golden_patch, docker_env
                    )
                    if fault_locations:
                        print(f"Info: Extracted {len(fault_locations)} fault locations from golden patch for blameless case {project_name}_{bug_id}")

            return {
                'project': project_name,
                'bug_id': f"{project_name}_{bug_id}",
                'repo_path': get_defects4j_work_dir(project_name, bug_id),
                "github_url": defects4j_project_name_url_map.get(project_name),
                'description': description,
                'commit_message': commit_message,
                'fault_locations': fault_locations,
                'failing_tests': failing_tests,
                'golden_patch': golden_patch
            }
            
        except Exception as e:
            print(f"Warning: Could not extract bug info for {project_name}_{bug_id}: {e}")
            return {
                'project': project_name,
                'bug_id': f"{project_name}_{bug_id}",
                'repo_path': get_defects4j_work_dir(project_name, bug_id),
                'description': "",
                'fault_locations': [],
                'failing_tests': []
            }


    @staticmethod
    def _read_commit_id_from_csv(docker_env: Defects4JDocker, project_name: str, bug_id: int, commit_type: str = "buggy") -> Optional[str]:
        """Read commit ID from active-bugs.csv in container.
        
        Args:
            commit_type: Either "buggy" (field 1) or "fixed" (field 2)
        """
        if project_name == "Chart":
            # For Chart project, use our corrected CSV file
            return Defects4JExtractor._read_chart_commit_id_from_corrected_csv(bug_id, commit_type)
        
        # For other projects, use standard Defects4J CSV
        csv_path = f"/defects4j/framework/projects/{project_name}/active-bugs.csv"
        result = docker_env.execute(f"grep '^{bug_id},' {csv_path}")

        if result['returncode'] == 0:
            fields = result['output'].strip().split(',')
            if commit_type == "buggy" and len(fields) >= 2:
                return fields[1]  # revision.id.buggy
            elif commit_type == "fixed" and len(fields) >= 3:
                return fields[2]  # revision.id.fixed

        return None

    @staticmethod
    def _read_chart_commit_id_from_corrected_csv(bug_id: int, commit_type: str = "buggy") -> Optional[str]:
        """Read Chart commit ID from our corrected CSV file.
        
        Args:
            commit_type: Either "buggy" (field 1) or "fixed" (field 2)
        """
        corrected_csv_path = project_root / "dataset" / "defects4j" / "Chart-active-bugs.csv"
        
        try:
            with open(corrected_csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('bug.id'):  # Skip header
                        continue
                    fields = line.strip().split(',')
                    if len(fields) >= 2 and fields[0] == str(bug_id):
                        if commit_type == "buggy" and len(fields) >= 2:
                            return fields[1]  # revision.id.buggy
                        elif commit_type == "fixed" and len(fields) >= 3:
                            return fields[2]  # revision.id.fixed
        except Exception as e:
            print(f"Warning: Could not read corrected Chart CSV {corrected_csv_path}: {e}")
        
        return None

    @staticmethod
    def _load_bug_description(project_name: str, bug_id: str, base_dir: Path) -> str:
        """Load bug description from mined JSON files."""
        bug_desc_file = base_dir / "dataset" / "defects4j" / "bug_description" / project_name / "bug_description.json"
        
        if not bug_desc_file.exists():
            print(f"Warning: Bug description file not found: {bug_desc_file}")
            return f"Bug {bug_id} in {project_name}"
        
        try:
            with open(bug_desc_file, 'r', encoding='utf-8') as f:
                bug_data = json.load(f)
                bug_key = f"{project_name}_{bug_id}"
                if bug_key in bug_data:
                    description = bug_data[bug_key].get('description', f"Bug {bug_id} in {project_name}")
                    return description
                else:
                    print(f"Warning: Bug key '{bug_key}' not found in {bug_desc_file}")
                    print(f"Available keys: {list(bug_data.keys())[:3]}...")
        except Exception as e:
            print(f"Warning: Could not load bug description for {project_name}_{bug_id}: {e}")
        
        return f"Bug {bug_id} in {project_name}"

    @staticmethod
    def _extract_commit_message(project_name: str, bug_id: str, docker_env) -> str:
        """Extract commit message from the fixed commit."""
        try:
            # Get fixed commit ID using unified function
            fixed_commit_id = Defects4JExtractor._read_commit_id_from_csv(docker_env, project_name, int(bug_id), "fixed")
            if not fixed_commit_id:
                return ""
            
            # Get repository work directory
            work_dir = get_defects4j_work_dir(project_name, bug_id)
            
            # Handle Chart project specially (clone repo if needed for commit message)
            if project_name == "Chart":
                # For Chart, we need to clone the repo to get the commit message
                chart_temp_dir = get_chart_temp_dir(bug_id)
                chart_repo_url = defects4j_project_name_url_map.get("Chart")
                
                # Clone and get commit message
                clone_result = docker_env.execute(
                    f"rm -rf {chart_temp_dir} && git clone {chart_repo_url} {chart_temp_dir}"
                )
                if clone_result['returncode'] == 0:
                    commit_msg_result = docker_env.execute(
                        f"cd {chart_temp_dir} && git log --format='%s' -n 1 {fixed_commit_id}"
                    )
                    # Cleanup temp directory
                    docker_env.execute(f"rm -rf {chart_temp_dir}")
                    
                    if commit_msg_result['returncode'] == 0:
                        return commit_msg_result['output'].strip()
            else:
                # For other projects, try to get from the standard D4J checkout
                commit_msg_result = docker_env.execute(
                    f"cd {work_dir} && git log --format='%s' -n 1 {fixed_commit_id}"
                )
                if commit_msg_result['returncode'] == 0:
                    if commit_msg_result['output'].strip() == "Missed on prior commit.":
                        return ""
                    else:
                        return commit_msg_result['output'].strip()

        except Exception as e:
            print(f"Warning: Could not extract commit message for {project_name}_{bug_id}: {e}")
        
        return ""


    @staticmethod
    def _extract_fault_locations_from_patch(project_name: str, bug_id: str, base_dir: Path, docker_env=None, include_blameless: bool = True) -> List[Dict]:
        """Extract fault locations from Defects4J patch files using dataset-agnostic PatchParser."""
        
        # Find patch file
        defects4j_base = base_dir / "vendor" / "defects4j" / "framework" / "projects" / project_name / "patches"
        patch_file = defects4j_base / f"{bug_id}.src.patch"
        
        if not patch_file.exists():
            print(f"Warning: Patch file not found: {patch_file}")
            return []
        
        try:
            with open(patch_file, 'r', encoding='utf-8', errors='replace') as f:
                patch_content = f.read()
            
            # Use dataset-agnostic PatchParser
            parser = PatchParser(patch_format=PatchFormat.REVERSE)  # .src.patch is reverse patch
            patch_lines = parser.extract_code_file_lines(patch_content)

            # Group by file first to load file content once
            files_content = {}
            for patch_line in patch_lines:
                if patch_line.file_path not in files_content:
                    files_content[patch_line.file_path] = Defects4JExtractor._load_source_file_content(
                        project_name, bug_id, patch_line.file_path, docker_env
                    )
            
            # Group patch lines by function/class context for true hunk-based approach
            fault_locations = Defects4JExtractor._group_lines_into_hunks(
                patch_lines, files_content, include_blameless
            )
            
            return fault_locations
        
        except Exception as e:
            print(f"Warning: Could not parse patch file for {project_name}_{bug_id}: {e}")
            return []


    @staticmethod
    def _group_lines_into_hunks(patch_lines, files_content, include_blameless: bool = True) -> List[Dict[str, Any]]:
        """Group patch lines into consecutive change blocks (true hunks by research definition)."""
        fault_locations = []

        # Group blameable lines by file first
        files_lines = {}
        for patch_line in patch_lines:
            if not patch_line.can_blame():
                continue
            if patch_line.file_path not in files_lines:
                files_lines[patch_line.file_path] = []
            files_lines[patch_line.file_path].append(patch_line)

        # For each file, group lines into consecutive blocks
        for file_path, lines in files_lines.items():
            # Sort lines by line number
            lines.sort(key=lambda x: x.line_number)
            file_content = files_content.get(file_path)

            # Group into consecutive blocks (hunks)
            current_hunk = []
            for line in lines:
                if not current_hunk:
                    # Start new hunk
                    current_hunk = [line]
                elif line.line_number == current_hunk[-1].line_number + 1:
                    # Consecutive line, add to current hunk
                    current_hunk.append(line)
                else:
                    # Gap found, finish current hunk and start new one
                    fault_locations.append(
                        Defects4JExtractor._create_fault_location(current_hunk, file_content, file_path)
                    )
                    current_hunk = [line]

            # Add the last hunk
            if current_hunk:
                fault_locations.append(
                    Defects4JExtractor._create_fault_location(current_hunk, file_content, file_path)
                )

        # For blameless cases, also process - lines as insertion points
        if include_blameless:
            blameless_files_lines = {}
            for patch_line in patch_lines:
                if patch_line.change_type == "-":  # Lines that should be inserted
                    if patch_line.file_path not in blameless_files_lines:
                        blameless_files_lines[patch_line.file_path] = []
                    blameless_files_lines[patch_line.file_path].append(patch_line)

            # Process blameless lines to find insertion points
            for file_path, lines in blameless_files_lines.items():
                file_content = files_content.get(file_path)
                insertion_points = Defects4JExtractor._convert_to_insertion_points(lines, file_content, file_path)
                fault_locations.extend(insertion_points)

        return fault_locations

    @staticmethod
    def _create_fault_location(hunk_lines, file_content, file_path) -> Dict[str, Any]:
        """Create a fault location from a consecutive group of patch lines."""
        start_line = min(line.line_number for line in hunk_lines)
        end_line = max(line.line_number for line in hunk_lines)
        line_numbers = [line.line_number for line in hunk_lines]

        # Get context from the first line of the hunk
        if file_content:
            context = Defects4JExtractor._extract_java_context(file_content, start_line)
            context['filename'] = file_path.split('/')[-1]  # Add filename
            fault_location = {
                'file': file_path,
                'start_line': start_line,
                'end_line': end_line,
                'line_numbers': line_numbers,
                **context
            }
        else:
            # Fallback for when source content isn't available
            fault_location = {
                'file': file_path,
                'start_line': start_line,
                'end_line': end_line,
                'line_numbers': line_numbers,
                'function_name': '',
                'class_name': '',
                'filename': file_path.split('/')[-1],
                'code_snippet': '',
                'granularity': 'file'
            }

        return fault_location

    @staticmethod 
    def _load_source_file_content(project_name: str, bug_id: str, file_path: str, docker_env) -> Optional[str]:
        """Load source file content from Docker container."""
        # At this point, docker_env is guaranteed to exist and be set up by the main method
        work_dir = get_defects4j_work_dir(project_name, bug_id)
        source_file_path = f"{work_dir}/{file_path}"
        
        result = docker_env.execute(f"cat {source_file_path}")
        if result['returncode'] == 0:
            return result['output']
        else:
            print(f"Warning: Could not read {source_file_path} from container: {result['output']}")
            return None

    @staticmethod
    def _extract_java_context(file_content: str, target_line: int) -> Dict[str, Any]:
        """Extract function/class context using javalang AST parsing."""
        try:
            tree = javalang.parse.parse(file_content)
            file_lines = file_content.split('\n')
            
            # Find enclosing method and class
            enclosing_method = None
            enclosing_class = None
            method_code = ''
            
            for path, node in tree:
                # Check for method declarations
                if isinstance(node, javalang.tree.MethodDeclaration):
                    method_start = node.position.line if node.position else 0
                    # Estimate method end by finding next method or class boundary
                    method_end = Defects4JExtractor._estimate_method_end(file_lines, method_start, tree, path)
                    
                    if method_start <= target_line <= method_end:
                        enclosing_method = node.name
                        # Extract method code
                        method_code = '\n'.join(file_lines[method_start-1:method_end])
                        break
                
                # Check for class declarations
                elif isinstance(node, javalang.tree.ClassDeclaration):
                    class_start = node.position.line if node.position else 0
                    # Classes typically span many lines - use a heuristic
                    if class_start <= target_line:
                        enclosing_class = node.name
            
            # Determine granularity and naming
            if enclosing_method:
                return {
                    'function_name': enclosing_method,
                    'class_name': enclosing_class or '',
                    'filename': '',  # Will be set by caller
                    'code_snippet': method_code,
                    'granularity': 'function'
                }
            elif enclosing_class:
                return {
                    'function_name': '',
                    'class_name': enclosing_class,
                    'filename': '',  # Will be set by caller
                    'code_snippet': '',
                    'granularity': 'class'
                }
            else:
                return {
                    'function_name': '',
                    'class_name': '',
                    'filename': '',  # Will be set by caller
                    'code_snippet': '',
                    'granularity': 'file'
                }

        except Exception as e:
            print(f"Warning: Could not parse Java file: {e}")
            return {
                'function_name': '',
                'class_name': '',
                'filename': '',  # Will be set by caller
                'code_snippet': '',
                'granularity': 'file'
            }

    # TODO: 1. not robust, can we return by the node.code
    @staticmethod
    def _estimate_method_end(file_lines: List[str], method_start: int, tree, current_path) -> int:
        """Estimate method end line using brace matching heuristic."""
        if method_start >= len(file_lines):
            return len(file_lines)
        
        brace_count = 0
        method_started = False
        
        for i, line in enumerate(file_lines[method_start-1:], method_start):
            for char in line:
                if char == '{':
                    brace_count += 1
                    method_started = True
                elif char == '}':
                    brace_count -= 1
                    if method_started and brace_count == 0:
                        return i
        
        # Fallback: return reasonable default
        return min(method_start + 50, len(file_lines))

    @staticmethod 
    def _extract_failing_tests(project_name: str, bug_id: str, base_dir: Path) -> list[Any] | list[str]:
        """Extract failing tests with essential stack trace context from Defects4J metadata"""
        
        # Try to read from trigger tests if available
        defects4j_base = base_dir / "vendor" / "defects4j" / "framework" / "projects" / project_name
        trigger_tests_file = defects4j_base / "trigger_tests" / f"{bug_id}"
        
        if trigger_tests_file.exists():
            try:
                with open(trigger_tests_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tests = []
                test_entries = content.split('---')[1:]  # Split by test entries
                
                for entry in test_entries[:10]:  # Limit to avoid overwhelming
                    lines = entry.strip().split('\n')
                    if not lines:
                        continue
                        
                    test_name = lines[0].strip()
                    error_msg = ""
                    key_stack_lines = []
                    
                    # Get error message and key stack trace lines
                    for line in lines[1:]:
                        line_stripped = line.strip()
                        if not line_stripped:
                            continue
                            
                        # Capture main error message
                        if not line_stripped.startswith('\tat ') and not error_msg:
                            error_msg = line_stripped
                        
                        # Capture all relevant stack trace lines (project-specific only)
                        elif line_stripped.startswith('\tat ') and project_name.lower() in line_stripped.lower():
                            key_stack_lines.append(line_stripped)
                    
                    tests.append({
                        'test_name': test_name,
                        'error_message': error_msg,
                        'key_stack_trace': key_stack_lines  # All relevant lines
                    })
                
                return [test['test_name'] for test in tests]
                
            except Exception as e:
                print(f"Warning: Could not read trigger tests for {project_name}_{bug_id}: {e}")
        
        # Fallback to generic test name
        return [f"{project_name}Test::test{bug_id}"]

    @staticmethod
    def _extract_golden_patch(project_name: str, bug_id: str, docker_env) -> Optional[str]:
        """Extract golden patch (forward patch: buggy -> fixed) from Defects4J reverse patch.

        Args:
            project_name: Defects4J project name
            bug_id: Bug ID
            docker_env: Docker environment for accessing patch files

        Returns:
            Forward patch string or None if extraction fails
        """
        try:
            # Read reverse patch from Defects4J
            reverse_patch_result = docker_env.execute(f"cat /defects4j/framework/projects/{project_name}/patches/{bug_id}.src.patch")
            if reverse_patch_result['returncode'] != 0:
                print(f"Warning: Could not read reverse patch for {project_name}_{bug_id}")
                return None

            reverse_patch = reverse_patch_result['output'].strip()
            if not reverse_patch:
                return None

            # Convert reverse patch to forward patch by inverting +/- lines
            # Note: This conversion only swaps +/- symbols in code lines.
            # Hunk header line numbers (@@ -X,Y +A,B @@) are not swapped, which is fine for PatchParser
            # (it dynamically tracks line numbers), but the result cannot be used with 'git apply'.
            lines = reverse_patch.split('\n')
            forward_lines = []

            for line in lines:
                if line.startswith('-') and not line.startswith('---'):
                    # Remove line becomes add line in forward patch
                    forward_lines.append('+' + line[1:])
                elif line.startswith('+') and not line.startswith('+++'):
                    # Add line becomes remove line in forward patch
                    forward_lines.append('-' + line[1:])
                else:
                    # Keep header, context, and other lines unchanged
                    forward_lines.append(line)

            return '\n'.join(forward_lines)

        except Exception as e:
            print(f"Warning: Could not extract golden patch for {project_name}_{bug_id}: {e}")
            return None


    @staticmethod
    def get_filtered_bugs(
        bug_filter: str,
        project_filter: str = "",
        limit: int = 0,
        blame_suitable_only: bool = False
    ) -> List[Tuple[str, str]]:
        """
        Get filtered list of bugs from pre-computed CSV file.

        Args:
            bug_filter: Category filter - 'SL', 'SH', 'SFMH', 'MFMH', or 'all'
            project_filter: Specific project name (e.g., 'Math', 'Lang')
            limit: Maximum number of bugs (0 = no limit)
            blame_suitable_only: Filter only bugs suitable for blame (Blame_Suitable=1)

        Returns:
            List of (project_name, bug_id) tuples
        """
        csv_path = Path(__file__).parent / "defects4j_blame_feasibility.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Apply category filter
        if bug_filter != "all":
            # Map bug_filter to CSV Category values
            category_map = {
                'single_line': 'SL',
                'single_hunk': 'SH',
                'single_file_multi_hunk': 'SFMH',
                'multi_file_multi_hunk': 'MFMH'
            }

            # Accept both full names and abbreviations
            category = category_map.get(bug_filter, bug_filter)

            if category not in ['SL', 'SH', 'SFMH', 'MFMH']:
                valid_filters = list(category_map.keys()) + ['SL', 'SH', 'SFMH', 'MFMH', 'all']
                raise ValueError(f"Invalid bug_filter: {bug_filter}. Must be one of: {valid_filters}")

            df = df[df['Category'] == category]

        # Apply project filter
        if project_filter:
            df = df[df['Project'] == project_filter]

        # Apply blame suitable filter
        if blame_suitable_only:
            df = df[df['Blame_Suitable'] == 1]

        # Convert to list of tuples
        bugs_to_evaluate = [(row['Project'], str(row['Bug_Number'])) for _, row in df.iterrows()]

        # Apply limit
        if limit > 0:
            bugs_to_evaluate = bugs_to_evaluate[:limit]

        return bugs_to_evaluate

    @staticmethod
    def get_filtered_bugs_with_blameless(
        bug_filter: str,
        project_filter: str = "",
        limit: int = 0,
        include_blameless: bool = True
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Get filtered list of bugs separated into blameable and blameless cases.

        Args:
            bug_filter: Category filter - 'SL', 'SH', 'SFMH', 'MFMH', or 'all'
            project_filter: Specific project name (e.g., 'Math', 'Lang')
            limit: Maximum number of bugs per category (0 = no limit)
            include_blameless: If True, include blameless cases; if False, return empty blameless list

        Returns:
            Tuple of (blameable_bugs, blameless_bugs) where each is a list of (project_name, bug_id) tuples
        """
        csv_path = Path(__file__).parent / "defects4j_blame_feasibility.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Apply category filter
        if bug_filter != "all":
            # Map bug_filter to CSV Category values
            category_map = {
                'single_line': 'SL',
                'single_hunk': 'SH',
                'single_file_multi_hunk': 'SFMH',
                'multi_file_multi_hunk': 'MFMH'
            }

            # Accept both full names and abbreviations
            category = category_map.get(bug_filter, bug_filter)

            if category not in ['SL', 'SH', 'SFMH', 'MFMH']:
                valid_filters = list(category_map.keys()) + ['SL', 'SH', 'SFMH', 'MFMH', 'all']
                raise ValueError(f"Invalid bug_filter: {bug_filter}. Must be one of: {valid_filters}")

            df = df[df['Category'] == category]

        # Apply project filter
        if project_filter:
            df = df[df['Project'] == project_filter]

        # Separate blameable and blameless cases
        blameable_df = df[df['Blame_Suitable'] == 1]
        blameless_df = df[df['Blame_Suitable'] == 0] if include_blameless else pd.DataFrame()

        # Convert to list of tuples
        blameable_bugs = [(row['Project'], str(row['Bug_Number'])) for _, row in blameable_df.iterrows()]
        blameless_bugs = [(row['Project'], str(row['Bug_Number'])) for _, row in blameless_df.iterrows()]

        # Apply limit to each category separately
        if limit > 0:
            blameable_bugs = blameable_bugs[:limit]
            blameless_bugs = blameless_bugs[:limit]

        return blameable_bugs, blameless_bugs

    @staticmethod
    def get_bug_category(project_name: str, bug_id: int) -> Optional[str]:
        """
        Look up the category for a specific bug from defects4j_blame_feasibility.csv.

        Args:
            project_name: Project name (e.g., 'Math')
            bug_id: Bug ID number

        Returns:
            Category string mapping:
            SL -> single_line
            SH -> single_hunk
            SFMH -> single_file_multi_hunk
            MFMH -> multi_file_multi_hunk
            None if bug not found
        """
        csv_path = Path(__file__).parent / "defects4j_blame_feasibility.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        try:
            df = pd.read_csv(csv_path)

            # Look up the bug
            bug_id_str = f"{project_name}_{bug_id}"
            bug_row = df[df['Bug_ID'] == bug_id_str]

            if bug_row.empty:
                return None

            category = bug_row.iloc[0]['Category']

            # Map categories to the expected format
            category_mapping = {
                'SL': 'single_line',
                'SH': 'single_hunk',
                'SFMH': 'single_file_multi_hunk',
                'MFMH': 'multi_file_multi_hunk'
            }

            return category_mapping.get(category)

        except Exception:
            return None

    @staticmethod
    def _is_blameless_case(project_name: str, bug_id: str) -> bool:
        """Check if a bug is blameless (Blame_Suitable=0) based on CSV data."""
        try:
            csv_path = Path(__file__).parent / "defects4j_blame_feasibility.csv"
            if not csv_path.exists():
                return False

            import pandas as pd
            df = pd.read_csv(csv_path)

            # Look up the bug
            bug_id_str = f"{project_name}_{bug_id}"
            bug_row = df[df['Bug_ID'] == bug_id_str]

            if bug_row.empty:
                return False

            # Return True if Blame_Suitable=0 (blameless)
            return bug_row.iloc[0]['Blame_Suitable'] == 0

        except Exception:
            return False

    @staticmethod
    def _extract_fault_locations_from_golden_patch(project_name: str, bug_id: str, golden_patch: str, docker_env) -> List[Dict]:
        """Extract fault locations from golden patch (forward patch) for blameless cases.

        Args:
            project_name: Defects4J project name
            bug_id: Bug ID
            golden_patch: Forward patch content (buggy -> fixed)
            docker_env: Docker environment for loading source files

        Returns:
            List of fault location dictionaries
        """
        if not golden_patch:
            return []

        try:
            # Use PatchParser with FORWARD format (golden_patch is buggy → fixed)
            parser = PatchParser(patch_format=PatchFormat.FORWARD)
            patch_lines = parser.extract_code_file_lines(golden_patch)

            # Group by file first to load file content once
            files_content = {}
            for patch_line in patch_lines:
                if patch_line.file_path not in files_content:
                    files_content[patch_line.file_path] = Defects4JExtractor._load_source_file_content(
                        project_name, bug_id, patch_line.file_path, docker_env
                    )

            # Group patch lines by function/class context
            fault_locations = Defects4JExtractor._group_lines_into_hunks(
                patch_lines, files_content
            )

            return fault_locations

        except Exception as e:
            print(f"Warning: Could not parse golden patch for {project_name}_{bug_id}: {e}")
            return []

    @staticmethod
    def _convert_to_insertion_points(blameless_lines, file_content, file_path) -> List[Dict[str, Any]]:
        """Convert blameless deletion lines to insertion point fault locations."""
        if not blameless_lines or not file_content:
            return []

        fault_locations = []
        file_lines = file_content.split('\n')

        # Group consecutive deletion blocks
        deletion_blocks = []
        current_block = []

        for line in sorted(blameless_lines, key=lambda x: x.line_number):
            if not current_block:
                current_block = [line]
            elif line.line_number == current_block[-1].line_number + 1:
                current_block.append(line)
            else:
                deletion_blocks.append(current_block)
                current_block = [line]

        if current_block:
            deletion_blocks.append(current_block)

        # Convert each deletion block to insertion point
        for block in deletion_blocks:
            # Find insertion point: line just before the deletion block
            first_deletion_line = min(line.line_number for line in block)
            insertion_point = max(1, first_deletion_line - 1)  # Line before deletion, minimum line 1

            # Ensure insertion point exists in buggy version
            if insertion_point <= len(file_lines):
                # Extract context from the insertion point
                context = Defects4JExtractor._extract_java_context(file_content, insertion_point)
                context['filename'] = file_path.split('/')[-1]

                fault_location = {
                    'file': file_path,
                    'start_line': insertion_point,
                    'end_line': insertion_point,
                    'line_numbers': [insertion_point],
                    **context
                }
                fault_locations.append(fault_location)

        return fault_locations

