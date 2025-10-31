"""
Core git blame operations for HAFixAgent.

This module contains dataset-agnostic git blame functionality that can be used
across different datasets (SWE-Bench, Defects4J, etc.).
"""

import re
from pathlib import Path
from typing import Dict, Optional, List
from enum import Enum
from .patch_parser import PatchParser
from .selection import get_selector
from .patch_parser import PatchFormat


class HistoryCategory(Enum):
    """HAFixAgent's novel historical context categories - core research contribution."""
    adaptive = 0  # RQ2: On-demand adaptive context loading
    baseline = 1
    baseline_co_evolved_functions_name_modified_file_blame = 2
    baseline_co_evolved_functions_name_all_files_blame = 3
    baseline_all_functions_name_modified_file_blame = 4
    baseline_all_functions_name_all_files_blame = 5
    baseline_all_co_evolved_files_name_blame = 6
    baseline_function_code_pair_blame = 7
    baseline_file_code_patch_blame = 8


# CLI-friendly mapping for user interface (your original key names)
history_name_to_category = {
    "adaptive": HistoryCategory.adaptive,  # RQ2 adaptive mode
    "baseline": HistoryCategory.baseline,
    "cfn_modified": HistoryCategory.baseline_co_evolved_functions_name_modified_file_blame,
    "cfn_all": HistoryCategory.baseline_co_evolved_functions_name_all_files_blame,
    "fn_modified": HistoryCategory.baseline_all_functions_name_modified_file_blame,
    "fn_all": HistoryCategory.baseline_all_functions_name_all_files_blame,
    "fln_all": HistoryCategory.baseline_all_co_evolved_files_name_blame,
    "fn_pair": HistoryCategory.baseline_function_code_pair_blame,
    "fl_diff": HistoryCategory.baseline_file_code_patch_blame,
}


def run_git_blame(docker_env, work_dir: str, filepath: str, line_number: int) -> Optional[Dict]:
    """
    Run git blame on a specific line using Docker container and return commit information.
    
    Args:
        docker_env: Docker environment instance with execute method
        work_dir: Working directory in container
        filepath: Relative path to file in repo
        line_number: Line number to blame
        
    Returns:
        Dictionary with blame information or None if failed
    """
    try:
        # Run git blame with porcelain format for easier parsing
        cmd = f"cd {work_dir} && git blame --porcelain -L {line_number},{line_number} {filepath}"
        
        result = docker_env.execute(cmd)
        
        if result['returncode'] != 0:
            return None
        
        # Parse porcelain output
        lines = result['output'].strip().split('\n')
        if not lines:
            return None
            
        # First line has commit hash
        commit_hash = lines[0].split()[0]
        
        # Parse metadata
        blame_info = {'commit_hash': commit_hash}
        for line in lines[1:]:
            if line.startswith('author '):
                blame_info['commit_author'] = line[7:]
            elif line.startswith('author-time '):
                # Convert Unix timestamp to readable format
                import time
                timestamp_str = line[12:].strip()
                try:
                    timestamp = int(timestamp_str)
                    blame_info['commit_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                except (ValueError, OSError):
                    blame_info['commit_timestamp'] = timestamp_str  # Fallback to raw value
            elif line.startswith('summary '):
                blame_info['commit_message'] = line[8:]
            elif line.startswith('filename '):
                # Capture the file path in the blamed commit (handles renames/moves)
                blame_info['blamed_file_path'] = line[9:]

        return blame_info
        
    except Exception as e:
        print(f"Error running git blame: {e}")
        return None


def get_commit_patch(docker_env, work_dir: str, commit_hash: str, filepath: str) -> Optional[str]:
    """
    Get the diff patch for a specific file in a commit using Docker container.
    
    Args:
        docker_env: Docker environment instance with execute method
        work_dir: Working directory in container
        commit_hash: Git commit hash
        filepath: File path to get patch for
        
    Returns:
        Patch string or None if failed
    """
    try:
        # Get only the diff portion without commit metadata (commit info already in blame_commit)
        cmd = f"cd {work_dir} && git show --no-commit-id --format= {commit_hash} -- {filepath}"

        result = docker_env.execute(cmd)

        if result['returncode'] != 0:
            print(f"Git show failed: {result['output']}")
            return None

        return result['output'].strip()
        
    except Exception as e:
        print(f"Error getting commit patch: {e}")
        return None


def get_changed_files_in_commit(docker_env, work_dir: str, commit_hash: str) -> List[str]:
    """
    Get list of changed files in a commit using Docker container.
    
    Args:
        docker_env: Docker environment instance with execute method
        work_dir: Working directory in container
        commit_hash: Git commit hash
        
    Returns:
        List of file paths changed in the commit
    """
    try:
        # Get list of changed files in the commit
        cmd = f"cd {work_dir} && git show --name-only --pretty=format: {commit_hash}"
        
        result = docker_env.execute(cmd)
        
        if result['returncode'] != 0:
            print(f"Git show --name-only failed: {result['output']}")
            return []
        
        # Filter out empty lines and return file paths
        files = [line.strip() for line in result['output'].split('\n') if line.strip()]
        return files
        
    except Exception as e:
        print(f"Error getting changed files: {e}")
        return []


def extract_function_names_from_code(code: str, file_extension: str) -> List[str]:
    """
    Extract function names from code for Python (SWE-Bench) and Java (Defects4J).
    
    Args:
        code: Source code string
        file_extension: File extension (e.g., '.py', '.java')
        
    Returns:
        List of function names found in the code
    """
    function_names = []
    
    if file_extension == '.py':
        # Python patterns for SWE-Bench projects
        patterns = [
            r'^[ \t]*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',      # def function_name(
            r'^[ \t]*async\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # async def function_name(
            r'^[ \t]*class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]'   # class ClassName:
        ]
    elif file_extension == '.java':
        # Java patterns for Defects4J projects
        patterns = [
            # Method patterns: [access] [static] [final] ReturnType methodName(
            r'^[ \t]*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:abstract\s+)?(?:synchronized\s+)?(?:void|boolean|byte|short|int|long|float|double|char|String|[A-Z][a-zA-Z0-9_]*(?:<[^>]*>)?)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            # Constructor patterns: [access] ClassName(
            r'^[ \t]*(?:public|private|protected)?\s+([A-Z][a-zA-Z0-9_]*)\s*\(',
            # Class patterns: [access] [final] [abstract] class ClassName
            r'^[ \t]*(?:public|private|protected)?\s*(?:final\s+)?(?:abstract\s+)?class\s+([A-Z][a-zA-Z0-9_]*)',
            # Interface patterns: [access] interface InterfaceName
            r'^[ \t]*(?:public|private|protected)?\s*interface\s+([A-Z][a-zA-Z0-9_]*)',
            # Enum patterns: [access] enum EnumName
            r'^[ \t]*(?:public|private|protected)?\s*enum\s+([A-Z][a-zA-Z0-9_]*)'
        ]
    else:
        # Only Python and Java are supported for SWE-Bench and Defects4J
        return []
    
    # Extract function names using patterns
    for line in code.split('\n'):
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                function_name = match.group(1)
                # Filter out common false positives
                if function_name not in ['import', 'from', 'if', 'else', 'return', 'try', 'except']:
                    if function_name not in function_names:
                        function_names.append(function_name)
    
    return function_names


def get_file_content_at_commit(docker_env, work_dir: str, commit_hash: str, filepath: str) -> Optional[str]:
    """
    Get the content of a file at a specific commit using Docker container.
    
    Args:
        docker_env: Docker environment instance with execute method
        work_dir: Working directory in container
        commit_hash: Git commit hash
        filepath: File path
        
    Returns:
        File content as string or None if failed
    """
    try:
        cmd = f"cd {work_dir} && git show {commit_hash}:{filepath}"
        
        result = docker_env.execute(cmd)
        
        if result['returncode'] != 0:
            print(f"Git show {commit_hash}:{filepath} failed: {result['output']}")
            return None
        
        return result['output']
        
    except Exception as e:
        print(f"Error getting file content: {e}")
        return None


def get_changed_functions_in_commit(docker_env, work_dir: str, commit_hash: str, filepath: str) -> List[str]:
    """
    Get list of functions that were modified in a specific file in a commit using Docker container.
    
    Args:
        docker_env: Docker environment instance with execute method
        work_dir: Working directory in container
        commit_hash: Git commit hash  
        filepath: File path to analyze
        
    Returns:
        List of function names that were modified
    """
    try:
        # Get the diff for this specific file
        cmd = f"cd {work_dir} && git show --no-merges {commit_hash} -- {filepath}"
        
        result = docker_env.execute(cmd)
        
        if result['returncode'] != 0:
            print(f"Git show diff failed: {result['output']}")
            return []
        
        diff_content = result['output']
        
        # Parse the diff to find function context
        file_extension = Path(filepath).suffix
        changed_functions = []
        
        # Look for function context in diff hunks
        lines = diff_content.split('\n')
        in_hunk = False
        
        for line in lines:
            # Check if this line shows function context (git usually shows this)
            if line.startswith('@@'):
                # Extract function name from hunk header if available
                # Format: @@ -start,count +start,count @@ function_name
                if '@@' in line and len(line.split('@@')) >= 3:
                    context = line.split('@@')[2].strip()
                    if context:
                        # Try to extract function name from context
                        func_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', context)
                        if func_match:
                            func_name = func_match.group(1)
                            if func_name not in changed_functions:
                                changed_functions.append(func_name)
                in_hunk = True
            elif in_hunk and (line.startswith('+') or line.startswith('-')):
                # This is a changed line, try to extract function names from it
                clean_line = line[1:].strip()  # Remove +/- prefix
                function_names = extract_function_names_from_code(clean_line, file_extension)
                for func_name in function_names:
                    if func_name not in changed_functions:
                        changed_functions.append(func_name)
        
        return changed_functions
        
    except Exception as e:
        print(f"Error analyzing changed functions: {e}")
        return []


def is_program_file(filepath: str, exclude_tests: bool = True) -> bool:
    """
    Check if a file is a program file that should be processed.
    
    Args:
        filepath: File path to check
        exclude_tests: Whether to exclude test files
        
    Returns:
        True if file should be processed, False otherwise
    """
    # Define program file extensions for SWE-Bench (Python) and Defects4J (Java)
    PROGRAM_EXTENSIONS = {'.py', '.java'}
    
    # Define test patterns
    TEST_PATTERNS = ['test_', '_test', 'Test', 'Spec', 'spec_', '/test/', '/tests/']
    
    # Check file extension
    file_path = Path(filepath)
    extension = file_path.suffix.lower()
    
    if extension not in PROGRAM_EXTENSIONS:
        return False
    
    # Check if it's a test file
    if exclude_tests:
        filename = file_path.name
        parent_dirs = str(file_path.parent)
        
        # Check if filename or path contains test patterns
        for pattern in TEST_PATTERNS:
            if pattern in filename or pattern in parent_dirs:
                return False
    
    return True


def extract_function_from_code(code: str, function_name: str, file_extension: str) -> str:
    """
    Extract a specific function's code from source code.
    
    Args:
        code: Source code content
        function_name: Name of function to extract
        file_extension: File extension (.py or .java)
        
    Returns:
        Function code or empty string if not found
    """
    if not code or not function_name:
        return ""
    
    lines = code.split('\n')
    function_lines = []
    in_function = False
    indent_level = 0
    
    if file_extension == '.py':
        # Python function extraction
        for i, line in enumerate(lines):
            if not in_function:
                # Look for function definition
                if (f"def {function_name}(" in line or 
                    f"async def {function_name}(" in line or
                    f"class {function_name}" in line):
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                    function_lines.append(line)
            else:
                # Continue collecting function lines
                if line.strip() == "":
                    function_lines.append(line)
                elif len(line) - len(line.lstrip()) > indent_level:
                    # Still inside function (more indented)
                    function_lines.append(line)
                elif len(line) - len(line.lstrip()) == indent_level and line.strip().startswith(('def ', 'class ', 'async def ')):
                    # Next function/class at same level - stop here
                    break
                elif len(line) - len(line.lstrip()) < indent_level and line.strip():
                    # Less indented non-empty line - function ended
                    break
                else:
                    function_lines.append(line)
                    
    elif file_extension == '.java':
        # Java method extraction (simplified)
        brace_count = 0
        for i, line in enumerate(lines):
            if not in_function:
                # Look for method/class definition
                if (function_name in line and 
                    ('public' in line or 'private' in line or 'protected' in line or 
                     'class' in line or 'interface' in line or 'enum' in line)):
                    in_function = True
                    function_lines.append(line)
                    brace_count += line.count('{') - line.count('}')
            else:
                function_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    break
    
    return '\n'.join(function_lines)


def find_function_containing_line(code: str, target_line: int, file_extension: str) -> str:
    """
    Find the function that contains a specific line number.
    
    Args:
        code: Source code content
        target_line: Line number to find (1-based)
        file_extension: File extension (.py or .java)
        
    Returns:
        Function name that contains the line, or empty string if not in any function
    """
    if not code or target_line <= 0:
        return ""
    
    lines = code.split('\n')
    if target_line > len(lines):
        return ""
    
    if file_extension == '.py':
        return _find_python_function_containing_line(lines, target_line)
    elif file_extension == '.java':
        return _find_java_function_containing_line(lines, target_line)
    
    return ""


def _find_python_function_containing_line(lines: list, target_line: int) -> str:
    """Find Python function containing target line with robust edge case handling."""
    
    function_stack = []  # Stack of (function_name, indent_level)
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            continue
            
        line_indent = len(line) - len(line.lstrip())
        
        # Clean up function stack based on indentation
        while function_stack and line_indent <= function_stack[-1][1]:
            # Check if this line is a function/class at the same or lower indent
            if (stripped.startswith(('def ', 'async def ', 'class ')) and 
                line_indent == function_stack[-1][1]):
                break  # Don't pop yet, let the new function replace it
            function_stack.pop()
        
        # Handle decorators - skip lines starting with @
        if stripped.startswith('@'):
            continue
            
        # Check for function/class definition
        if stripped.startswith('def ') or stripped.startswith('async def ') or stripped.startswith('class '):
            function_name = _extract_python_function_name(stripped)
            if function_name:
                # Replace function at same indent level or add new one
                if function_stack and function_stack[-1][1] == line_indent:
                    function_stack[-1] = (function_name, line_indent)
                else:
                    function_stack.append((function_name, line_indent))
        
        # Check if this is our target line (AFTER processing the line)
        if i == target_line:
            # Return the innermost function
            return function_stack[-1][0] if function_stack else ""
    
    return ""


def _extract_python_function_name(stripped_line: str) -> str:
    """Extract function/class name from Python definition line."""
    try:
        if stripped_line.startswith('class '):
            # Handle: class MyClass, class MyClass(Base), class MyClass:
            name_part = stripped_line[6:].split('(')[0].split(':')[0].strip()
        elif stripped_line.startswith('async def '):
            # Handle: async def func_name(params)
            name_part = stripped_line[10:].split('(')[0].strip()
        elif stripped_line.startswith('def '):
            # Handle: def func_name(params)
            name_part = stripped_line[4:].split('(')[0].strip()
        else:
            return ""
        
        # Clean up any generic type annotations like MyClass[T]
        if '[' in name_part:
            name_part = name_part.split('[')[0]
            
        return name_part if name_part.isidentifier() else ""
    except:
        return ""


def _find_java_function_containing_line(lines: list, target_line: int) -> str:
    """Find Java method containing target line with robust edge case handling."""
    
    brace_count = 0
    in_class = False
    class_brace_level = 0
    current_method = ""
    method_start_brace = -1
    
    # Track multiline constructs
    in_multiline_signature = False
    multiline_signature = ""
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip empty lines and comments  
        if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
            continue
        
        # Track brace levels
        old_brace_count = brace_count
        brace_count += line.count('{') - line.count('}')
        
        # Handle multiline signatures
        if in_multiline_signature:
            multiline_signature += " " + stripped
            if '{' in line:
                in_multiline_signature = False
                method_name = _extract_java_method_name(multiline_signature)
                if method_name:
                    current_method = method_name
                    method_start_brace = old_brace_count
                multiline_signature = ""
        else:
            # Check for class/interface/enum definition
            if (('class ' in line or 'interface ' in line or 'enum ' in line) and 
                ('{' in line or not ';' in line)):
                in_class = True
                class_brace_level = old_brace_count
                current_method = ""  # Reset method when entering class
            
            # Check for method definition (inside class)
            elif in_class and _is_java_method_line(stripped):
                if '{' in line:
                    # Single-line method signature
                    method_name = _extract_java_method_name(stripped)
                    if method_name:
                        current_method = method_name
                        method_start_brace = old_brace_count
                else:
                    # Multiline method signature
                    in_multiline_signature = True
                    multiline_signature = stripped
        
        # Reset method when exiting method scope
        if current_method and method_start_brace >= 0 and brace_count <= method_start_brace:
            current_method = ""
            method_start_brace = -1
        
        # Reset everything when exiting class
        if in_class and brace_count <= class_brace_level:
            in_class = False
            current_method = ""
            method_start_brace = -1
        
        # Check if this is our target line (AFTER processing the line)
        if i == target_line:
            return current_method if (in_class and current_method and 
                                    brace_count > method_start_brace >= 0) else ""
    
    return ""


def _is_java_method_line(stripped_line: str) -> bool:
    """Check if line contains a Java method definition."""
    # Skip annotations, static blocks, and other non-method constructs
    if (stripped_line.startswith('@') or 
        stripped_line.startswith('{') or
        stripped_line.startswith('static {') or
        stripped_line.startswith('import ') or
        stripped_line.startswith('package ')):
        return False
    
    # Must have method characteristics
    has_access_modifier = any(keyword in stripped_line for keyword in 
                             ['public', 'private', 'protected', 'static', 'final', 'abstract'])
    has_parentheses = '(' in stripped_line and ')' in stripped_line
    not_field_declaration = not (stripped_line.endswith(';') and '{' not in stripped_line)
    
    return has_access_modifier and has_parentheses and not_field_declaration


def _extract_java_method_name(signature: str) -> str:
    """Extract method name from Java method signature."""
    try:
        # Remove annotations and extra whitespace
        signature = re.sub(r'@\w+(\([^)]*\))?\s*', '', signature)
        signature = ' '.join(signature.split())
        
        # Find the method name (word before opening parenthesis)
        if '(' not in signature:
            return ""
        
        before_paren = signature.split('(')[0].strip()
        words = before_paren.split()
        
        if not words:
            return ""
        
        # Method name is typically the last word before (
        method_name = words[-1]
        
        # Handle generics: methodName<T>(...) -> methodName
        if '<' in method_name:
            method_name = method_name.split('<')[0]
        
        # Handle constructor case: class name == method name
        # For constructors, the method name should be the class name
        return method_name if method_name.replace('_', '').replace('$', '').isalnum() else ""
        
    except:
        return ""


def extract_function_by_signature(code: str, function_name: str, class_name: str = None, file_extension: str = ".java") -> str:
    """
    Extract function code by signature (name + class) instead of line number.
    This is more robust as function signatures don't change as often as line numbers.

    Args:
        code: Source code content
        function_name: Function name to find (e.g., "findWrapPos")
        class_name: Class name containing the function (e.g., "HelpFormatter")
        file_extension: File extension (.py or .java)

    Returns:
        Function code or empty string if not found
    """
    if not code or not function_name:
        return ""

    if file_extension == '.java':
        return _extract_java_function_by_signature(code, function_name, class_name)
    elif file_extension == '.py':
        return _extract_python_function_by_signature(code, function_name, class_name)

    return ""


def _extract_java_function_by_signature(code: str, function_name: str, class_name: str = None) -> str:
    """Extract Java function by signature using enhanced regex."""
    lines = code.split('\n')

    # If we have class name, first find the class boundaries
    class_start = 0
    class_end = len(lines)

    if class_name:
        in_target_class = False
        class_brace_count = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Look for class declaration - handle various patterns
            if (f"class {class_name}" in line or f"class {class_name} " in line) and not in_target_class:
                class_start = i
                in_target_class = True
                class_brace_count = 0

            if in_target_class:
                # Count braces to find class boundaries
                class_brace_count += line.count('{') - line.count('}')
                if class_brace_count == 0 and i > class_start and '{' in lines[class_start:i+1]:
                    class_end = i + 1
                    break

    # Search for function within the class boundaries
    for i in range(class_start, class_end):
        line = lines[i]
        stripped = line.strip()

        # Look for method declaration with our function name
        # Handle various Java method patterns
        if (function_name in line and '(' in line and
            (any(modifier in line for modifier in ['public', 'private', 'protected', 'static', 'final']) or
             f" {function_name}(" in line or line.strip().startswith(f"{function_name}("))):

            # Extract the complete method including javadoc
            method_lines = []

            # Look backwards for javadoc comments
            javadoc_start = i
            for j in range(i - 1, max(class_start - 1, i - 20), -1):
                prev_line = lines[j].strip()
                if prev_line.startswith('/**') or '/**' in prev_line:
                    javadoc_start = j
                    break
                elif prev_line and not (prev_line.startswith('*') or prev_line.startswith('/') or prev_line.startswith('@')):
                    break

            # Collect method with javadoc
            brace_count = 0
            started_body = False

            for j in range(javadoc_start, class_end):
                method_line = lines[j]
                method_lines.append(method_line)

                # Count braces to determine method boundaries
                if '{' in method_line:
                    started_body = True
                brace_count += method_line.count('{') - method_line.count('}')

                if started_body and brace_count == 0:
                    break

            return '\n'.join(method_lines)

    return ""


def _extract_python_function_by_signature(code: str, function_name: str, class_name: str = None) -> str:
    """Extract Python function by signature using enhanced regex."""
    lines = code.split('\n')

    # If we have class name, first find the class boundaries
    class_start = 0
    class_end = len(lines)
    class_indent = 0

    if class_name:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(f"class {class_name}") or stripped.startswith(f"class {class_name}("):
                class_start = i
                class_indent = len(line) - len(line.lstrip())

                # Find end of class by indentation
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    if (next_line.strip() and
                        len(next_line) - len(next_line.lstrip()) <= class_indent and
                        not next_line.strip().startswith('#')):
                        class_end = j
                        break
                break

    # Search for function within class boundaries
    for i in range(class_start, class_end):
        line = lines[i]
        stripped = line.strip()

        # Look for function definition
        if (stripped.startswith(f"def {function_name}(") or
            stripped.startswith(f"async def {function_name}(")):

            # Extract the complete function including decorators and docstring
            func_lines = []
            func_indent = len(line) - len(line.lstrip())

            # Look backwards for decorators
            decorator_start = i
            for j in range(i - 1, max(class_start - 1, i - 10), -1):
                prev_line = lines[j]
                if prev_line.strip().startswith('@'):
                    decorator_start = j
                elif prev_line.strip() and not prev_line.strip().startswith('#'):
                    break

            # Collect function with decorators and docstring
            for j in range(decorator_start, class_end):
                func_line = lines[j]
                func_lines.append(func_line)

                # Check if we've reached the end of the function by indentation
                if (j > i and func_line.strip() and
                    len(func_line) - len(func_line.lstrip()) <= func_indent):
                    func_lines.pop()  # Remove the line that's not part of function
                    break

            return '\n'.join(func_lines)

    return ""



def extract_function_code_pairs(docker_env, work_dir: str, commit_hash: str, filepath: str, blamed_line: int, function_name: str = None, class_name: str = None) -> tuple:
    """
    Extract before/after function code pairs using Docker container.
    Uses signature-based search when function info available, falls back to line-based search.

    Args:
        docker_env: Docker environment instance
        work_dir: Working directory in container
        commit_hash: Git commit hash (blame commit)
        filepath: File path to analyze
        blamed_line: Line number that was blamed (1-based, from buggy commit)
        function_name: Function name from fault_locations (preferred approach)
        class_name: Class name from fault_locations (for Java)

    Returns:
        Tuple of (before_code, after_code) strings
    """
    try:
        file_extension = Path(filepath).suffix

        # Get both before and after versions
        before_commit = f"{commit_hash}~1"
        before_content = get_file_content_at_commit(docker_env, work_dir, before_commit, filepath)
        after_content = get_file_content_at_commit(docker_env, work_dir, commit_hash, filepath)

        before_code = ""
        after_code = ""

        # Signature-based search (should handle 99% of cases with function_name from fault_locations)
        if function_name:
            if before_content:
                before_code = extract_function_by_signature(before_content, function_name, class_name, file_extension)

            if after_content:
                after_code = extract_function_by_signature(after_content, function_name, class_name, file_extension)

            # Return signature-based results (even if one is empty)
            return before_code, after_code

        # No function signature available - cannot reliably extract function code
        # Note: Line-based search is unreliable because blamed_line refers to buggy commit,
        # but we're searching in different commit versions where line numbers have shifted
        return before_code, after_code

    except Exception as e:
        print(f"Error extracting function code pairs for {filepath}:{blamed_line}: {e}")
        return "", ""


def is_binary_file(docker_env, work_dir: str, commit_hash: str, filepath: str) -> bool:
    """
    Check if a file is binary by attempting to get its content using Docker container.
    
    Args:
        docker_env: Docker environment instance with execute method
        work_dir: Working directory in container
        commit_hash: Git commit hash
        filepath: File path to check
        
    Returns:
        True if file appears to be binary, False otherwise
    """
    try:
        content = get_file_content_at_commit(docker_env, work_dir, commit_hash, filepath)
        if content is None:
            return True  # Treat inaccessible files as binary
        
        # Check for null bytes (simple binary detection)
        return '\0' in content
    except:
        return True  # Treat error cases as binary


def extract_all_history_context(
        docker_env, work_dir: str, commit_hash: str, buggy_files: List[str],
        blamed_file: str = None, blamed_line: int = None,
        max_files: int = 50, exclude_tests: bool = True,
        function_name: str = None, class_name: str = None,
        buggy_to_blamed_mapping: Dict[str, str] = None
) -> Dict:
    """
    Extract all history context information for all categories with Docker container.

    Args:
        class_name:
        function_name:
        docker_env: Docker environment instance with execute method
        work_dir: Working directory in container
        commit_hash: Git commit hash (blame commit)
        buggy_files: List of buggy file paths from dataset instance
        blamed_file: Specific file that was blamed (for function code pairs and file patch)
        blamed_line: Specific line that was blamed (for function code pairs)
        max_files: Maximum number of files to process (to handle large commits)
        exclude_tests: Whether to exclude test files from processing
        buggy_to_blamed_mapping: Mapping from buggy file paths to their corresponding blamed file paths

    Returns:
        Dictionary containing all extracted history information
    """
    context = {
        'function': {
            'functions_name_co_evolved_modified_file': [],      # Category 2
            'functions_name_co_evolved_all_files': [],          # Category 3
            'functions_name_modified_file': [],                 # Category 4
            'functions_name_all_files': [],                     # Category 5
            'function_code_before': "",                         # Category 7
            'function_code_after': "",                          # Category 7
        },
        'file': {
            'files_name_in_blame_commit': [],                   # Category 6
            'file_patches': {}                                  # Category 8 - commit patches per file
        },
        'stats': {
            'total_files_in_commit': 0,
            'program_files_processed': 0,
            'files_skipped_non_program': 0,
            'files_skipped_binary': 0,
            'files_skipped_errors': 0,
            'files_limited_by_max': False
        }
    }
    
    try:
        # Category 6: Get all changed files in blame commit
        all_changed_files_raw = get_changed_files_in_commit(docker_env, work_dir, commit_hash)
        context['stats']['total_files_in_commit'] = len(all_changed_files_raw)
        
        # Filter to program files only
        all_changed_files = []
        for filepath in all_changed_files_raw:
            if is_program_file(filepath, exclude_tests):
                # Check if it's binary
                if not is_binary_file(docker_env, work_dir, commit_hash, filepath):
                    all_changed_files.append(filepath)
                else:
                    context['stats']['files_skipped_binary'] += 1
            else:
                context['stats']['files_skipped_non_program'] += 1
        
        # Apply file limit for performance
        if len(all_changed_files) > max_files:
            context['stats']['files_limited_by_max'] = True
            # Prioritize buggy files using blamed file paths for comparison
            prioritized_files = []
            for buggy_file in buggy_files:
                # Map buggy file to blamed file path for consistent comparison
                blamed_file_path = buggy_file  # Default fallback
                if buggy_to_blamed_mapping and buggy_file in buggy_to_blamed_mapping:
                    blamed_file_path = buggy_to_blamed_mapping[buggy_file]

                if blamed_file_path in all_changed_files:
                    prioritized_files.append(blamed_file_path)  # Use blamed path consistently
            
            # Add remaining files up to limit
            remaining_files = [f for f in all_changed_files if f not in prioritized_files]
            remaining_count = max_files - len(prioritized_files)
            if remaining_count > 0:
                prioritized_files.extend(remaining_files[:remaining_count])
            
            all_changed_files = prioritized_files[:max_files]
        
        context['file']['files_name_in_blame_commit'] = all_changed_files
        context['stats']['program_files_processed'] = len(all_changed_files)
        
        # Categories 2 & 4: Process only buggy files (modified files in dataset)
        for buggy_file in buggy_files:
            # Get the corresponding blamed file path for design consistency
            blamed_file_path = buggy_file  # Default fallback
            if buggy_to_blamed_mapping and buggy_file in buggy_to_blamed_mapping:
                blamed_file_path = buggy_to_blamed_mapping[buggy_file]

            if blamed_file_path in all_changed_files:  # Only if this blamed file was changed in blame commit
                try:

                    # Category 2: Co-evolved functions in modified files (use blamed path)
                    changed_funcs = get_changed_functions_in_commit(docker_env, work_dir, commit_hash, blamed_file_path)
                    context['function']['functions_name_co_evolved_modified_file'].extend(changed_funcs)

                    # Category 4: All functions in modified files at blame commit (use blamed path)
                    file_content = get_file_content_at_commit(docker_env, work_dir, commit_hash, blamed_file_path)
                    if file_content:
                        file_extension = Path(blamed_file_path).suffix
                        all_funcs = extract_function_names_from_code(file_content, file_extension)
                        context['function']['functions_name_modified_file'].extend(all_funcs)

                except Exception as e:
                    print(f"Error processing buggy file {buggy_file} -> {blamed_file_path}: {e}")
                    context['stats']['files_skipped_errors'] += 1
                    continue
        
        # Categories 3 & 5: Process all changed files in blame commit
        for changed_file in all_changed_files:
            try:
                # Category 3: Co-evolved functions in all files
                changed_funcs = get_changed_functions_in_commit(docker_env, work_dir, commit_hash, changed_file)
                context['function']['functions_name_co_evolved_all_files'].extend(changed_funcs)
                
                # Category 5: All functions in all files at blame commit
                file_content = get_file_content_at_commit(docker_env, work_dir, commit_hash, changed_file)
                if file_content:
                    file_extension = Path(changed_file).suffix
                    all_funcs = extract_function_names_from_code(file_content, file_extension)
                    context['function']['functions_name_all_files'].extend(all_funcs)
                    
            except Exception as e:
                print(f"Error processing changed file {changed_file}: {e}")
                context['stats']['files_skipped_errors'] += 1
                continue
        
        # Category 7: Function code pairs (before/after) - extract for specific blamed function
        # Early check: Only proceed if we have function information (our core motivation is function evolution)
        if blamed_file and blamed_line and blamed_file in all_changed_files and function_name:
            try:
                before_code, after_code = extract_function_code_pairs(
                    docker_env, work_dir, commit_hash, blamed_file, blamed_line,
                    function_name=function_name, class_name=class_name
                )
                context['function']['function_code_before'] = before_code
                context['function']['function_code_after'] = after_code
            except Exception as e:
                print(f"Error extracting function code pairs: {e}")
        elif blamed_file and blamed_line and blamed_file in all_changed_files and not function_name:
            # Bug is not inside a function - skip function code extraction (core motivation is function evolution)
            print(f"Skipping function code extraction: No function_name for {blamed_file}:{blamed_line}")
        
        # Category 8: File patch for only the blamed file (not all co-changed files)
        if blamed_file and blamed_file in all_changed_files:
            try:
                file_patch = get_commit_patch(docker_env, work_dir, commit_hash, blamed_file)
                if file_patch:
                    context['file']['file_patches'][blamed_file] = file_patch
            except Exception as e:
                print(f"Error getting patch for {blamed_file}: {e}")
        
        # Remove duplicates from all lists
        for key in context['function']:
            if isinstance(context['function'][key], list):
                context['function'][key] = list(set(context['function'][key]))
                
    except Exception as e:
        print(f"Error extracting history context: {e}")
        context['stats']['files_skipped_errors'] += 1
    
    return context


def extract_blame_context(patch_content: str, docker_env, work_dir: str,
                          patch_format: PatchFormat, selector_type: str = "first",
                          n_lines: int = 1, buggy_files: List[str] = None, bug_info: Dict = None,
                          include_blameless: bool = True, model_config: dict = None) -> Dict:
    """
    Container-based multi-line blame extraction with HAFixAgent's sophisticated analysis.

    Args:
        bug_info: For function extraction in blame commit
        patch_content: Patch content to analyze
        docker_env: Docker environment instance
        work_dir: Working directory in container
        selector_type: Line selection strategy
        n_lines: Number of lines to select
        patch_format: Patch format (defaults to Defects4J)
        buggy_files: List of buggy files for context analysis
        include_blameless: If True, use pre-hunk lines as fallback for blameless patches
        model_config: Model configuration for LLMJudgeSelector

    Returns:
        Aggregated blame context from multiple lines with sophisticated analysis
    """

    # 1. Parse patch to extract all blamable lines from code files
    parser = PatchParser(patch_format=patch_format)
    all_lines = parser.extract_code_file_lines(patch_content)
    blamable_lines = parser.get_blamable_lines(all_lines)
    
    if not blamable_lines:
        if include_blameless:
            print("No blamable lines found in patch, trying pre-hunk fallback lines")
            # Try to extract pre-hunk lines as fallback for blameless patches
            try:
                pre_hunk_lines = parser.extract_pre_hunk_blameable_lines(patch_content, docker_env, work_dir)
                if pre_hunk_lines:
                    print(f"Found {len(pre_hunk_lines)} pre-hunk fallback lines")
                    blamable_lines = pre_hunk_lines
                else:
                    print("No suitable pre-hunk lines found")
                    return {}
            except Exception as e:
                print(f"Error extracting pre-hunk lines: {e}")
                return {}
        else:
            print("No blamable lines found in patch")
            return {}
    else:
        print(f"Found {len(blamable_lines)} blamable lines in patch")
    
    # 2. Select top N lines to blame using specified strategy
    selector = get_selector(selector_type, model_config)
    # Debug: make sure the blamable_lines are from multi hunks when it it a MFMH bug
    selected_lines = selector.select(blamable_lines, n_lines)
    
    print(f"Selected {len(selected_lines)} lines to blame using '{selector_type}' strategy:")
    for line in selected_lines:
        print(f"  {line}")
    
    # 3. Run blame extraction on each selected line with deduplication
    blame_contexts = []
    seen_commits = set()  # Track commit hashes to avoid duplicates
    
    for line in selected_lines:
        try:
            # Run git blame on this specific line using Docker
            blame_info = run_git_blame(docker_env, work_dir, line.file_path, line.line_number)
            if not blame_info:
                print(f"Git blame failed for {line.file_path}:{line.line_number}")
                continue
            
            commit_hash = blame_info['commit_hash']
            
            # Skip if we've already processed this commit
            if commit_hash in seen_commits:
                print(f"Skipping duplicate commit {commit_hash[:8]} for {line.file_path}:{line.line_number}")
                continue
            
            seen_commits.add(commit_hash)
            
            # Use the blamed file path from git blame output (handles renames/moves)
            blamed_file_in_commit = blame_info.get('blamed_file_path', line.file_path)

            # Extract history context for this blame result using Docker
            # Use provided buggy_files or fallback to the blamed file
            context_buggy_files = buggy_files if buggy_files else [line.file_path]

            # Create mapping: only map the specific blamed file, others use same path
            buggy_to_blamed_path_mapping = {}
            for buggy_file in context_buggy_files:
                if buggy_file == line.file_path:
                    # This is the specific file we blamed - use the actual blamed path from git blame
                    buggy_to_blamed_path_mapping[buggy_file] = blamed_file_in_commit
                else:
                    # For other files, assume same path (git operations will handle gracefully if path doesn't exist)
                    buggy_to_blamed_path_mapping[buggy_file] = buggy_file

            # Extract function information from bug_info for this specific line
            function_name = None
            class_name = None
            if bug_info and 'fault_locations' in bug_info:
                for fault_location in bug_info['fault_locations']:
                    if (fault_location.get('file') == line.file_path and
                        fault_location.get('start_line', 0) <= line.line_number <= fault_location.get('end_line', float('inf'))):
                        function_name = fault_location.get('function_name')
                        class_name = fault_location.get('class_name')
                        break

            history_context = extract_all_history_context(
                docker_env,
                work_dir,
                blame_info['commit_hash'],
                context_buggy_files,
                blamed_file=blamed_file_in_commit,
                blamed_line=line.line_number,
                function_name=function_name,
                class_name=class_name,
                buggy_to_blamed_mapping=buggy_to_blamed_path_mapping
            )
            
            if history_context:
                blame_contexts.append({
                    'commit_hash': commit_hash,
                    'blame_info': blame_info,
                    'history_context': history_context,
                    'source_line': line
                })
                print(f"Added unique commit {commit_hash[:8]} from {line.file_path}:{line.line_number}")
                
        except Exception as e:
            print(f"Error processing line {line.file_path}:{line.line_number}: {e}")
            continue
    
    # 4. Return individual blame contexts without aggregation
    if not blame_contexts:
        print("No successful blame operations")
        return {}
    
    print(f"Successfully blamed {len(blame_contexts)} lines with {len(blame_contexts)} unique commits")
    
    # Return multiple individual commits instead of aggregating
    if len(blame_contexts) == 1:
        # Single commit - return in legacy format for compatibility
        one_blame_context = blame_contexts[0]
        return {
            'blame_commit': {
                'commit': {
                    'commit_author': one_blame_context['blame_info'].get('commit_author', 'Unknown'),
                    'commit_date': one_blame_context['blame_info'].get('commit_timestamp', 'Unknown'),
                    'commit_message': one_blame_context['blame_info'].get('commit_message', 'No message'),
                    'commit_hash': one_blame_context['blame_info'].get('commit_hash')
                },
                'function': one_blame_context['history_context']['function'],
                'file': one_blame_context['history_context']['file']
            }
        }
    else:
        # Multiple commits - preserve them separately
        return {
            'blame_commits': [
                {
                    'commit': {
                        'commit_author': one_blame_context['blame_info'].get('commit_author', 'Unknown'),
                        'commit_date': one_blame_context['blame_info'].get('commit_timestamp', 'Unknown'),
                        'commit_message': one_blame_context['blame_info'].get('commit_message', 'No message'),
                        'commit_hash': one_blame_context['blame_info'].get('commit_hash')
                    },
                    'function': one_blame_context['history_context']['function'],
                    'file': one_blame_context['history_context']['file'],
                    'source_line': f"{one_blame_context['source_line'].file_path}:{one_blame_context['source_line'].line_number}"
                }
                for one_blame_context in blame_contexts
            ],
            'commit_count': len(blame_contexts)
        }


