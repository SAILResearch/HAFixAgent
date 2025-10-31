import csv
import os
from typing import Optional, Tuple, Dict, Any
from hafix_agent.environments.defects4j_docker import Defects4JDocker


def get_defects4j_work_dir(project_name: str, bug_id: str) -> str:
    """Get consistent work directory path for Defects4J bugs."""
    return f"/defects4j/framework/bin/temp/{project_name}_{bug_id}"


def get_chart_blame_dir(bug_id: str) -> str:
    """Get Chart blame directory path for special Chart project handling."""
    return f"/tmp/Chart_blame/Chart_{bug_id}"


def get_chart_temp_dir(bug_id: str) -> str:
    """Get Chart temporary directory path for commit message extraction."""
    return f"/tmp/Chart/Chart_{bug_id}"


defects4j_project_name_repository_map = {
    'Chart': 'jfreechart',
    'Cli': 'commons-cli',
    'Closure': 'closure-compiler',
    'Codec': 'commons-codec',
    'Collections': 'commons-collections',
    'Compress': 'commons-compress',
    'Csv': 'commons-csv',
    'Gson': 'gson',
    'JacksonCore': 'jackson-core',
    'JacksonDatabind': 'jackson-databind',
    'JacksonXml': 'jackson-dataformat-xml',
    'Jsoup': 'jsoup',
    'JxPath': 'commons-jxpath',
    'Lang': 'commons-lang',
    'Math': 'commons-math',
    'Mockito': 'mockito',
    'Time': 'joda-time'
}


defects4j_project_name_url_map = {
    'Chart': 'https://github.com/jfree/jfreechart.git',
    'Cli': 'https://github.com/apache/commons-cli.git',
    'Closure': 'https://github.com/google/closure-compiler.git',
    'Codec': 'https://github.com/apache/commons-codec.git',
    'Collections': 'https://github.com/apache/commons-collections.git',
    'Compress': 'https://github.com/apache/commons-compress.git',
    'Csv': 'https://github.com/apache/commons-csv.git',
    'Gson': 'https://github.com/google/gson.git',
    'JacksonCore': 'https://github.com/FasterXML/jackson-core.git',
    'JacksonDatabind': 'https://github.com/FasterXML/jackson-databind.git',
    'JacksonXml': 'https://github.com/FasterXML/jackson-dataformat-xml.git',
    'Jsoup': 'https://github.com/jhy/jsoup.git',
    'JxPath': 'https://github.com/apache/commons-jxpath.git',
    'Lang': 'https://github.com/apache/commons-lang.git',
    'Math': 'https://github.com/apache/commons-math.git',
    'Mockito': 'https://github.com/mockito/mockito.git',
    'Time': 'https://github.com/JodaOrg/joda-time.git'
}

# defects4j_project_bug_description_example_map = {
#     'Chart': 'https://sourceforge.net/p/jfreechart/bugs/983',
#     'Cli': 'https://issues.apache.org/jira/browse/CLI-13',
#     'Closure': 'https://storage.googleapis.com/google-code-archive/v2/code.google.com/closure-compiler/issues/issue-884.json',
#     'Codec': 'https://issues.apache.org/jira/browse/CODEC-65',
#     'Collections': 'https://issues.apache.org/jira/browse/COLLECTIONS-586',
#     'Compress': 'https://issues.apache.org/jira/browse/COMPRESS-171',
#     'Csv': 'https://issues.apache.org/jira/browse/CSV-224',
#     'Gson': 'https://github.com/google/gson/issues/40',
#     'JacksonCore': 'https://github.com/FasterXML/jackson-core/issues/531',
#     'JacksonDatabind': 'https://github.com/FasterXML/jackson-core/issues/531',
#     'JacksonXml': 'https://github.com/FasterXML/jackson-dataformat-xml/issues/204',
#     'Jsoup': 'https://github.com/jhy/jsoup/issues/23',
#     'JxPath': 'https://github.com/jhy/jsoup/issues/23',
#     'Lang': 'https://issues.apache.org/jira/browse/LANG-747',
#     'Math': 'https://issues.apache.org/jira/browse/MATH-934',
#     'Mockito': 'https://github.com/mockito/mockito/issues/188',
#     'Time': 'https://github.com/JodaOrg/joda-time/issues/21'
# }
# 'Mockito': 'https://code.google.com/archive/p/mockito/issues/484',

def get_active_bugs(base_path):
    projects_bugs = {}

    # Iterate over each project directory in the given directory
    for project_name in os.listdir(base_path):
        project_path = os.path.join(base_path, project_name)
        if os.path.isdir(project_path):
            csv_file_path = os.path.join(project_path, 'active-bugs.csv')
            if os.path.exists(csv_file_path):
                with open(csv_file_path, mode='r', newline='') as file:
                    reader = csv.DictReader(file)
                    bug_ids = [int(row['bug.id']) for row in reader]
                    projects_bugs[project_name] = bug_ids

    return projects_bugs


def ensure_defects4j_docker_container(
    project_name: str,
    bug_id: str,
    docker_env=None,
    image: str = "defects4j:latest",
    use_existing_container: str = None,
    cleanup_on_exit: bool = True
) -> Tuple[Optional[Any], Optional[Dict]]:
    """
    Ensure Docker container is ready with Defects4J project checked out. 
    Returns (docker_env, error_dict_or_None).
    
    Args:
        project_name: Defects4J project name
        bug_id: Bug ID string
        docker_env: Optional existing Docker container to reuse
        image: Docker image name
        use_existing_container: Name of existing container to reuse
        cleanup_on_exit: Whether to cleanup container on exit
        
    Returns:
        Tuple of (docker_env, error_dict). If successful, error_dict is None.
        If failed, docker_env is None and error_dict contains error info.
    """

    # Use shared container if provided, otherwise create new one
    if docker_env:
        print(f"Reusing shared Docker container: {project_name}_{bug_id}")
        # Container should already be checked out by the orchestration layer
        return docker_env, None
    else:
        print(f"Creating new Docker container: {project_name}_{bug_id}")
        # 1. Initialize Docker environment
        docker_env = Defects4JDocker(
            image=image,
            use_existing_container=use_existing_container,
            cleanup_on_exit=cleanup_on_exit
        )
        
        # 2. Setup Defects4J checkout in container
        container_work_dir = get_defects4j_work_dir(project_name, bug_id)
        checkout_result = docker_env.execute(
            f"defects4j checkout -p {project_name} -v {bug_id}b -w {container_work_dir}"
        )
        
        if checkout_result['returncode'] != 0:
            error_dict = {
                "bug_info": None, 
                "blame_info": None, 
                "error": f"Failed to checkout {project_name}_{bug_id}: {checkout_result['output']}"
            }
            return None, error_dict
        
        return docker_env, None
