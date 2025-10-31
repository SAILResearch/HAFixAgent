"""
Defects4J Docker environment wrapper.
Extends mini-swe-agent's DockerEnvironment with Defects4J-specific defaults.
"""

import subprocess
import uuid
from typing import Dict, Any

from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
from dataclasses import dataclass, field


@dataclass
class Defects4JDockerConfig(DockerEnvironmentConfig):
    """Configuration for Defects4J Docker environment."""

    # Override defaults for Defects4J
    image: str = "defects4j:latest"
    cwd: str = "/defects4j"
    timeout: int = 300  # 5 minutes for tests
    container_timeout: str = "1h"

    # Defects4J environment variables
    env: dict[str, str] = field(default_factory=lambda: {
        "JAVA_HOME": "/usr/lib/jvm/java-11-openjdk-amd64",
        "DEFECTS4J_HOME": "/defects4j",
        "PATH": "/defects4j/framework/bin:/usr/lib/jvm/java-11-openjdk-amd64/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "_JAVA_OPTIONS": "-Xmx4g -XX:MaxPermSize=512m",
        "DEBIAN_FRONTEND": "noninteractive",
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R"
    })


class Defects4JDocker(DockerEnvironment):
    """
    Defects4J Docker environment for container-based bug analysis and repair.
    """

    def __init__(self, use_existing_container: str = None, cleanup_on_exit: bool = True, **kwargs):
        """Initialize with Defects4J defaults and optional existing container."""
        # Set existing container name first
        self.existing_container_name = use_existing_container
        self.cleanup_on_exit = cleanup_on_exit
        
        # Initialize parent class
        super().__init__(config_class=Defects4JDockerConfig, **kwargs)
        
        # If using existing container, override the container_id
        if use_existing_container:
            self.container_id = use_existing_container
    
    def _start_container(self):
        """Override to use HAFixAgent container naming and give proper credit."""
        if self.existing_container_name:
            # Use existing container, don't start new one
            self.container_id = self.existing_container_name
            self.logger.info(f"Using existing container: {self.existing_container_name}")
        else:
            # Start new container with HAFixAgent naming
            container_name = f"hafixagent-{uuid.uuid4().hex[:8]}"
            cmd = [
                self.config.executable,
                "run",
                "-d",
                "--name",
                container_name,
                "-w",
                self.config.cwd,
                *self.config.run_args,
                self.config.image,
                "sleep",
                self.config.container_timeout,
            ]
            
            result = subprocess.run(
                cmd,
                text=True,
                timeout=60,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                msg = f"Failed to start container: {result.stderr}"
                self.logger.error(msg)
                raise RuntimeError(msg)
            
            self.logger.info(f"Started HAFixAgent container {container_name} with ID {result.stdout.strip()}")
            self.container_id = result.stdout.strip()
    
    def cleanup(self):
        """Override cleanup to respect cleanup_on_exit setting."""
        if hasattr(self, 'cleanup_on_exit') and not self.cleanup_on_exit:
            # Don't cleanup - keep container for debugging/reuse
            self.logger.info(f"Keeping container {getattr(self, 'container_id', 'unknown')} for debugging")
            return
        
        # Use parent cleanup (removes container)
        super().cleanup()
        
    def force_cleanup(self):
        """Force cleanup regardless of cleanup_on_exit setting."""
        super().cleanup()
        
    def checkout_bug(self, project: str, bug_id: int, work_dir: str = None) -> Dict[str, Any]:
        """
        Convenience method to checkout a Defects4J bug.

        Args:
            project: Project name (e.g., "Lang")
            bug_id: Bug number (e.g., 1)
            work_dir: Working directory path (if None, uses default pattern)

        Returns:
            Execution result
        """
        if work_dir is None:
            work_dir = f"/defects4j/framework/bin/temp/{project}_{bug_id}"
        cmd = f"defects4j checkout -p {project} -v {bug_id}b -w {work_dir}"
        return self.execute(cmd)