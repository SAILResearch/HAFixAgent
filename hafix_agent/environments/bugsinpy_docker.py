"""
BugsInPy Docker environment wrapper.
Extends mini-swe-agent's DockerEnvironment with BugsInPy-specific defaults.

Key differences from Defects4J:
- Python-based projects (no JAVA_HOME)
- bugsinpy-checkout -p <project> -i <bug_id> -v 0 (buggy version)
- bugsinpy-compile (install deps) required before running tests
- bugsinpy-test instead of defects4j test
- Pre-warmed images available: bugsinpy:{project}_{bug_id}
"""

import subprocess
import time
import uuid
from typing import Dict, Any
from dataclasses import dataclass, field

from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig


@dataclass
class BugsInPyDockerConfig(DockerEnvironmentConfig):
    """Configuration for BugsInPy Docker environment."""

    image: str = "bugsinpy_image:clean"
    cwd: str = "/BugsInPy"
    timeout: int = 300  # 5 minutes for tests
    container_timeout: str = "1h"

    env: dict[str, str] = field(default_factory=lambda: {
        "BUGSINPY_HOME": "/BugsInPy",
        "PATH": "/BugsInPy/framework/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
    })


class BugsInPyDocker(DockerEnvironment):
    """
    BugsInPy Docker environment for container-based Python bug repair.
    Provides the same interface as Defects4JDocker.
    """

    def __init__(self, use_existing_container: str = None, cleanup_on_exit: bool = True, **kwargs):
        self.existing_container_name = use_existing_container
        self.cleanup_on_exit = cleanup_on_exit
        super().__init__(config_class=BugsInPyDockerConfig, **kwargs)
        if use_existing_container:
            self.container_id = use_existing_container

    def _start_container(self):
        """Override to use HAFixAgent container naming."""
        if self.existing_container_name:
            self.container_id = self.existing_container_name
            self.logger.info(f"Using existing container: {self.existing_container_name}")
        else:
            # Retry container creation: under high concurrency `docker run` can exceed
            # the timeout (saturated daemon); a short backoff lets the burst subside.
            # On timeout, force-remove the half-started container so it does not leak.
            last_err = None
            for attempt in range(3):
                container_name = f"hafixagent-bip-{uuid.uuid4().hex[:8]}"
                cmd = [
                    self.config.executable,
                    "run", "-d",
                    "--name", container_name,
                    "-w", self.config.cwd,
                    *self.config.run_args,
                    self.config.image,
                    "sleep", self.config.container_timeout,
                ]
                try:
                    result = subprocess.run(
                        cmd, text=True, timeout=120,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                except subprocess.TimeoutExpired as e:
                    last_err = e
                    try:  # the daemon may still be starting it -> remove the leak
                        subprocess.run([self.config.executable, "rm", "-f", container_name],
                                       capture_output=True, timeout=30)
                    except Exception:
                        pass
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Failed to start container after retries: {e}")
                if result.returncode != 0:
                    msg = f"Failed to start container: {result.stderr}"
                    self.logger.error(msg)
                    raise RuntimeError(msg)
                self.logger.info(f"Started BugsInPy container {container_name}")
                self.container_id = result.stdout.strip()
                return

    def cleanup(self):
        """Override cleanup to remove container with volumes (-v) and respect cleanup_on_exit."""
        if hasattr(self, 'cleanup_on_exit') and not self.cleanup_on_exit:
            self.logger.info(f"Keeping container {getattr(self, 'container_id', 'unknown')} for debugging")
            return
        if getattr(self, "container_id", None):
            cmd = f"(timeout 60 {self.config.executable} stop {self.container_id} || true) && {self.config.executable} rm -f -v {self.container_id}"
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute command and strip 'mesg: ttyname failed' noise from bash -l in Docker."""
        result = super().execute(command, cwd, timeout=timeout)
        output = result.get("output", "")
        # BugsInPy Docker images produce "mesg: ttyname failed" noise on bash -l
        if output.startswith("mesg:"):
            output = output.split("\n", 1)[1] if "\n" in output else ""
            result["output"] = output
        return result

    def force_cleanup(self):
        """Force cleanup regardless of cleanup_on_exit setting."""
        super().cleanup()

    def checkout_bug(self, project: str, bug_id: int, work_dir: str = None) -> Dict[str, Any]:
        """Checkout a BugsInPy bug (buggy version).

        Args:
            project: Project name (e.g., "pandas")
            bug_id: Bug number (e.g., 1)
            work_dir: Working directory path (if None, uses default pattern)

        Returns:
            Execution result
        """
        if work_dir is None:
            work_dir = f"/BugsInPy/framework/bin/temp/{project}_{bug_id}"
        cmd = f"bugsinpy-checkout -p {project} -i {bug_id} -v 0 -w {work_dir}"
        return self.execute(cmd)

    def compile_bug(self, work_dir: str) -> Dict[str, Any]:
        """Install project dependencies after checkout.

        BugsInPy requires bugsinpy-compile before tests can run.
        """
        return self.execute(f"cd {work_dir} && bugsinpy-compile", timeout=1800)

    def run_tests(self, work_dir: str) -> Dict[str, Any]:
        """Run bug-specific tests using bugsinpy-test."""
        return self.execute(f"cd {work_dir} && bugsinpy-test", timeout=300)
