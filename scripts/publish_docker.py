"""
Docker Hub Publication Script

Build and publish Docker images to Docker Hub.

Usage:
    python scripts/publish_docker.py --build                  # Build image locally
    python scripts/publish_docker.py --test                   # Build and test image
    python scripts/publish_docker.py --push                   # Push to Docker Hub
    python scripts/publish_docker.py --all                    # Build, test, and push
    python scripts/publish_docker.py --registry ghcr.io       # Use GitHub Container Registry
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


class DockerPublisher:
    """Handles Docker image building and publishing."""

    def __init__(
        self,
        project_root: Path = Path.cwd(),
        registry: str = "docker.io",
        organization: str = "sigmalang",
        image_name: str = "sigmalang"
    ):
        self.project_root = project_root
        self.registry = registry
        self.organization = organization
        self.image_name = image_name
        self.dockerfile = project_root / "Dockerfile.prod"

        # Get version from pyproject.toml
        self.version = self.get_version()

    def get_version(self) -> str:
        """Extract version from pyproject.toml."""
        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists():
            return "latest"

        import re
        content = pyproject_path.read_text()
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        return match.group(1) if match else "latest"

    def get_full_image_name(self, tag: str = None) -> str:
        """Get full image name with registry and tag."""
        tag = tag or self.version
        if self.registry == "docker.io":
            return f"{self.organization}/{self.image_name}:{tag}"
        else:
            return f"{self.registry}/{self.organization}/{self.image_name}:{tag}"

    def run_command(
        self,
        cmd: List[str],
        description: str,
        capture_output: bool = False
    ) -> Tuple[bool, str]:
        """Run a command and return success status."""
        print(f"\n{description}...")
        print(f"Command: {' '.join(cmd)}\n")

        try:
            if capture_output:
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=True
                )
                output = result.stdout
            else:
                subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    check=True
                )
                output = ""

            print(f"[PASS] {description} successful")
            return True, output

        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {description} failed")
            if capture_output:
                print(f"Error: {e.stderr}")
            return False, ""

    def build_image(self, target: str = "production", tags: Optional[List[str]] = None) -> bool:
        """Build Docker image."""
        print("=" * 70)
        print("BUILDING DOCKER IMAGE")
        print("=" * 70)

        if not self.dockerfile.exists():
            print(f"[FAIL] Dockerfile not found: {self.dockerfile}")
            return False

        # Default tags: version and latest
        if tags is None:
            tags = [self.version, "latest"]

        # Build tag arguments
        tag_args = []
        for tag in tags:
            image_name = self.get_full_image_name(tag)
            tag_args.extend(["-t", image_name])

        cmd = [
            "docker", "build",
            "-f", str(self.dockerfile),
            "--target", target,
            *tag_args,
            "."
        ]

        success, _ = self.run_command(
            cmd,
            f"Building Docker image (target: {target})"
        )

        if success:
            print("\nBuilt images:")
            for tag in tags:
                print(f"  - {self.get_full_image_name(tag)}")

        return success

    def test_image(self, tag: str = None) -> bool:
        """Test the built Docker image."""
        print("\n" + "=" * 70)
        print("TESTING DOCKER IMAGE")
        print("=" * 70)

        image_name = self.get_full_image_name(tag)

        # Test 1: Inspect image
        print("\n1. Inspecting image metadata...")
        cmd = ["docker", "inspect", image_name]
        success, output = self.run_command(cmd, "Inspecting image", capture_output=True)

        if not success:
            return False

        # Test 2: Check image size
        print("\n2. Checking image size...")
        cmd = ["docker", "images", image_name, "--format", "{{.Size}}"]
        success, output = self.run_command(cmd, "Getting image size", capture_output=True)

        if success:
            size = output.strip()
            print(f"Image size: {size}")

        # Test 3: Run container (health check)
        print("\n3. Testing container startup...")
        container_name = f"sigmalang-test-{int(time.time())}"

        # Start container in background
        cmd = [
            "docker", "run",
            "--name", container_name,
            "--rm",
            "-d",
            "-p", "8001:8001",
            "-p", "9091:9091",
            "-e", "SIGMALANG_WORKERS=1",  # Single worker for testing
            image_name
        ]

        success, container_id = self.run_command(
            cmd,
            "Starting test container",
            capture_output=True
        )

        if not success:
            return False

        container_id = container_id.strip()

        try:
            # Wait for container to be healthy
            print("\nWaiting for container health check...")
            time.sleep(10)

            # Check if container is still running
            cmd = ["docker", "ps", "-q", "-f", f"id={container_id}"]
            success, output = self.run_command(
                cmd,
                "Checking container status",
                capture_output=True
            )

            if output.strip():
                print("[PASS] Container is running")

                # Check logs
                print("\nContainer logs (last 20 lines):")
                cmd = ["docker", "logs", "--tail", "20", container_id]
                subprocess.run(cmd)

                result = True
            else:
                print("[FAIL] Container exited")
                result = False

        finally:
            # Cleanup: stop container
            print("\nCleaning up test container...")
            subprocess.run(["docker", "stop", container_id], capture_output=True)

        return result

    def push_images(self, tags: Optional[List[str]] = None) -> bool:
        """Push images to registry."""
        print("\n" + "=" * 70)
        print(f"PUSHING TO {self.registry.upper()}")
        print("=" * 70)

        if tags is None:
            tags = [self.version, "latest"]

        print(f"\nRegistry: {self.registry}")
        print(f"Organization: {self.organization}")
        print(f"Image: {self.image_name}")
        print(f"Tags: {', '.join(tags)}\n")

        # Check if logged in
        if self.registry == "docker.io":
            print("NOTE: Ensure you're logged in to Docker Hub:")
            print("  docker login")
        elif self.registry.startswith("ghcr.io"):
            print("NOTE: Ensure you're logged in to GitHub Container Registry:")
            print("  echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin")

        print()
        response = input("Are you logged in and ready to push? (yes/no): ")
        if response.lower() != "yes":
            print("\n[CANCELLED] Push cancelled")
            return False

        success = True
        for tag in tags:
            image_name = self.get_full_image_name(tag)
            cmd = ["docker", "push", image_name]

            tag_success, _ = self.run_command(
                cmd,
                f"Pushing {image_name}"
            )

            if not tag_success:
                success = False

        if success:
            print("\n" + "=" * 70)
            print("[PASS] All images pushed successfully!")
            print("=" * 70)
            print("\nImages available at:")
            for tag in tags:
                print(f"  - {self.get_full_image_name(tag)}")

            if self.registry == "docker.io":
                print(f"\nDocker Hub: https://hub.docker.com/r/{self.organization}/{self.image_name}")
            elif self.registry.startswith("ghcr.io"):
                print(f"\nGitHub Packages: https://github.com/{self.organization}/{self.image_name}/pkgs/container/{self.image_name}")

        return success

    def tag_additional(self, source_tag: str, new_tags: List[str]) -> bool:
        """Tag an existing image with additional tags."""
        print("\n" + "=" * 70)
        print("CREATING ADDITIONAL TAGS")
        print("=" * 70)

        source_image = self.get_full_image_name(source_tag)

        for new_tag in new_tags:
            target_image = self.get_full_image_name(new_tag)

            cmd = ["docker", "tag", source_image, target_image]
            success, _ = self.run_command(
                cmd,
                f"Tagging {source_tag} as {new_tag}"
            )

            if not success:
                return False

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Docker Hub Publication Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--build',
        action='store_true',
        help='Build Docker image'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Test built Docker image'
    )

    parser.add_argument(
        '--push',
        action='store_true',
        help='Push images to registry'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Build, test, and push'
    )

    parser.add_argument(
        '--registry',
        type=str,
        default='docker.io',
        help='Docker registry (default: docker.io, options: ghcr.io)'
    )

    parser.add_argument(
        '--organization',
        type=str,
        default='sigmalang',
        help='Organization/username (default: sigmalang)'
    )

    parser.add_argument(
        '--tags',
        type=str,
        help='Comma-separated list of tags (default: version,latest)'
    )

    parser.add_argument(
        '--target',
        type=str,
        default='production',
        choices=['production', 'debug'],
        help='Build target (default: production)'
    )

    args = parser.parse_args()

    # Parse tags
    tags = args.tags.split(',') if args.tags else None

    publisher = DockerPublisher(
        registry=args.registry,
        organization=args.organization
    )

    print(f"\nDocker Image: {publisher.get_full_image_name()}")
    print(f"Version: {publisher.version}\n")

    # Default to --build if no arguments
    if not any([args.build, args.test, args.push, args.all]):
        args.build = True

    success = True

    if args.all:
        # Full workflow
        success = publisher.build_image(target=args.target, tags=tags)
        if success:
            success = publisher.test_image()
        if success:
            success = publisher.push_images(tags=tags)
    else:
        if args.build:
            success = publisher.build_image(target=args.target, tags=tags)

        if args.test and success:
            success = publisher.test_image()

        if args.push and success:
            success = publisher.push_images(tags=tags)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
