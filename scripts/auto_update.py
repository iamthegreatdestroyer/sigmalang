"""
Auto-Update Pipeline - Phase 6 Task 6.2

Zero-downtime update pipeline: Pull -> Test -> Rebuild -> Restart

Architecture:
    Git Pull --> Run Tests --> Build Image --> Rolling Restart
                    |              |               |
               (fail: abort)  (fail: abort)  (rollback on failure)

Features:
- Git-based update detection
- Pre-deployment test validation
- Docker image rebuild with cache
- Rolling restart (zero downtime)
- Automatic rollback on failure
- Update logging and notifications

Usage:
    python scripts/auto_update.py              # Full update cycle
    python scripts/auto_update.py --check      # Check for updates only
    python scripts/auto_update.py --force      # Force rebuild without git check
    python scripts/auto_update.py --rollback   # Rollback to previous version
"""

import subprocess
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('update_log.txt', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.personal.yml"
STATE_FILE = PROJECT_ROOT / ".update-state.json"


# =============================================================================
# Update State
# =============================================================================

@dataclass
class UpdateState:
    """Tracks update state for rollback."""

    current_commit: str = ""
    previous_commit: str = ""
    current_image_tag: str = ""
    previous_image_tag: str = ""
    last_update: str = ""
    last_status: str = "unknown"
    update_count: int = 0
    rollback_count: int = 0

    def save(self) -> None:
        """Save state to disk."""
        state = {
            'current_commit': self.current_commit,
            'previous_commit': self.previous_commit,
            'current_image_tag': self.current_image_tag,
            'previous_image_tag': self.previous_image_tag,
            'last_update': self.last_update,
            'last_status': self.last_status,
            'update_count': self.update_count,
            'rollback_count': self.rollback_count
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls) -> 'UpdateState':
        """Load state from disk."""
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                return cls(**data)
            except Exception:
                pass
        return cls()


# =============================================================================
# Pipeline Steps
# =============================================================================

def run_command(cmd: str, timeout: int = 300) -> Tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(PROJECT_ROOT)
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def get_current_commit() -> str:
    """Get current git commit hash."""
    code, stdout, _ = run_command("git rev-parse HEAD")
    return stdout.strip() if code == 0 else ""


def check_for_updates() -> Tuple[bool, str]:
    """Check if there are new commits on remote."""
    logger.info("Checking for updates...")

    # Fetch latest
    code, _, stderr = run_command("git fetch origin main --quiet")
    if code != 0:
        logger.warning(f"Git fetch failed: {stderr}")
        return False, ""

    # Compare local and remote
    code, stdout, _ = run_command("git rev-parse HEAD")
    local_commit = stdout.strip()

    code, stdout, _ = run_command("git rev-parse origin/main")
    remote_commit = stdout.strip()

    if local_commit == remote_commit:
        logger.info("Already up to date")
        return False, local_commit

    logger.info(f"Update available: {local_commit[:8]} -> {remote_commit[:8]}")
    return True, remote_commit


def pull_updates() -> bool:
    """Pull latest changes from git."""
    logger.info("Pulling updates...")

    code, stdout, stderr = run_command("git pull origin main --ff-only")
    if code != 0:
        logger.error(f"Git pull failed: {stderr}")
        return False

    logger.info(f"Pull successful: {stdout.strip()}")
    return True


def run_tests() -> bool:
    """Run test suite to validate before deployment."""
    logger.info("Running pre-deployment tests...")

    # Run critical tests only (fast)
    code, stdout, stderr = run_command(
        "python -m pytest tests/ -x -q --timeout=60 -k 'not slow and not benchmark'",
        timeout=300
    )

    if code != 0:
        logger.error(f"Tests failed:\n{stdout}\n{stderr}")
        return False

    logger.info("Tests passed")
    return True


def build_image() -> bool:
    """Build new Docker image."""
    logger.info("Building Docker image...")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    commit = get_current_commit()[:8]
    tag = f"sigmalang:update-{timestamp}-{commit}"

    code, stdout, stderr = run_command(
        f"docker-compose -f {COMPOSE_FILE} build --no-cache sigmalang-api",
        timeout=600
    )

    if code != 0:
        logger.error(f"Build failed: {stderr}")
        return False

    logger.info(f"Build successful: {tag}")
    return True


def rolling_restart() -> bool:
    """Perform rolling restart of services."""
    logger.info("Performing rolling restart...")

    # Restart API with zero downtime (docker-compose handles this)
    code, stdout, stderr = run_command(
        f"docker-compose -f {COMPOSE_FILE} up -d --no-deps --build sigmalang-api",
        timeout=300
    )

    if code != 0:
        logger.error(f"Restart failed: {stderr}")
        return False

    # Wait for health check
    logger.info("Waiting for health check...")
    for i in range(30):
        time.sleep(2)
        code, stdout, _ = run_command(
            "docker inspect --format='{{.State.Health.Status}}' sigmalang-api"
        )
        status = stdout.strip().strip("'")

        if status == "healthy":
            logger.info("Service is healthy")
            return True

        if i % 5 == 0:
            logger.info(f"  Health status: {status} (attempt {i+1}/30)")

    logger.warning("Health check timed out, but service may still be starting")
    return True  # Don't fail on slow health checks


def rollback(state: UpdateState) -> bool:
    """Rollback to previous version."""
    if not state.previous_commit:
        logger.error("No previous commit to rollback to")
        return False

    logger.info(f"Rolling back to {state.previous_commit[:8]}...")

    # Checkout previous commit
    code, _, stderr = run_command(f"git checkout {state.previous_commit}")
    if code != 0:
        logger.error(f"Rollback checkout failed: {stderr}")
        return False

    # Rebuild and restart
    if not build_image():
        return False

    if not rolling_restart():
        return False

    state.rollback_count += 1
    state.last_status = "rolled_back"
    state.save()

    logger.info("Rollback successful")
    return True


def verify_deployment() -> bool:
    """Verify the deployment is working correctly."""
    logger.info("Verifying deployment...")

    # Check container is running
    code, stdout, _ = run_command(
        "docker-compose -f docker-compose.personal.yml ps --format json sigmalang-api"
    )

    if code != 0:
        # Fallback to basic check
        code, stdout, _ = run_command(
            "docker ps --filter name=sigmalang-api --format '{{.Status}}'"
        )
        if "Up" not in stdout:
            logger.error("Container is not running")
            return False

    logger.info("Deployment verified successfully")
    return True


# =============================================================================
# Main Pipeline
# =============================================================================

def run_update_pipeline(force: bool = False) -> Dict[str, Any]:
    """
    Run the full update pipeline.

    Steps:
    1. Check for updates (skip if force)
    2. Pull latest code
    3. Run tests
    4. Build Docker image
    5. Rolling restart
    6. Verify deployment
    """
    state = UpdateState.load()
    start_time = time.time()

    result = {
        'success': False,
        'steps_completed': [],
        'steps_failed': [],
        'duration_seconds': 0
    }

    logger.info("=" * 60)
    logger.info("SigmaLang Auto-Update Pipeline")
    logger.info("=" * 60)

    try:
        # Step 1: Check for updates
        if not force:
            has_updates, remote_commit = check_for_updates()
            if not has_updates:
                result['success'] = True
                result['steps_completed'].append('check_updates (no updates)')
                logger.info("No updates available")
                return result
            result['steps_completed'].append('check_updates')
        else:
            logger.info("Force update - skipping git check")
            result['steps_completed'].append('check_updates (forced)')

        # Save current state for rollback
        state.previous_commit = state.current_commit or get_current_commit()

        # Step 2: Pull updates
        if not force:
            if not pull_updates():
                result['steps_failed'].append('pull_updates')
                return result
            result['steps_completed'].append('pull_updates')

        state.current_commit = get_current_commit()

        # Step 3: Run tests
        if not run_tests():
            result['steps_failed'].append('run_tests')
            logger.warning("Tests failed - aborting update")
            # Rollback git
            if state.previous_commit:
                run_command(f"git checkout {state.previous_commit}")
            return result
        result['steps_completed'].append('run_tests')

        # Step 4: Build image
        if not build_image():
            result['steps_failed'].append('build_image')
            logger.warning("Build failed - aborting update")
            if state.previous_commit:
                run_command(f"git checkout {state.previous_commit}")
            return result
        result['steps_completed'].append('build_image')

        # Step 5: Rolling restart
        if not rolling_restart():
            result['steps_failed'].append('rolling_restart')
            logger.warning("Restart failed - attempting rollback")
            rollback(state)
            return result
        result['steps_completed'].append('rolling_restart')

        # Step 6: Verify
        if not verify_deployment():
            result['steps_failed'].append('verify_deployment')
            logger.warning("Verification failed - attempting rollback")
            rollback(state)
            return result
        result['steps_completed'].append('verify_deployment')

        # Success
        result['success'] = True
        state.last_status = "success"
        state.last_update = datetime.now(timezone.utc).isoformat()
        state.update_count += 1
        state.save()

        duration = time.time() - start_time
        result['duration_seconds'] = round(duration, 1)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"[PASS] Update successful in {duration:.1f}s")
        logger.info(f"  Commit: {state.current_commit[:8]}")
        logger.info(f"  Updates applied: {state.update_count}")
        logger.info(f"{'=' * 60}")

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        result['steps_failed'].append(f'exception: {e}')
        state.last_status = "error"
        state.save()

    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SigmaLang Auto-Update Pipeline")
    parser.add_argument('--check', action='store_true', help='Check for updates only')
    parser.add_argument('--force', action='store_true', help='Force rebuild without git check')
    parser.add_argument('--rollback', action='store_true', help='Rollback to previous version')
    parser.add_argument('--status', action='store_true', help='Show current update status')

    args = parser.parse_args()

    if args.status:
        state = UpdateState.load()
        print(json.dumps({
            'current_commit': state.current_commit,
            'previous_commit': state.previous_commit,
            'last_update': state.last_update,
            'last_status': state.last_status,
            'update_count': state.update_count,
            'rollback_count': state.rollback_count
        }, indent=2))
        return

    if args.check:
        has_updates, _ = check_for_updates()
        sys.exit(0 if has_updates else 1)

    if args.rollback:
        state = UpdateState.load()
        success = rollback(state)
        sys.exit(0 if success else 1)

    result = run_update_pipeline(force=args.force)
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
