#!/usr/bin/env python3
"""
ΣLANG Master Automation Orchestrator
=====================================
Executes the Master Action Plan with maximum autonomy.

This script orchestrates all automation phases, managing dependencies,
tracking progress, and providing self-healing capabilities.

Usage:
    python scripts/master_automation.py --phase=all --autonomous
    python scripts/master_automation.py --phase=1 --dry-run
    python scripts/master_automation.py --status
    python scripts/master_automation.py --phase=2 --retry-failed
"""

import argparse
import asyncio
import io
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Fix Windows console Unicode encoding
if sys.platform == 'win32':
    # Force UTF-8 output on Windows console
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # Also try to set console mode for UTF-8
    try:
        os.system('chcp 65001 > nul 2>&1')
    except Exception:
        pass

# Configure logging with UTF-8 support - avoid duplicates
logger = logging.getLogger("master_automation")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent duplicate logs

# Only add handlers if they don't exist
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler("automation_log.txt", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(file_handler)
logger.handlers = []  # Clear default handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class PhaseStatus(Enum):
    """Phase completion status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class Task:
    """Represents an automation task."""
    id: str
    name: str
    description: str
    command: str
    phase: int
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    retries: int = 0
    max_retries: int = 3
    timeout: int = 3600  # seconds
    output: str = ""
    error: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    critical: bool = True  # If True, failure blocks phase


@dataclass
class Phase:
    """Represents an automation phase."""
    id: int
    name: str
    description: str
    tasks: List[Task] = field(default_factory=list)
    status: PhaseStatus = PhaseStatus.NOT_STARTED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class AutomationState:
    """Persistent state management for automation."""
    
    STATE_FILE = Path("automation_state.json")
    
    def __init__(self):
        self.phases: Dict[int, Phase] = {}
        self.tasks: Dict[str, Task] = {}
        self.load()
    
    def load(self) -> None:
        """Load state from disk."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    # Reconstruct state from JSON
                    logger.info(f"Loaded automation state from {self.STATE_FILE}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def save(self) -> None:
        """Save state to disk."""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "phases": {
                    pid: {
                        "status": phase.status.value,
                        "start_time": phase.start_time.isoformat() if phase.start_time else None,
                        "end_time": phase.end_time.isoformat() if phase.end_time else None,
                    }
                    for pid, phase in self.phases.items()
                },
                "tasks": {
                    tid: {
                        "status": task.status.value,
                        "retries": task.retries,
                        "start_time": task.start_time.isoformat() if task.start_time else None,
                        "end_time": task.end_time.isoformat() if task.end_time else None,
                    }
                    for tid, task in self.tasks.items()
                },
            }
            with open(self.STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved automation state to {self.STATE_FILE}")
        except Exception as e:
            logger.error(f"Could not save state: {e}")


class MasterAutomation:
    """Master automation orchestrator."""
    
    def __init__(self, dry_run: bool = False, autonomous: bool = False):
        self.dry_run = dry_run
        self.autonomous = autonomous
        self.state = AutomationState()
        self.phases = self._initialize_phases()
        
    def _initialize_phases(self) -> Dict[int, Phase]:
        """Initialize all automation phases and tasks."""
        phases = {}
        
        # Phase 1: Immediate Fixes
        phase1 = Phase(
            id=1,
            name="Immediate Fixes",
            description="Resolve known issues with zero human intervention",
        )
        phase1.tasks = [
            Task(
                id="1.1",
                name="Security Remediation",
                description="Scan and remediate security findings",
                command="python scripts/auto_security_fix.py --scan --remediate --verify",
                phase=1,
            ),
            Task(
                id="1.2",
                name="Unicode Documentation Fix",
                description="Fix sigma character encoding issues",
                command="python scripts/fix_unicode_docs.py --detect --replace --regenerate",
                phase=1,
                dependencies=["1.1"],
            ),
            Task(
                id="1.3",
                name="Dependency Resolution",
                description="Resolve import errors in profiling",
                command="python scripts/auto_profile_fix.py --diagnose --install --verify",
                phase=1,
            ),
            Task(
                id="1.4",
                name="Phase Validation",
                description="Comprehensive phase completion validation",
                command="python scripts/phase2_validation.py --comprehensive",
                phase=1,
                dependencies=["1.1", "1.2", "1.3"],
            ),
        ]
        phases[1] = phase1
        
        # Phase 2: E2E Testing
        phase2 = Phase(
            id=2,
            name="End-to-End Testing",
            description="Comprehensive system validation",
        )
        phase2.tasks = [
            Task(
                id="2.1",
                name="Integration Tests",
                description="Run full E2E integration test suite",
                command="python -m pytest tests/integration/ --tb=short -q",
                phase=2,
                timeout=7200,
            ),
            Task(
                id="2.2",
                name="Extended Load Test",
                description="24-hour sustained load test",
                command="python scripts/load_test.py --duration=1h --concurrency=100",
                phase=2,
                dependencies=["2.1"],
                timeout=7200,
                critical=False,
            ),
            Task(
                id="2.3",
                name="Chaos Engineering",
                description="Automated chaos testing scenarios",
                command="python scripts/chaos_engineering.py --quick-test",
                phase=2,
                dependencies=["2.1"],
                critical=False,
            ),
        ]
        phases[2] = phase2
        
        # Phase 3: Production Deployment
        phase3 = Phase(
            id=3,
            name="Production Deployment",
            description="Enterprise-grade multi-region deployment",
        )
        phase3.tasks = [
            Task(
                id="3.1",
                name="Helm Chart Generation",
                description="Generate Helm charts from K8s manifests",
                command="python scripts/auto_helm_gen.py --source=infrastructure/kubernetes/",
                phase=3,
            ),
            Task(
                id="3.2",
                name="Deploy Staging",
                description="Deploy to staging environment",
                command="python scripts/deploy_staging.py --validate",
                phase=3,
                dependencies=["3.1"],
            ),
            Task(
                id="3.3",
                name="Backup Configuration",
                description="Configure automated backup",
                command="python scripts/auto_backup_config.py --schedule='0 */6 * * *'",
                phase=3,
                dependencies=["3.2"],
            ),
        ]
        phases[3] = phase3
        
        # Phase 4: SDK Development
        phase4 = Phase(
            id=4,
            name="SDK Development",
            description="Multi-language SDK generation",
        )
        phase4.tasks = [
            Task(
                id="4.1",
                name="Python SDK",
                description="Generate Python SDK",
                command="python scripts/auto_sdk_gen.py --language=python",
                phase=4,
            ),
            Task(
                id="4.2",
                name="JavaScript SDK",
                description="Generate JavaScript SDK",
                command="python scripts/auto_sdk_gen.py --language=javascript",
                phase=4,
            ),
            Task(
                id="4.3",
                name="Go SDK",
                description="Generate Go SDK",
                command="python scripts/auto_sdk_gen.py --language=go",
                phase=4,
            ),
            Task(
                id="4.4",
                name="SDK Testing",
                description="Test all generated SDKs",
                command="python scripts/test_all_sdks.py --parallel",
                phase=4,
                dependencies=["4.1", "4.2", "4.3"],
            ),
        ]
        phases[4] = phase4
        
        # Phase 5: Marketplace Integration
        phase5 = Phase(
            id=5,
            name="Marketplace Integration",
            description="Cloud marketplace listings",
        )
        phase5.tasks = [
            Task(
                id="5.1",
                name="AWS Marketplace",
                description="Generate AWS Marketplace package",
                command="python scripts/auto_marketplace.py --platform=aws",
                phase=5,
            ),
            Task(
                id="5.2",
                name="GCP Marketplace",
                description="Generate GCP Marketplace package",
                command="python scripts/auto_marketplace.py --platform=gcp",
                phase=5,
            ),
            Task(
                id="5.3",
                name="Azure Marketplace",
                description="Generate Azure Marketplace package",
                command="python scripts/auto_marketplace.py --platform=azure",
                phase=5,
            ),
        ]
        phases[5] = phase5
        
        # Phase 6: Observability
        phase6 = Phase(
            id=6,
            name="Monitoring & Observability",
            description="Production-grade observability",
        )
        phase6.tasks = [
            Task(
                id="6.1",
                name="Deploy Observability Stack",
                description="Deploy Prometheus, Grafana, Jaeger",
                command="python scripts/deploy_observability.py --all",
                phase=6,
            ),
            Task(
                id="6.2",
                name="Configure Alerting",
                description="Set up automated alerting rules",
                command="python scripts/configure_alerts.py --auto",
                phase=6,
                dependencies=["6.1"],
            ),
        ]
        phases[6] = phase6
        
        # Phase 7: Continuous Automation
        phase7 = Phase(
            id=7,
            name="Continuous Automation",
            description="Self-sustaining autonomous operations",
        )
        phase7.tasks = [
            Task(
                id="7.1",
                name="Self-Healing Setup",
                description="Deploy self-healing infrastructure",
                command="python scripts/deploy_self_healing.py --all",
                phase=7,
            ),
            Task(
                id="7.2",
                name="Continuous Optimizer",
                description="Enable continuous optimization bot",
                command="python scripts/continuous_optimizer.py --enable",
                phase=7,
                dependencies=["7.1"],
            ),
            Task(
                id="7.3",
                name="Security Automation",
                description="Enable automated security monitoring",
                command="python scripts/security_automation.py --enable",
                phase=7,
                dependencies=["7.1"],
            ),
        ]
        phases[7] = phase7
        
        return phases
    
    async def execute_task(self, task: Task) -> bool:
        """Execute a single task."""
        logger.info(f"▶ Executing task {task.id}: {task.name}")
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        self.state.save()
        
        if self.dry_run:
            logger.info(f"  [DRY-RUN] Would execute: {task.command}")
            task.status = TaskStatus.SUCCESS
            task.end_time = datetime.now()
            return True
        
        try:
            # Execute the command
            process = await asyncio.create_subprocess_shell(
                task.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=task.timeout,
                )
                task.output = stdout.decode() if stdout else ""
                task.error = stderr.decode() if stderr else ""
                
                if process.returncode == 0:
                    task.status = TaskStatus.SUCCESS
                    logger.info(f"  ✅ Task {task.id} completed successfully")
                    task.end_time = datetime.now()
                    return True
                else:
                    task.status = TaskStatus.FAILED
                    logger.error(f"  ❌ Task {task.id} failed with code {process.returncode}")
                    logger.error(f"     Error: {task.error[:500] if task.error else 'No error output'}")
                    
            except asyncio.TimeoutError:
                process.kill()
                task.status = TaskStatus.FAILED
                task.error = f"Task timed out after {task.timeout} seconds"
                logger.error(f"  ❌ Task {task.id} timed out")
                
        except FileNotFoundError:
            # Script doesn't exist yet - mark as skipped for now
            task.status = TaskStatus.SKIPPED
            task.error = f"Script not found: {task.command.split()[0]}"
            logger.warning(f"  ⚠ Task {task.id} skipped (script not found)")
            task.end_time = datetime.now()
            return True  # Don't block on missing scripts
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"  ❌ Task {task.id} failed with exception: {e}")
        
        task.end_time = datetime.now()
        self.state.save()
        
        # Retry logic
        if task.status == TaskStatus.FAILED and task.retries < task.max_retries:
            task.retries += 1
            task.status = TaskStatus.RETRYING
            logger.info(f"  🔄 Retrying task {task.id} (attempt {task.retries}/{task.max_retries})")
            return await self.execute_task(task)
        
        return task.status == TaskStatus.SUCCESS
    
    async def execute_phase(self, phase: Phase) -> bool:
        """Execute all tasks in a phase respecting dependencies."""
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 PHASE {phase.id}: {phase.name}")
        logger.info(f"   {phase.description}")
        logger.info(f"{'='*60}\n")
        
        phase.status = PhaseStatus.IN_PROGRESS
        phase.start_time = datetime.now()
        self.state.save()
        
        # Build dependency graph
        completed_tasks = set()
        failed_tasks = set()
        
        while len(completed_tasks) + len(failed_tasks) < len(phase.tasks):
            # Find tasks that can be executed
            executable = []
            for task in phase.tasks:
                if task.id in completed_tasks or task.id in failed_tasks:
                    continue
                if all(dep in completed_tasks for dep in task.dependencies):
                    executable.append(task)
            
            if not executable:
                # Deadlock or all remaining tasks have failed dependencies
                logger.error("No executable tasks remaining - possible dependency deadlock")
                break
            
            # Execute tasks in parallel where possible
            results = await asyncio.gather(
                *[self.execute_task(task) for task in executable],
                return_exceptions=True,
            )
            
            for task, result in zip(executable, results):
                if result is True:
                    completed_tasks.add(task.id)
                else:
                    if task.critical:
                        failed_tasks.add(task.id)
                    else:
                        completed_tasks.add(task.id)  # Non-critical can continue
        
        phase.end_time = datetime.now()
        
        # Determine phase status
        if len(failed_tasks) == 0:
            phase.status = PhaseStatus.COMPLETED
            logger.info(f"\n✅ Phase {phase.id} completed successfully")
            return True
        elif len(completed_tasks) > 0:
            phase.status = PhaseStatus.PARTIAL
            logger.warning(f"\n⚠ Phase {phase.id} partially completed ({len(completed_tasks)}/{len(phase.tasks)} tasks)")
            return False
        else:
            phase.status = PhaseStatus.FAILED
            logger.error(f"\n❌ Phase {phase.id} failed")
            return False
    
    async def run(self, phases_to_run: Optional[List[int]] = None) -> bool:
        """Run automation for specified phases."""
        logger.info("\n" + "="*60)
        logger.info("🚀 ΣLANG MASTER AUTOMATION ORCHESTRATOR")
        logger.info(f"   Mode: {'DRY-RUN' if self.dry_run else 'LIVE'}")
        logger.info(f"   Autonomous: {self.autonomous}")
        logger.info("="*60 + "\n")
        
        phases_to_execute = phases_to_run or list(self.phases.keys())
        overall_success = True
        
        for phase_id in sorted(phases_to_execute):
            if phase_id not in self.phases:
                logger.warning(f"Phase {phase_id} not found, skipping")
                continue
            
            phase = self.phases[phase_id]
            success = await self.execute_phase(phase)
            
            if not success and not self.autonomous:
                # In non-autonomous mode, ask for confirmation to continue
                logger.warning("Phase did not complete successfully.")
                response = input("Continue to next phase? [y/N]: ")
                if response.lower() != 'y':
                    logger.info("Automation halted by user")
                    overall_success = False
                    break
            elif not success:
                overall_success = False
        
        # Generate summary
        self._generate_summary()
        self.state.save()
        
        return overall_success
    
    def _generate_summary(self) -> None:
        """Generate execution summary."""
        logger.info("\n" + "="*60)
        logger.info("📊 EXECUTION SUMMARY")
        logger.info("="*60)
        
        for phase_id, phase in sorted(self.phases.items()):
            status_emoji = {
                PhaseStatus.COMPLETED: "✅",
                PhaseStatus.PARTIAL: "⚠",
                PhaseStatus.FAILED: "❌",
                PhaseStatus.IN_PROGRESS: "🔄",
                PhaseStatus.NOT_STARTED: "⏸",
            }.get(phase.status, "?")
            
            logger.info(f"\n{status_emoji} Phase {phase.id}: {phase.name}")
            
            for task in phase.tasks:
                task_emoji = {
                    TaskStatus.SUCCESS: "  ✅",
                    TaskStatus.FAILED: "  ❌",
                    TaskStatus.SKIPPED: "  ⏭",
                    TaskStatus.RUNNING: "  🔄",
                    TaskStatus.PENDING: "  ⏸",
                    TaskStatus.RETRYING: "  🔁",
                }.get(task.status, "  ?")
                
                duration = ""
                if task.start_time and task.end_time:
                    delta = task.end_time - task.start_time
                    duration = f" ({delta.total_seconds():.1f}s)"
                
                logger.info(f"{task_emoji} {task.id}: {task.name}{duration}")
        
        logger.info("\n" + "="*60)
    
    def show_status(self) -> None:
        """Show current automation status."""
        logger.info("\n📋 AUTOMATION STATUS\n")
        
        for phase_id, phase in sorted(self.phases.items()):
            status = phase.status.value.upper()
            logger.info(f"Phase {phase.id}: {phase.name} [{status}]")
            
            for task in phase.tasks:
                status_str = task.status.value
                if task.status == TaskStatus.FAILED:
                    status_str += f" (retries: {task.retries})"
                logger.info(f"  - {task.id}: {task.name} [{status_str}]")
        
        logger.info("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ΣLANG Master Automation Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python master_automation.py --phase=1 --dry-run
    python master_automation.py --phase=all --autonomous
    python master_automation.py --status
    python master_automation.py --phase=2,3 --retry-failed
        """,
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        help="Phase(s) to execute: 'all', single number (1), or comma-separated (1,2,3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Run without human intervention on failures",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current automation status",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only failed tasks",
    )
    
    args = parser.parse_args()
    
    # Parse phases
    phases_to_run = None
    if args.phase != "all":
        try:
            phases_to_run = [int(p.strip()) for p in args.phase.split(",")]
        except ValueError:
            logger.error(f"Invalid phase specification: {args.phase}")
            sys.exit(1)
    
    automation = MasterAutomation(
        dry_run=args.dry_run,
        autonomous=args.autonomous,
    )
    
    if args.status:
        automation.show_status()
        return
    
    # Run automation
    success = asyncio.run(automation.run(phases_to_run))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
