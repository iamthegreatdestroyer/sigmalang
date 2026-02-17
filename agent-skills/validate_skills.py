"""Validate agent skills format compliance."""
import json
from pathlib import Path

def validate_skill(skill_dir):
    """Validate agentskills.io format compliance."""
    skill_json = skill_dir / "skill.json"
    instructions_md = skill_dir / "instructions.md"

    errors = []

    if not skill_json.exists():
        errors.append(f"Missing skill.json in {skill_dir.name}")
    if not instructions_md.exists():
        errors.append(f"Missing instructions.md in {skill_dir.name}")

    if skill_json.exists():
        try:
            with open(skill_json) as f:
                data = json.load(f)
                required = ["name", "version", "description", "capabilities", "triggers"]
                for field in required:
                    if field not in data:
                        errors.append(f"{skill_dir.name}: Missing '{field}'")
        except json.JSONDecodeError as e:
            errors.append(f"{skill_dir.name}: Invalid JSON - {e}")

    return errors

if __name__ == '__main__':
    # Validate all skills
    skills_dir = Path("S:/sigmalang/agent-skills")
    all_errors = []

    for skill_path in skills_dir.iterdir():
        if skill_path.is_dir() and not skill_path.name.startswith('.'):
            errors = validate_skill(skill_path)
            all_errors.extend(errors)

    if all_errors:
        print("VALIDATION FAILED:")
        for error in all_errors:
            print(f"  - {error}")
    else:
        skill_count = len([p for p in skills_dir.iterdir() if p.is_dir() and not p.name.startswith('.')])
        print(f"[OK] All skills validated successfully")
        print(f"[OK] {skill_count} skills ready for deployment")
