"""
Generate OpenAPI specification from FastAPI app

Usage:
    python scripts/generate_openapi_spec.py --output docs/api/openapi.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_openapi_spec(output_file: str = "docs/api/openapi.json"):
    """Generate OpenAPI specification from FastAPI app."""
    try:
        from sigmalang.core.api_server import create_app

        # Create FastAPI app
        app = create_app()

        # Get OpenAPI schema
        openapi_schema = app.openapi()

        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(output_path, 'w') as f:
            json.dump(openapi_schema, f, indent=2)

        print(f"[PASS] OpenAPI specification generated: {output_path}")
        print(f"       Schema version: {openapi_schema.get('openapi', 'unknown')}")
        print(f"       API title: {openapi_schema.get('info', {}).get('title', 'unknown')}")
        print(f"       API version: {openapi_schema.get('info', {}).get('version', 'unknown')}")
        print(f"       Endpoints: {len(openapi_schema.get('paths', {}))}")

        return True

    except Exception as e:
        print(f"[FAIL] Failed to generate OpenAPI spec: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate OpenAPI specification from FastAPI app"
    )

    parser.add_argument(
        '--output',
        type=str,
        default='docs/api/openapi.json',
        help='Output file path (default: docs/api/openapi.json)'
    )

    args = parser.parse_args()

    success = generate_openapi_spec(args.output)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
