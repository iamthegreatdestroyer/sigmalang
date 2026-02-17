"""
Generate client SDKs from OpenAPI specification

Generates JavaScript and Java SDKs using OpenAPI Generator.

Prerequisites:
    npm install @openapitools/openapi-generator-cli -g

Usage:
    python scripts/generate_sdks.py --all
    python scripts/generate_sdks.py --javascript
    python scripts/generate_sdks.py --java
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class SDKGenerator:
    """Generate client SDKs from OpenAPI specification."""

    def __init__(self, project_root: Path = Path.cwd()):
        self.project_root = project_root
        self.openapi_spec = project_root / "docs" / "api" / "openapi.json"
        self.sdks_dir = project_root / "sdks"

    def check_prerequisites(self) -> bool:
        """Check if OpenAPI Generator is installed."""
        print("Checking prerequisites...")

        # Check if openapi-generator-cli is available
        try:
            result = subprocess.run(
                ["npx", "@openapitools/openapi-generator-cli", "version"],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                print(f"[PASS] OpenAPI Generator found: {result.stdout.strip()}")
                return True
            else:
                print("[FAIL] OpenAPI Generator not found")
                print("Install with: npm install @openapitools/openapi-generator-cli -g")
                return False

        except FileNotFoundError:
            print("[FAIL] npx not found. Install Node.js first.")
            return False

    def check_openapi_spec(self) -> bool:
        """Check if OpenAPI spec exists."""
        if not self.openapi_spec.exists():
            print(f"[FAIL] OpenAPI spec not found: {self.openapi_spec}")
            print("Generate it with: python scripts/generate_openapi_spec.py")
            return False

        print(f"[PASS] OpenAPI spec found: {self.openapi_spec}")
        return True

    def run_generator(
        self,
        generator: str,
        output_dir: str,
        additional_props: dict = None
    ) -> Tuple[bool, str]:
        """Run OpenAPI Generator for a specific language."""
        print(f"\n{'='*70}")
        print(f"Generating {generator.upper()} SDK")
        print(f"{'='*70}\n")

        output_path = self.sdks_dir / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "npx", "@openapitools/openapi-generator-cli", "generate",
            "-i", str(self.openapi_spec),
            "-g", generator,
            "-o", str(output_path),
            "--additional-properties",
            ",".join(f"{k}={v}" for k, v in (additional_props or {}).items())
        ]

        print(f"Command: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                check=True,
                capture_output=False
            )

            print(f"\n[PASS] {generator.upper()} SDK generated successfully")
            print(f"       Location: {output_path}")

            return True, str(output_path)

        except subprocess.CalledProcessError as e:
            print(f"\n[FAIL] {generator.upper()} SDK generation failed")
            return False, ""

    def generate_javascript_sdk(self) -> bool:
        """Generate JavaScript/TypeScript SDK."""
        additional_props = {
            "projectName": "sigmalang-client",
            "projectVersion": "1.0.0",
            "projectDescription": "ΣLANG JavaScript/TypeScript Client SDK",
            "moduleName": "SigmaLangClient",
            "npmName": "@sigmalang/client",
            "npmVersion": "1.0.0",
            "supportsES6": "true",
            "withInterfaces": "true"
        }

        success, output_dir = self.run_generator(
            "typescript-fetch",
            "javascript",
            additional_props
        )

        if success:
            # Create package.json customizations
            self._create_javascript_readme(Path(output_dir))

        return success

    def generate_java_sdk(self) -> bool:
        """Generate Java SDK."""
        additional_props = {
            "groupId": "io.sigmalang",
            "artifactId": "sigmalang-client",
            "artifactVersion": "1.0.0",
            "apiPackage": "io.sigmalang.client.api",
            "modelPackage": "io.sigmalang.client.model",
            "invokerPackage": "io.sigmalang.client",
            "library": "okhttp-gson",
            "java8": "true",
            "dateLibrary": "java8"
        }

        success, output_dir = self.run_generator(
            "java",
            "java",
            additional_props
        )

        if success:
            self._create_java_readme(Path(output_dir))

        return success

    def _create_javascript_readme(self, output_dir: Path):
        """Create README for JavaScript SDK."""
        readme_content = """# ΣLANG JavaScript/TypeScript Client

Official JavaScript/TypeScript client for the ΣLANG API.

## Installation

```bash
npm install @sigmalang/client
```

## Usage

```typescript
import { Configuration, DefaultApi } from '@sigmalang/client';

const config = new Configuration({
  basePath: 'https://api.sigmalang.io'
});

const api = new DefaultApi(config);

// Encode text
const response = await api.encodeText({
  text: 'Hello, world!'
});

console.log(response);
```

## API Documentation

See [API documentation](https://docs.sigmalang.io/api) for details.

## License

MIT
"""
        (output_dir / "README.md").write_text(readme_content)

    def _create_java_readme(self, output_dir: Path):
        """Create README for Java SDK."""
        readme_content = """# ΣLANG Java Client

Official Java client for the ΣLANG API.

## Installation

### Maven

```xml
<dependency>
    <groupId>io.sigmalang</groupId>
    <artifactId>sigmalang-client</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Gradle

```gradle
implementation 'io.sigmalang:sigmalang-client:1.0.0'
```

## Usage

```java
import io.sigmalang.client.*;
import io.sigmalang.client.api.*;
import io.sigmalang.client.model.*;

public class Example {
    public static void main(String[] args) {
        ApiClient defaultClient = Configuration.getDefaultApiClient();
        defaultClient.setBasePath("https://api.sigmalang.io");

        DefaultApi api = new DefaultApi(defaultClient);

        try {
            EncodeRequest request = new EncodeRequest().text("Hello, world!");
            EncodeResponse result = api.encodeText(request);
            System.out.println(result);
        } catch (ApiException e) {
            e.printStackTrace();
        }
    }
}
```

## API Documentation

See [API documentation](https://docs.sigmalang.io/api) for details.

## License

MIT
"""
        (output_dir / "README.md").write_text(readme_content)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate client SDKs from OpenAPI specification"
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all SDKs (JavaScript and Java)'
    )

    parser.add_argument(
        '--javascript',
        action='store_true',
        help='Generate JavaScript/TypeScript SDK'
    )

    parser.add_argument(
        '--java',
        action='store_true',
        help='Generate Java SDK'
    )

    args = parser.parse_args()

    generator = SDKGenerator()

    # Check prerequisites
    if not generator.check_prerequisites():
        return 1

    if not generator.check_openapi_spec():
        return 1

    # Default to --all if no specific SDK selected
    if not any([args.all, args.javascript, args.java]):
        args.all = True

    success = True

    if args.all or args.javascript:
        if not generator.generate_javascript_sdk():
            success = False

    if args.all or args.java:
        if not generator.generate_java_sdk():
            success = False

    if success:
        print("\n" + "="*70)
        print("[PASS] All SDKs generated successfully!")
        print("="*70)
        print(f"\nSDKs location: {generator.sdks_dir}")
        print("\nNext steps:")
        print("  1. Test the generated SDKs")
        print("  2. Publish to package registries (npm, Maven Central)")
        print("  3. Update documentation with SDK examples")
    else:
        print("\n[FAIL] Some SDKs failed to generate")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
