#!/bin/bash
# ΣLANG SDK Generation Script
# Generates TypeScript/JavaScript and Java SDKs from OpenAPI spec

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    # Check OpenAPI Generator
    if ! command -v openapi-generator-cli &> /dev/null; then
        print_error "openapi-generator-cli not installed"
        echo "Install with: npm install -g @openapitools/openapi-generator-cli"
        exit 1
    fi

    print_success "Dependencies OK"
}

# Export OpenAPI spec from running server
export_spec() {
    print_info "Exporting OpenAPI specification..."

    mkdir -p sdk-output

    # Check if server is running
    if ! curl -s http://localhost:26080/health > /dev/null 2>&1; then
        print_error "API server not running at http://localhost:26080"
        echo "Start with: docker compose up -d"
        exit 1
    fi

    # Download OpenAPI spec
    curl -s http://localhost:26080/openapi.json -o sdk-output/openapi.json

    if [ ! -f sdk-output/openapi.json ]; then
        print_error "Failed to download OpenAPI spec"
        exit 1
    fi

    print_success "OpenAPI spec exported"
}

# Generate TypeScript SDK
generate_typescript_sdk() {
    print_header "Generating TypeScript/JavaScript SDK"

    LANG="typescript-axios"
    OUTPUT_DIR="sdks/typescript"

    print_info "Generating $LANG SDK to $OUTPUT_DIR..."

    openapi-generator-cli generate \
        -i sdk-output/openapi.json \
        -g "$LANG" \
        -o "$OUTPUT_DIR" \
        --additional-properties=npmName=@sigmalang/sdk,npmVersion=1.0.0,supportsES6=true

    print_success "TypeScript SDK generated at $OUTPUT_DIR"

    # Create package.json if needed
    if [ ! -f "$OUTPUT_DIR/package.json" ]; then
        print_info "Creating package.json..."
        cat > "$OUTPUT_DIR/package.json" <<EOF
{
  "name": "@sigmalang/sdk",
  "version": "1.0.0",
  "description": "ΣLANG TypeScript SDK",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "test": "jest",
    "prepublish": "npm run build"
  },
  "dependencies": {
    "axios": "^1.0.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "typescript": "^5.0.0",
    "jest": "^29.0.0",
    "@types/jest": "^29.0.0"
  },
  "repository": "https://github.com/iamthegreatdestroyer/sigmalang",
  "license": "MIT"
}
EOF
        print_success "package.json created"
    fi
}

# Generate Java SDK
generate_java_sdk() {
    print_header "Generating Java SDK"

    LANG="java"
    OUTPUT_DIR="sdks/java"

    print_info "Generating $LANG SDK to $OUTPUT_DIR..."

    openapi-generator-cli generate \
        -i sdk-output/openapi.json \
        -g "$LANG" \
        -o "$OUTPUT_DIR" \
        --additional-properties=packageName=io.sigmalang,packageVersion=1.0.0,useJakartaEe=false

    print_success "Java SDK generated at $OUTPUT_DIR"

    # Create Maven pom.xml if needed
    if [ ! -f "$OUTPUT_DIR/pom.xml" ]; then
        print_info "Creating pom.xml..."
        cat > "$OUTPUT_DIR/pom.xml" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>io.sigmalang</groupId>
    <artifactId>sigmalang-java</artifactId>
    <version>1.0.0</version>

    <name>ΣLANG Java SDK</name>
    <description>Java SDK for ΣLANG</description>

    <properties>
        <maven.compiler.source>8</maven.compiler.source>
        <maven.compiler.target>8</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>

    <dependencies>
        <dependency>
            <groupId>com.squareup.okhttp3</groupId>
            <artifactId>okhttp</artifactId>
            <version>4.10.0</version>
        </dependency>
        <dependency>
            <groupId>com.google.code.gson</groupId>
            <artifactId>gson</artifactId>
            <version>2.10.0</version>
        </dependency>
    </dependencies>
</project>
EOF
        print_success "pom.xml created"
    fi
}

# Generate Python SDK
generate_python_sdk() {
    print_header "Generating Python SDK"

    LANG="python"
    OUTPUT_DIR="sdks/python"

    print_info "Generating $LANG SDK to $OUTPUT_DIR..."

    openapi-generator-cli generate \
        -i sdk-output/openapi.json \
        -g "$LANG" \
        -o "$OUTPUT_DIR" \
        --additional-properties=packageName=sigmalang_sdk,packageVersion=1.0.0

    print_success "Python SDK generated at $OUTPUT_DIR"
}

# Generate Go SDK
generate_go_sdk() {
    print_header "Generating Go SDK"

    LANG="go"
    OUTPUT_DIR="sdks/go"

    print_info "Generating $LANG SDK to $OUTPUT_DIR..."

    openapi-generator-cli generate \
        -i sdk-output/openapi.json \
        -g "$LANG" \
        -o "$OUTPUT_DIR" \
        --additional-properties=packageName=sigmalang

    print_success "Go SDK generated at $OUTPUT_DIR"
}

# Generate all SDKs
generate_all_sdks() {
    print_header "Generating All SDKs"

    generate_typescript_sdk
    echo ""
    generate_java_sdk
    echo ""
    generate_python_sdk
    echo ""
    generate_go_sdk

    print_success "All SDKs generated"
}

# Show usage
show_usage() {
    cat << EOF
ΣLANG SDK Generation Script

Usage: $0 [COMMAND]

Commands:
    all         Generate all SDKs (TypeScript, Java, Python, Go)
    typescript  Generate TypeScript/JavaScript SDK
    java        Generate Java SDK
    python      Generate Python SDK
    go          Generate Go SDK
    help        Show this help message

Prerequisites:
    - OpenAPI Generator: npm install -g @openapitools/openapi-generator-cli
    - Running ΣLANG API: docker compose up -d

Output directories:
    - TypeScript: sdks/typescript/
    - Java: sdks/java/
    - Python: sdks/python/
    - Go: sdks/go/

Examples:
    $0 all              # Generate all SDKs
    $0 typescript       # Generate TypeScript only

EOF
}

# Main
main() {
    local command="${1:-help}"

    case "$command" in
        all)
            check_dependencies
            export_spec
            generate_all_sdks
            ;;
        typescript)
            check_dependencies
            export_spec
            generate_typescript_sdk
            ;;
        java)
            check_dependencies
            export_spec
            generate_java_sdk
            ;;
        python)
            check_dependencies
            export_spec
            generate_python_sdk
            ;;
        go)
            check_dependencies
            export_spec
            generate_go_sdk
            ;;
        help)
            show_usage
            ;;
        *)
            echo "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
