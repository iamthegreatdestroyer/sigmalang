#!/bin/bash
# ΣLANG Coverage Reporting Script
# Safely runs coverage with proper timeout and configuration

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

# Check if pytest is installed
check_dependencies() {
    print_info "Checking dependencies..."

    if ! python -c "import pytest" 2>/dev/null; then
        print_error "pytest not installed"
        echo "Install with: pip install -e '.[dev]'"
        exit 1
    fi

    if ! python -c "import coverage" 2>/dev/null; then
        print_error "coverage not installed"
        echo "Install with: pip install pytest-cov"
        exit 1
    fi

    print_success "Dependencies OK"
}

# Run fast coverage (unit tests only)
run_fast_coverage() {
    print_header "Fast Coverage (Unit Tests Only)"
    print_info "Running tests with 120 second timeout..."

    timeout 300 pytest \
        tests/ \
        --cov=sigmalang \
        --cov-report=term-missing \
        --cov-report=html \
        --cov-branch \
        --cov-config=.coveragerc \
        --timeout=120 \
        -x \
        --tb=short \
        -q \
        -k "not slow" \
        2>&1 || {
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 124 ]; then
                print_error "Coverage collection timed out (script timeout)"
                echo "Try: run_coverage --full"
                exit 1
            fi
            exit $EXIT_CODE
        }

    print_success "Coverage report generated"
    print_info "View report: open htmlcov/index.html"
}

# Run full coverage (all tests)
run_full_coverage() {
    print_header "Full Coverage (All Tests)"
    print_info "Running all tests with 300 second timeout..."
    print_info "This may take 5-10 minutes..."

    timeout 900 pytest \
        tests/ \
        --cov=sigmalang \
        --cov-report=term-missing \
        --cov-report=html \
        --cov-report=xml \
        --cov-branch \
        --cov-config=.coveragerc \
        --timeout=300 \
        --tb=short \
        -v \
        2>&1 || {
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 124 ]; then
                print_error "Coverage collection timed out"
                echo "Individual test timeout may be too low"
                exit 1
            fi
            exit $EXIT_CODE
        }

    print_success "Full coverage report generated"
    print_info "View HTML report: open htmlcov/index.html"
    print_info "View XML report: coverage.xml"
}

# Run coverage on specific test file
run_file_coverage() {
    local testfile="$1"

    if [ ! -f "$testfile" ]; then
        print_error "Test file not found: $testfile"
        exit 1
    fi

    print_header "Coverage for $testfile"
    print_info "Running with timeout..."

    timeout 300 pytest \
        "$testfile" \
        --cov=sigmalang \
        --cov-report=term-missing \
        --cov-report=html \
        --cov-branch \
        --cov-config=.coveragerc \
        --timeout=120 \
        -v \
        2>&1 || {
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 124 ]; then
                print_error "Coverage timed out"
                exit 1
            fi
            exit $EXIT_CODE
        }

    print_success "Coverage report generated"
}

# Generate coverage badge
generate_badge() {
    print_header "Generating Coverage Badge"
    print_info "Reading coverage.xml..."

    if [ ! -f "coverage.xml" ]; then
        print_error "coverage.xml not found"
        echo "Run coverage first: $0 --full"
        exit 1
    fi

    # Extract coverage percentage from XML
    COVERAGE=$(python -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('coverage.xml')
    root = tree.getroot()
    coverage = root.get('line-rate')
    if coverage:
        percentage = int(float(coverage) * 100)
        print(f'{percentage}%')
    else:
        print('Unknown')
except:
    print('Unknown')
    ")

    print_success "Coverage: $COVERAGE"

    # Create badge (simple text)
    cat > coverage_badge.txt << EOF
Coverage: $COVERAGE
EOF

    print_info "Badge created: coverage_badge.txt"
}

# Show usage
show_usage() {
    cat << EOF
ΣLANG Coverage Reporting Script

Usage: $0 [OPTION]

Options:
    --fast          Run fast coverage (unit tests only, default)
    --full          Run full coverage (all tests, 5-10 min)
    --file FILE     Run coverage on specific test file
    --badge         Generate coverage badge
    --help          Show this help message

Examples:
    $0                      # Fast coverage
    $0 --full               # Full coverage with all tests
    $0 --file tests/test_optimizations.py
    $0 --badge              # Generate badge

Configuration:
    - Timeout per test: 120 seconds
    - Global timeout: 300-900 seconds
    - Config file: .coveragerc
    - HTML output: htmlcov/index.html
    - XML output: coverage.xml

Tips:
    - Use --fast for regular development
    - Use --full for CI/CD and final checks
    - Increase individual test timeout if needed
    - Check .coveragerc for exclude patterns

EOF
}

# Main
main() {
    local option="${1:-fast}"

    check_dependencies

    case "$option" in
        --fast)
            run_fast_coverage
            ;;
        --full)
            run_full_coverage
            ;;
        --file)
            if [ -z "$2" ]; then
                print_error "No test file specified"
                show_usage
                exit 1
            fi
            run_file_coverage "$2"
            ;;
        --badge)
            generate_badge
            ;;
        --help)
            show_usage
            ;;
        *)
            echo "Unknown option: $option"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
