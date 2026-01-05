#!/bin/bash

# ΣLANG Phase 1: Automated Testing Pipeline
# Runs full test suite with parallel execution, coverage, and performance metrics

set -e  # Exit on any error

echo "🚀 ΣLANG Phase 1: Automated Testing Pipeline"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Environment: $(uname -a)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Not in ΣLANG project root directory. Please run from project root."
    exit 1
fi

print_status "Starting comprehensive test suite..."

# Parse command line arguments
PARALLEL=true
COVERAGE=true
PERFORMANCE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-parallel)
            PARALLEL=false
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --no-performance)
            PERFORMANCE=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --no-parallel    Disable parallel test execution"
            echo "  --no-coverage    Skip coverage analysis"
            echo "  --no-performance Skip performance benchmarks"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONDONTWRITEBYTECODE=1

# Create reports directory
REPORTS_DIR="test_reports/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORTS_DIR"

print_status "Reports will be saved to: $REPORTS_DIR"

# Function to run tests with coverage
run_tests() {
    local test_type=$1
    local test_args=$2

    print_status "Running $test_type tests..."

    if [ "$PARALLEL" = true ]; then
        test_args="$test_args -n auto"
    fi

    if [ "$COVERAGE" = true ]; then
        coverage run --source=sigmalang -m pytest $test_args \
            --tb=short \
            --strict-markers \
            --disable-warnings \
            --junitxml="$REPORTS_DIR/${test_type}_results.xml" \
            2>&1 | tee "$REPORTS_DIR/${test_type}_output.log"
    else
        python -m pytest $test_args \
            --tb=short \
            --strict-markers \
            --disable-warnings \
            --junitxml="$REPORTS_DIR/${test_type}_results.xml" \
            2>&1 | tee "$REPORTS_DIR/${test_type}_output.log"
    fi

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        print_success "$test_type tests passed"
    else
        print_error "$test_type tests failed (exit code: $exit_code)"
        return $exit_code
    fi
}

# 1. Unit Tests
print_status "Step 1: Running unit tests..."
if ! run_tests "unit" "tests/ -m unit"; then
    print_error "Unit tests failed. Aborting."
    exit 1
fi

# 2. Integration Tests
print_status "Step 2: Running integration tests..."
if ! run_tests "integration" "tests/ -m integration"; then
    print_error "Integration tests failed. Aborting."
    exit 1
fi

# 3. End-to-End Tests
print_status "Step 3: Running end-to-end tests..."
if ! run_tests "e2e" "tests/ -m e2e"; then
    print_error "E2E tests failed. Aborting."
    exit 1
fi

# 4. Performance Tests
if [ "$PERFORMANCE" = true ]; then
    print_status "Step 4: Running performance tests..."
    if ! run_tests "performance" "tests/ -m performance"; then
        print_warning "Performance tests failed, but continuing..."
    fi
fi

# 5. Generate Coverage Report
if [ "$COVERAGE" = true ]; then
    print_status "Step 5: Generating coverage report..."
    coverage combine 2>/dev/null || true
    coverage report --show-missing > "$REPORTS_DIR/coverage_report.txt"
    coverage html -d "$REPORTS_DIR/coverage_html"
    coverage xml -o "$REPORTS_DIR/coverage.xml"

    # Check coverage threshold
    COVERAGE_PERCENT=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')
    if (( $(echo "$COVERAGE_PERCENT < 95" | bc -l) )); then
        print_warning "Coverage is below 95%: ${COVERAGE_PERCENT}%"
    else
        print_success "Coverage meets threshold: ${COVERAGE_PERCENT}%"
    fi
fi

# 6. Run Benchmarks
if [ "$PERFORMANCE" = true ]; then
    print_status "Step 6: Running performance benchmarks..."

    # Memory benchmarks
    python benchmark_buffer_pool.py > "$REPORTS_DIR/memory_benchmark.txt" 2>&1 || true
    python benchmark_streaming_demo.py > "$REPORTS_DIR/streaming_benchmark.txt" 2>&1 || true

    # Compression benchmarks
    python benchmark_phase4a2.py > "$REPORTS_DIR/compression_benchmark.txt" 2>&1 || true

    print_success "Benchmarks completed"
fi

# 7. Static Analysis
print_status "Step 7: Running static analysis..."

# Type checking
mypy sigmalang/ --ignore-missing-imports > "$REPORTS_DIR/mypy_report.txt" 2>&1 || true

# Linting
flake8 sigmalang/ --max-line-length=100 > "$REPORTS_DIR/flake8_report.txt" 2>&1 || true

# Import sorting check
isort --check-only --diff sigmalang/ > "$REPORTS_DIR/isort_report.txt" 2>&1 || true

print_success "Static analysis completed"

# 8. Security Scan
print_status "Step 8: Running security scan..."
bandit -r sigmalang/ -f json -o "$REPORTS_DIR/security_scan.json" 2>/dev/null || true
print_success "Security scan completed"

# 9. Generate Summary Report
print_status "Step 9: Generating summary report..."

cat > "$REPORTS_DIR/SUMMARY.md" << EOF
# ΣLANG Test Suite Summary Report

**Generated:** $(date)
**Test Run:** $(basename "$REPORTS_DIR")

## Test Results

### Unit Tests
- **Status:** $(grep -c "PASSED" "$REPORTS_DIR/unit_output.log" 2>/dev/null || echo "Unknown")
- **Report:** [unit_results.xml](unit_results.xml)
- **Log:** [unit_output.log](unit_output.log)

### Integration Tests
- **Status:** $(grep -c "PASSED" "$REPORTS_DIR/integration_output.log" 2>/dev/null || echo "Unknown")
- **Report:** [integration_results.xml](integration_results.xml)
- **Log:** [integration_output.log](integration_output.log)

### E2E Tests
- **Status:** $(grep -c "PASSED" "$REPORTS_DIR/e2e_output.log" 2>/dev/null || echo "Unknown")
- **Report:** [e2e_results.xml](e2e_results.xml)
- **Log:** [e2e_output.log](e2e_output.log)

### Performance Tests
- **Status:** $(grep -c "PASSED" "$REPORTS_DIR/performance_output.log" 2>/dev/null || echo "Unknown")
- **Report:** [performance_results.xml](performance_results.xml)
- **Log:** [performance_output.log](performance_output.log)

## Code Quality

### Coverage
$(if [ "$COVERAGE" = true ]; then
    echo "- **Coverage:** ${COVERAGE_PERCENT}%"
    echo "- **Report:** [coverage_report.txt](coverage_report.txt)"
    echo "- **HTML:** [coverage_html/index.html](coverage_html/index.html)"
else
    echo "- **Coverage:** Skipped"
fi)

### Static Analysis
- **MyPy:** [mypy_report.txt](mypy_report.txt)
- **Flake8:** [flake8_report.txt](flake8_report.txt)
- **isort:** [isort_report.txt](isort_report.txt)

## Security
- **Bandit Scan:** [security_scan.json](security_scan.json)

## Performance Benchmarks
$(if [ "$PERFORMANCE" = true ]; then
    echo "- **Memory:** [memory_benchmark.txt](memory_benchmark.txt)"
    echo "- **Streaming:** [streaming_benchmark.txt](streaming_benchmark.txt)"
    echo "- **Compression:** [compression_benchmark.txt](compression_benchmark.txt)"
else
    echo "- **Benchmarks:** Skipped"
fi)

## Phase 1 Status

✅ **Automated Testing Pipeline:** Complete
$(if [ "$COVERAGE" = true ] && (( $(echo "$COVERAGE_PERCENT >= 95" | bc -l) )); then
    echo "✅ **Coverage Threshold:** Met (95%+)"
else
    echo "⚠️ **Coverage Threshold:** Not met or skipped"
fi)
$(if [ "$PERFORMANCE" = true ]; then
    echo "✅ **Performance Benchmarks:** Complete"
else
    echo "⚠️ **Performance Benchmarks:** Skipped"
fi)

**Next:** Infrastructure validation
EOF

print_success "Summary report generated: $REPORTS_DIR/SUMMARY.md"

# 10. Final Status
echo ""
echo "🎉 ΣLANG Phase 1 Testing Pipeline Complete!"
echo "=========================================="
echo "Reports saved to: $REPORTS_DIR"
echo ""
echo "Key Files:"
echo "  📊 Summary: $REPORTS_DIR/SUMMARY.md"
if [ "$COVERAGE" = true ]; then
    echo "  📈 Coverage: $REPORTS_DIR/coverage_report.txt"
fi
if [ "$PERFORMANCE" = true ]; then
    echo "  ⚡ Benchmarks: $REPORTS_DIR/*_benchmark.txt"
fi
echo ""
echo "Next Steps:"
echo "  1. Review test results in $REPORTS_DIR"
echo "  2. Run: ./scripts/validate_deployments.py --comprehensive"
echo "  3. Run: ./scripts/deploy_staging.sh --blue-green"
echo ""

# Exit with success
exit 0