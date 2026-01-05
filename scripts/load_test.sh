#!/bin/bash

# ΣLANG Phase 1: Load Testing
# Automated performance validation with configurable load

set -e  # Exit on any error

echo "⚡ ΣLANG Phase 1: Load Testing"
echo "============================"
echo "Timestamp: $(date)"
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

# Default values
DURATION="1h"
CONCURRENCY=1000
REPORT=true
SERVICE_URL=""
NAMESPACE="neurectomy-staging"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration=*)
            DURATION="${1#*=}"
            shift
            ;;
        --concurrency=*)
            CONCURRENCY="${1#*=}"
            shift
            ;;
        --no-report)
            REPORT=false
            shift
            ;;
        --service-url=*)
            SERVICE_URL="${1#*=}"
            shift
            ;;
        --namespace=*)
            NAMESPACE="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --duration=DURATION    Test duration (default: 1h)"
            echo "  --concurrency=NUM      Number of concurrent requests (default: 1000)"
            echo "  --no-report            Skip report generation"
            echo "  --service-url=URL      Direct service URL (auto-detect if not provided)"
            echo "  --namespace=NAMESPACE  Kubernetes namespace (default: neurectomy-staging)"
            echo "  --help                 Show this help"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check if deployment exists
    if ! kubectl get deployment -n "$NAMESPACE" 2>/dev/null | grep -q sigmalang; then
        print_error "No ΣLANG deployment found in namespace $NAMESPACE"
        print_error "Run staging deployment first: ./scripts/deploy_staging.sh"
        exit 1
    fi

    # Get service URL if not provided
    if [ -z "$SERVICE_URL" ]; then
        SERVICE_URL=$(kubectl get service sigmalang-service -n "$NAMESPACE" \
            -o jsonpath='{.spec.clusterIP}:{.spec.ports[0].port}' 2>/dev/null)

        if [ -z "$SERVICE_URL" ]; then
            print_error "Could not determine service URL"
            exit 1
        fi
    fi

    # Check for load testing tools
    local tools=("curl" "python3")
    local missing_tools=()

    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    print_success "Prerequisites check passed"
    print_status "Service URL: http://$SERVICE_URL"
}

# Create test data
create_test_data() {
    print_status "Creating test data..."

    # Create directory for test data
    TEST_DATA_DIR="/tmp/sigmalang-load-test-$(date +%s)"
    mkdir -p "$TEST_DATA_DIR"

    # Generate sample texts of different sizes
    cat > "$TEST_DATA_DIR/small.txt" << 'EOF'
Hello, world! This is a small test message for compression testing.
EOF

    cat > "$TEST_DATA_DIR/medium.txt" << 'EOF'
The quick brown fox jumps over the lazy dog. This is a medium-sized test message that contains various words and punctuation marks. It should provide a good test case for the compression algorithm to work with. The message includes multiple sentences and should compress reasonably well.
EOF

    cat > "$TEST_DATA_DIR/large.txt" << 'EOF'
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

This is a much larger text sample designed to test the compression capabilities of the ΣLANG system. It contains multiple paragraphs and various types of content including technical terms, common phrases, and structured text. The compression algorithm should be able to identify patterns and reduce redundancy effectively.

Technical details: The ΣLANG compression system uses semantic encoding techniques to represent meaning directly rather than relying on traditional token-based approaches. This allows for much higher compression ratios while maintaining semantic fidelity.
EOF

    print_success "Test data created in $TEST_DATA_DIR"
}

# Run load test using curl and parallel processes
run_load_test() {
    print_status "Starting load test..."
    print_status "Duration: $DURATION"
    print_status "Concurrency: $CONCURRENCY"
    print_status "Service: http://$SERVICE_URL"

    local start_time=$(date +%s)
    local end_time=$((start_time + $(duration_to_seconds "$DURATION")))

    # Create results directory
    RESULTS_DIR="load_test_results/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RESULTS_DIR"

    # Initialize counters
    local total_requests=0
    local successful_requests=0
    local failed_requests=0
    local total_response_time=0
    local min_response_time=999999
    local max_response_time=0

    # Response time tracking
    local response_times_file="$RESULTS_DIR/response_times.txt"
    touch "$response_times_file"

    print_status "Load test running... (Ctrl+C to stop early)"

    # Trap SIGINT for clean shutdown
    trap 'print_status "Load test interrupted by user"; generate_report; exit 0' INT

    # Main load testing loop
    while [ $(date +%s) -lt $end_time ]; do
        # Launch concurrent requests
        for ((i=1; i<=CONCURRENCY; i++)); do
            (
                # Select random test file
                local test_files=("$TEST_DATA_DIR"/*.txt)
                local random_file=${test_files[RANDOM % ${#test_files[@]}]}

                # Measure response time
                local request_start=$(date +%s%N)
                local response=$(curl -s -w "%{http_code}\n%{time_total}" \
                    -X POST \
                    -H "Content-Type: application/json" \
                    -d "{\"text\": \"$(cat "$random_file" | tr '\n' ' ' | sed 's/"/\\"/g')\"}" \
                    "http://$SERVICE_URL/v1/encode" 2>/dev/null)

                local request_end=$(date +%s%N)
                local response_time=$(( (request_end - request_start) / 1000000 )) # Convert to milliseconds

                # Parse response
                local status_code=$(echo "$response" | head -1)
                local curl_time=$(echo "$response" | tail -1)

                # Update counters
                echo "$response_time" >> "$response_times_file"

                if [ "$status_code" = "200" ]; then
                    ((successful_requests++))
                else
                    ((failed_requests++))
                fi

                ((total_requests++))
            ) &
        done

        # Wait for all requests in this batch to complete
        wait

        # Progress update every 10 seconds
        local current_time=$(date +%s)
        if (( (current_time - start_time) % 10 == 0 )); then
            local elapsed=$((current_time - start_time))
            local remaining=$((end_time - current_time))
            print_status "Progress: ${elapsed}s elapsed, ${remaining}s remaining | Total: $total_requests requests"
        fi

        # Small delay to prevent overwhelming the system
        sleep 0.1
    done

    print_success "Load test completed"
    print_status "Total requests: $total_requests"
    print_status "Successful: $successful_requests"
    print_status "Failed: $failed_requests"
}

# Convert duration string to seconds
duration_to_seconds() {
    local duration=$1
    local unit=${duration: -1}
    local value=${duration:0:-1}

    case $unit in
        s) echo $value ;;
        m) echo $((value * 60)) ;;
        h) echo $((value * 3600)) ;;
        *) echo 3600 ;; # Default to 1 hour
    esac
}

# Generate load test report
generate_report() {
    if [ ! -f "$response_times_file" ]; then
        print_error "No response time data found"
        return 1
    fi

    print_status "Generating load test report..."

    # Calculate statistics
    local response_times=($(sort -n "$response_times_file"))
    local count=${#response_times[@]}

    if [ $count -eq 0 ]; then
        print_error "No response times recorded"
        return 1
    fi

    # Calculate percentiles
    local p50_index=$((count * 50 / 100))
    local p95_index=$((count * 95 / 100))
    local p99_index=$((count * 99 / 100))

    local p50=${response_times[$p50_index]}
    local p95=${response_times[$p95_index]}
    local p99=${response_times[$p99_index]}

    # Calculate average
    local sum=0
    for time in "${response_times[@]}"; do
        sum=$((sum + time))
    done
    local avg=$((sum / count))

    # Calculate requests per second
    local test_duration=$(duration_to_seconds "$DURATION")
    local rps=$((total_requests / test_duration))

    cat > "$RESULTS_DIR/LOAD_TEST_REPORT.md" << EOF
# ΣLANG Load Test Report

**Generated:** $(date)
**Test Duration:** $DURATION
**Concurrency:** $CONCURRENCY
**Service URL:** http://$SERVICE_URL

## Test Results

### Request Statistics
- **Total Requests:** $total_requests
- **Successful Requests:** $successful_requests
- **Failed Requests:** $failed_requests
- **Success Rate:** $((successful_requests * 100 / total_requests))%

### Performance Metrics
- **Requests/Second:** $rps
- **Average Response Time:** ${avg}ms
- **Median Response Time (P50):** ${p50}ms
- **95th Percentile (P95):** ${p95}ms
- **99th Percentile (P99):** ${p99}ms

### Response Time Distribution
$(python3 -c "
import numpy as np
times = [int(x.strip()) for x in open('$response_times_file').readlines() if x.strip()]
if times:
    print(f'- **Min:** {min(times)}ms')
    print(f'- **Max:** {max(times)}ms')
    print(f'- **Std Dev:** {int(np.std(times))}ms')
else:
    print('- No data available')
" 2>/dev/null || echo "- Statistics calculation failed")

## Phase 1 Status

✅ **Load Testing:** Complete
$(if [ $((successful_requests * 100 / total_requests)) -ge 95 ]; then
    echo "✅ **Success Rate:** ≥95%"
else
    echo "⚠️ **Success Rate:** <95%"
fi)
$(if [ $p95 -le 1000 ]; then
    echo "✅ **P95 Latency:** ≤1000ms"
else
    echo "⚠️ **P95 Latency:** >1000ms"
fi)

## Recommendations

$(if [ $rps -ge 100 ]; then
    echo "✅ **Throughput:** Good (≥100 RPS)"
elif [ $rps -ge 50 ]; then
    echo "⚠️ **Throughput:** Moderate (50-99 RPS)"
else
    echo "❌ **Throughput:** Low (<50 RPS)"
fi)

$(if [ $p95 -le 500 ]; then
    echo "✅ **Latency:** Excellent (≤500ms P95)"
elif [ $p95 -le 1000 ]; then
    echo "✅ **Latency:** Good (≤1000ms P95)"
elif [ $p95 -le 2000 ]; then
    echo "⚠️ **Latency:** Acceptable (≤2000ms P95)"
else
    echo "❌ **Latency:** Poor (>2000ms P95)"
fi)

## Next Steps

$(if [ $((successful_requests * 100 / total_requests)) -ge 95 ] && [ $p95 -le 1000 ]; then
    echo "**🎉 Phase 1 Complete!**"
    echo ""
    echo "**Ready for production:**"
    echo "1. ✅ Automated testing pipeline"
    echo "2. ✅ Infrastructure validation"
    echo "3. ✅ Staging deployment"
    echo "4. ✅ Load testing"
    echo ""
    echo "**Next:** Phase 2 - Production Hardening"
else
    echo "**⚠️ Performance Issues Detected**"
    echo ""
    echo "**Actions Required:**"
    echo "1. Investigate failed requests"
    echo "2. Optimize response times"
    echo "3. Scale infrastructure if needed"
    echo "4. Re-run load test after fixes"
fi)

## Raw Data

- **Response Times:** [response_times.txt](response_times.txt)
- **Test Configuration:** Duration=$DURATION, Concurrency=$CONCURRENCY
- **Service:** http://$SERVICE_URL

---
**Load test completed at $(date)**
EOF

    print_success "Load test report generated: $RESULTS_DIR/LOAD_TEST_REPORT.md"

    # Clean up test data
    rm -rf "$TEST_DATA_DIR"
}

# Main load testing flow
main() {
    print_status "ΣLANG Load Testing Starting..."
    echo "  Duration: $DURATION"
    echo "  Concurrency: $CONCURRENCY"
    echo "  Service: http://$SERVICE_URL"
    echo "  Namespace: $NAMESPACE"
    echo ""

    check_prerequisites
    create_test_data
    run_load_test
    generate_report

    print_success "🎉 Load testing completed!"
    echo ""
    echo "📋 Load Test Report: Check load_test_results/ directory"
    echo ""
}

# Run main load test
main "$@"