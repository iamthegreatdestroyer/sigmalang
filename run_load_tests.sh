#!/bin/bash
# ΣLANG Load Testing Script
# Runs various load test scenarios

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
HOST="${1:-http://localhost:26080}"
RESULTS_DIR="load_test_results"

# Create results directory
mkdir -p "$RESULTS_DIR"

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Check if API is running
check_api() {
    print_info "Checking if API is running at $HOST..."
    if ! curl -s "$HOST/health" > /dev/null; then
        echo -e "${RED}✗ API not responding at $HOST${NC}"
        echo "Start with: docker compose up -d"
        exit 1
    fi
    print_success "API is healthy"
}

# Check if Locust is installed
check_locust() {
    if ! command -v locust &> /dev/null; then
        echo -e "${RED}✗ Locust not installed${NC}"
        echo "Install with: pip install locust"
        exit 1
    fi
    print_success "Locust is installed"
}

# Run baseline test
run_baseline() {
    print_header "Baseline Performance Test"
    print_info "Running with 5 users, 1 user/sec spawn rate, 3 minutes"
    print_info "This tests basic functionality and response times"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    CSVFILE="$RESULTS_DIR/baseline_$TIMESTAMP.csv"

    locust -f load_test.py \
        --host="$HOST" \
        --users 5 \
        --spawn-rate 1 \
        --run-time 3m \
        --headless \
        --csv="$CSVFILE"

    print_success "Baseline test complete: $CSVFILE"
}

# Run normal load test
run_normal() {
    print_header "Normal Load Test"
    print_info "Running with 50 users, 5 users/sec spawn rate, 10 minutes"
    print_info "This simulates typical production traffic"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    CSVFILE="$RESULTS_DIR/normal_$TIMESTAMP.csv"

    locust -f load_test.py \
        --host="$HOST" \
        --users 50 \
        --spawn-rate 5 \
        --run-time 10m \
        --headless \
        --csv="$CSVFILE"

    print_success "Normal load test complete: $CSVFILE"
}

# Run peak load test
run_peak() {
    print_header "Peak Load Test"
    print_info "Running with 200 users, 10 users/sec spawn rate, 15 minutes"
    print_info "This tests system under high concurrent load"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    CSVFILE="$RESULTS_DIR/peak_$TIMESTAMP.csv"

    locust -f load_test.py \
        --host="$HOST" \
        --users 200 \
        --spawn-rate 10 \
        --run-time 15m \
        --headless \
        --csv="$CSVFILE"

    print_success "Peak load test complete: $CSVFILE"
}

# Run spike test
run_spike() {
    print_header "Spike Test"
    print_info "Running with 100 users, 50 users/sec spawn rate, 5 minutes"
    print_info "This simulates sudden traffic surge"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    CSVFILE="$RESULTS_DIR/spike_$TIMESTAMP.csv"

    locust -f load_test.py \
        --host="$HOST" \
        --users 100 \
        --spawn-rate 50 \
        --run-time 5m \
        --headless \
        --csv="$CSVFILE"

    print_success "Spike test complete: $CSVFILE"
}

# Run endurance test
run_endurance() {
    print_header "Endurance Test"
    print_info "Running with 30 users, 3 users/sec spawn rate, 30 minutes"
    print_info "This tests stability over extended period"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    CSVFILE="$RESULTS_DIR/endurance_$TIMESTAMP.csv"

    locust -f load_test.py \
        --host="$HOST" \
        --users 30 \
        --spawn-rate 3 \
        --run-time 30m \
        --headless \
        --csv="$CSVFILE"

    print_success "Endurance test complete: $CSVFILE"
}

# Run interactive test with UI
run_interactive() {
    print_header "Interactive Load Test"
    print_info "Starting Locust with web UI"
    print_info "Open http://localhost:8089 in your browser"

    locust -f load_test.py --host="$HOST"
}

# Run all tests
run_all() {
    print_header "Running All Load Tests"
    print_info "This will take approximately 1 hour"

    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cancelled"
        return
    fi

    run_baseline
    echo ""
    run_normal
    echo ""
    run_peak
    echo ""
    run_spike
}

# Show usage
show_usage() {
    cat << EOF
ΣLANG Load Testing Script

Usage: $0 [COMMAND] [HOST]

Commands:
    baseline      Run baseline performance test (5 users, 3 min)
    normal        Run normal load test (50 users, 10 min)
    peak          Run peak load test (200 users, 15 min)
    spike         Run spike test (100 users quick ramp, 5 min)
    endurance     Run endurance test (30 users, 30 min)
    interactive   Run interactive test with web UI
    all           Run all tests sequentially (~1 hour)
    help          Show this help message

Options:
    HOST          API host (default: http://localhost:26080)

Examples:
    $0 baseline                          # Run baseline against localhost
    $0 normal http://192.168.1.100:8000 # Run normal test against remote
    $0 all                               # Run all tests

Results are saved to: $RESULTS_DIR/

For detailed documentation, see LOAD_TESTING_GUIDE.md

EOF
}

# Main
main() {
    local command="${1:-help}"

    # Check prerequisites
    check_locust
    check_api

    case "$command" in
        baseline)
            run_baseline
            ;;
        normal)
            run_normal
            ;;
        peak)
            run_peak
            ;;
        spike)
            run_spike
            ;;
        endurance)
            run_endurance
            ;;
        interactive)
            run_interactive
            ;;
        all)
            run_all
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

# Run main
main "$@"
