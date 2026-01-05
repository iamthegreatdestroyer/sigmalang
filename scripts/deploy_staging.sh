#!/bin/bash

# ΣLANG Phase 1: Staging Deployment
# Automated blue-green deployment to staging environment

set -e  # Exit on any error

echo "🚀 ΣLANG Phase 1: Staging Deployment"
echo "==================================="
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
BLUE_GREEN=true
ROLLBACK_ON_FAILURE=true
NAMESPACE="neurectomy-staging"
TIMEOUT="600s"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --blue-green)
            BLUE_GREEN=true
            shift
            ;;
        --no-blue-green)
            BLUE_GREEN=false
            shift
            ;;
        --rollback-on-failure)
            ROLLBACK_ON_FAILURE=true
            shift
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE=false
            shift
            ;;
        --namespace=*)
            NAMESPACE="${1#*=}"
            shift
            ;;
        --timeout=*)
            TIMEOUT="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --blue-green          Enable blue-green deployment (default)"
            echo "  --no-blue-green       Disable blue-green deployment"
            echo "  --rollback-on-failure Enable automatic rollback on failure (default)"
            echo "  --no-rollback         Disable automatic rollback"
            echo "  --namespace=NAMESPACE Set deployment namespace (default: neurectomy-staging)"
            echo "  --timeout=TIMEOUT     Set deployment timeout (default: 600s)"
            echo "  --help                Show this help"
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

    # Check kubectl
    if ! kubectl cluster-info &>/dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check namespace exists or create it
    if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
        print_status "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    fi

    # Check if infrastructure directory exists
    if [ ! -d "infrastructure/kubernetes" ]; then
        print_error "Kubernetes infrastructure directory not found"
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Build and push Docker image
build_and_push_image() {
    print_status "Building and pushing Docker image..."

    local image_tag="sigmalang:$(date +%Y%m%d_%H%M%S)"
    local registry="localhost:5000"  # Local registry for staging

    # Build image
    if ! docker build -t "$registry/$image_tag" .; then
        print_error "Docker build failed"
        return 1
    fi

    # Push image
    if ! docker push "$registry/$image_tag"; then
        print_error "Docker push failed"
        return 1
    fi

    # Update kustomization with new image
    sed -i "s|newTag:.*|newTag: $image_tag|g" infrastructure/kubernetes/kustomization.yaml

    print_success "Image built and pushed: $registry/$image_tag"
}

# Blue-green deployment
blue_green_deploy() {
    print_status "Starting blue-green deployment..."

    local current_color=""
    local new_color=""

    # Determine current active deployment
    if kubectl get deployment sigmalang-blue -n "$NAMESPACE" &>/dev/null; then
        current_color="blue"
        new_color="green"
    elif kubectl get deployment sigmalang-green -n "$NAMESPACE" &>/dev/null; then
        current_color="green"
        new_color="blue"
    else
        # First deployment
        current_color=""
        new_color="blue"
    fi

    print_status "Current active: $current_color, Deploying: $new_color"

    # Create temporary kustomization for new deployment
    local temp_dir="/tmp/sigmalang-deploy-$(date +%s)"
    mkdir -p "$temp_dir"

    # Copy kustomization and modify for new deployment
    cp infrastructure/kubernetes/kustomization.yaml "$temp_dir/"
    cp -r infrastructure/kubernetes/deployments "$temp_dir/"
    cp -r infrastructure/kubernetes/configmaps "$temp_dir/" 2>/dev/null || true

    # Modify deployment name and labels
    local deployment_file="$temp_dir/deployments/sigmalang-deployment.yaml"
    if [ -f "$deployment_file" ]; then
        sed -i "s/name: sigmalang/name: sigmalang-$new_color/g" "$deployment_file"
        sed -i "s/app: sigmalang/app: sigmalang-$new_color/g" "$deployment_file"
        sed -i "s/version: .*/version: $new_color/g" "$deployment_file"
    fi

    # Deploy new version
    print_status "Deploying $new_color version..."
    if ! kubectl apply -k "$temp_dir" -n "$NAMESPACE" --timeout="$TIMEOUT"; then
        print_error "Deployment of $new_color version failed"
        cleanup_temp "$temp_dir"
        return 1
    fi

    # Wait for rollout to complete
    print_status "Waiting for $new_color deployment rollout..."
    if ! kubectl rollout status deployment/sigmalang-$new_color -n "$NAMESPACE" --timeout="$TIMEOUT"; then
        print_error "Rollout of $new_color version failed"
        if [ "$ROLLBACK_ON_FAILURE" = true ]; then
            rollback_deployment "$new_color"
        fi
        cleanup_temp "$temp_dir"
        return 1
    fi

    # Run health checks
    if ! run_health_checks "$new_color"; then
        print_error "Health checks failed for $new_color version"
        if [ "$ROLLBACK_ON_FAILURE" = true ]; then
            rollback_deployment "$new_color"
        fi
        cleanup_temp "$temp_dir"
        return 1
    fi

    # Switch traffic (update service selector)
    if [ -n "$current_color" ]; then
        print_status "Switching traffic from $current_color to $new_color..."
        switch_traffic "$current_color" "$new_color"
    fi

    # Wait for traffic switch to stabilize
    sleep 30

    # Final health check
    if ! run_health_checks "$new_color"; then
        print_error "Final health checks failed"
        if [ "$ROLLBACK_ON_FAILURE" = true ]; then
            switch_traffic "$new_color" "$current_color"
            rollback_deployment "$new_color"
        fi
        cleanup_temp "$temp_dir"
        return 1
    fi

    # Clean up old deployment
    if [ -n "$current_color" ]; then
        print_status "Cleaning up old $current_color deployment..."
        kubectl delete deployment sigmalang-$current_color -n "$NAMESPACE" --ignore-not-found=true
    fi

    cleanup_temp "$temp_dir"
    print_success "Blue-green deployment completed successfully"
}

# Simple deployment (no blue-green)
simple_deploy() {
    print_status "Starting simple deployment..."

    # Apply all manifests
    if ! kubectl apply -k infrastructure/kubernetes/ -n "$NAMESPACE" --timeout="$TIMEOUT"; then
        print_error "Deployment failed"
        return 1
    fi

    # Wait for rollout
    print_status "Waiting for deployment rollout..."
    if ! kubectl rollout status deployment/sigmalang -n "$NAMESPACE" --timeout="$TIMEOUT"; then
        print_error "Rollout failed"
        return 1
    fi

    # Run health checks
    if ! run_health_checks "sigmalang"; then
        print_error "Health checks failed"
        return 1
    fi

    print_success "Simple deployment completed successfully"
}

# Switch traffic between deployments
switch_traffic() {
    local from_color=$1
    local to_color=$2

    # Update service selector
    kubectl patch service sigmalang-service -n "$NAMESPACE" \
        -p "{\"spec\":{\"selector\":{\"app\":\"sigmalang-$to_color\"}}}"
}

# Rollback deployment
rollback_deployment() {
    local color=$1
    print_warning "Rolling back $color deployment..."
    kubectl delete deployment sigmalang-$color -n "$NAMESPACE" --ignore-not-found=true
}

# Run health checks
run_health_checks() {
    local deployment_name=$1
    print_status "Running health checks for $deployment_name..."

    # Wait for pods to be ready
    local pods_ready=""
    for i in {1..30}; do
        pods_ready=$(kubectl get pods -n "$NAMESPACE" -l "app=$deployment_name" \
            -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
        if [[ "$pods_ready" == "True" ]]; then
            break
        fi
        sleep 10
    done

    if [[ "$pods_ready" != "True" ]]; then
        print_error "Pods not ready within timeout"
        return 1
    fi

    # Test service endpoints
    local service_ip=$(kubectl get service sigmalang-service -n "$NAMESPACE" \
        -o jsonpath='{.spec.clusterIP}' 2>/dev/null)

    if [ -n "$service_ip" ]; then
        # Test health endpoint
        if ! kubectl run test-health --image=curlimages/curl --rm -i --restart=Never \
            -- curl -f "http://$service_ip:8000/health" --max-time 10 &>/dev/null; then
            print_error "Health check failed"
            return 1
        fi
    fi

    print_success "Health checks passed"
}

# Cleanup temporary files
cleanup_temp() {
    local temp_dir=$1
    rm -rf "$temp_dir"
}

# Generate deployment report
generate_report() {
    local deployment_type=$1
    local success=$2

    local report_dir="deployment_reports/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$report_dir"

    cat > "$report_dir/DEPLOYMENT_REPORT.md" << EOF
# ΣLANG Staging Deployment Report

**Generated:** $(date)
**Deployment Type:** $deployment_type
**Namespace:** $NAMESPACE
**Status:** $([ "$success" = true ] && echo "✅ SUCCESS" || echo "❌ FAILED")

## Deployment Details

### Configuration
- **Blue-Green:** $([ "$BLUE_GREEN" = true ] && echo "Enabled" || echo "Disabled")
- **Auto Rollback:** $([ "$ROLLBACK_ON_FAILURE" = true ] && echo "Enabled" || echo "Disabled")
- **Timeout:** $TIMEOUT

### Resources Deployed
$(kubectl get all -n "$NAMESPACE" -o name | sed 's/^/- /')

### Pod Status
$(kubectl get pods -n "$NAMESPACE" -o wide)

### Service Status
$(kubectl get services -n "$NAMESPACE" -o wide)

## Health Checks

### Pod Readiness
$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}')

### Service Endpoints
$(kubectl get endpoints -n "$NAMESPACE")

## Next Steps

$(if [ "$success" = true ]; then
    echo "✅ **Deployment Successful**"
    echo ""
    echo "**Ready for Phase 1 completion:**"
    echo "1. ✅ Staging deployment complete"
    echo "2. 🔄 Run: \`./scripts/load_test.sh --duration=1h\`"
    echo "3. 🔄 Monitor dashboards and logs"
    echo "4. 🔄 Proceed to Phase 2 when ready"
else
    echo "❌ **Deployment Failed**"
    echo ""
    echo "**Troubleshooting:**"
    echo "1. Check pod logs: \`kubectl logs -n $NAMESPACE deployment/sigmalang\`"
    echo "2. Check events: \`kubectl get events -n $NAMESPACE\`"
    echo "3. Verify cluster resources"
    echo "4. Re-run deployment after fixes"
fi)

---
**Report generated at $(date)**
EOF

    if [ "$success" = true ]; then
        print_success "Deployment report: $report_dir/DEPLOYMENT_REPORT.md"
    else
        print_error "Deployment report: $report_dir/DEPLOYMENT_REPORT.md"
    fi
}

# Main deployment flow
main() {
    print_status "ΣLANG Staging Deployment Starting..."
    echo "  Blue-Green: $([ "$BLUE_GREEN" = true ] && echo "Enabled" || echo "Disabled")"
    echo "  Auto Rollback: $([ "$ROLLBACK_ON_FAILURE" = true ] && echo "Enabled" || echo "Disabled")"
    echo "  Namespace: $NAMESPACE"
    echo "  Timeout: $TIMEOUT"
    echo ""

    check_prerequisites

    # Build and push image (optional, skip if using pre-built)
    if [ -f "Dockerfile" ]; then
        build_and_push_image
    fi

    local success=false

    if [ "$BLUE_GREEN" = true ]; then
        if blue_green_deploy; then
            success=true
        fi
    else
        if simple_deploy; then
            success=true
        fi
    fi

    generate_report "$([ "$BLUE_GREEN" = true ] && echo "Blue-Green" || echo "Simple")" "$success"

    if [ "$success" = true ]; then
        print_success "🎉 Staging deployment completed successfully!"
        echo ""
        echo "📋 Deployment Report: Check deployment_reports/ directory"
        echo ""
        echo "🚀 Ready for next step: Load testing"
        echo "   Run: ./scripts/load_test.sh --duration=1h --concurrency=1000"
        echo ""
        return 0
    else
        print_error "❌ Staging deployment failed"
        return 1
    fi
}

# Run main deployment
main "$@"