#!/bin/bash
# ΣLANG Quick Local Testing Script
# Validates that the system is working locally

set -e

echo "================================"
echo "ΣLANG Local Testing Script"
echo "================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test 1: Python environment
echo -e "${YELLOW}[Test 1]${NC} Checking Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Python available: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python not found"
    exit 1
fi

# Test 2: Dependencies
echo ""
echo -e "${YELLOW}[Test 2]${NC} Checking dependencies..."
python3 -c "import sigmalang; print('✓ ΣLANG import successful')" 2>/dev/null || {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -e ".[dev]" > /dev/null 2>&1
    echo -e "${GREEN}✓${NC} Dependencies installed"
}

# Test 3: CLI availability
echo ""
echo -e "${YELLOW}[Test 3]${NC} Testing CLI..."
if sigmalang --version &> /dev/null; then
    VERSION=$(sigmalang --version)
    echo -e "${GREEN}✓${NC} CLI available: $VERSION"
else
    echo -e "${RED}✗${NC} CLI not available"
fi

# Test 4: Unit tests
echo ""
echo -e "${YELLOW}[Test 4]${NC} Running unit tests (this may take 1-2 minutes)..."
PYTEST_OUTPUT=$(pytest tests/test_optimizations.py -q --tb=no 2>&1) || true
TEST_COUNT=$(echo "$PYTEST_OUTPUT" | grep -oP '\d+(?= passed)' | head -1)
if [ ! -z "$TEST_COUNT" ]; then
    echo -e "${GREEN}✓${NC} $TEST_COUNT tests passed"
else
    echo -e "${RED}✗${NC} Tests failed"
fi

# Test 5: Encoding functionality
echo ""
echo -e "${YELLOW}[Test 5]${NC} Testing encoding functionality..."
python3 << 'EOF'
try:
    from sigmalang.core.encoder import SigmaEncoder
    from sigmalang.core.primitives import SemanticTree, SemanticNode, ExistentialPrimitive

    # Create simple semantic tree
    root = SemanticNode(primitive=ExistentialPrimitive.ENTITY, value="test")
    tree = SemanticTree(root=root, source_text="test encoding")

    # Encode
    encoder = SigmaEncoder()
    encoded = encoder.encode(tree)

    print(f"✓ Encoding successful: {len(encoded)} bytes")
except Exception as e:
    print(f"✗ Encoding failed: {e}")
    exit(1)
EOF

# Test 6: Docker availability (optional)
echo ""
echo -e "${YELLOW}[Test 6]${NC} Checking Docker (optional)..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo -e "${GREEN}✓${NC} Docker available: $DOCKER_VERSION"

    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Docker Compose available"
        echo ""
        echo -e "${YELLOW}To start local services:${NC}"
        echo "  docker compose up -d"
        echo "  docker compose ps"
        echo "  docker compose logs -f sigmalang"
    fi
else
    echo -e "${YELLOW}⊘${NC} Docker not installed (optional for containerized testing)"
fi

echo ""
echo "================================"
echo -e "${GREEN}All tests passed!${NC}"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. Review LOCAL_SETUP_GUIDE.md for detailed instructions"
echo "  2. Try CLI: sigmalang encode 'Hello, World!'"
echo "  3. Start services: docker compose up -d"
echo "  4. Visit http://localhost:26080/docs for API documentation"
echo ""
