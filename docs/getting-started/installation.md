# Installation

## Requirements

- **Python**: 3.9 or higher
- **pip**: Latest version
- **OS**: Linux, macOS, or Windows
- **Optional**: Docker & Docker Compose for containerized deployment

## Standard Installation

Install ΣLANG from PyPI:

```bash
pip install sigmalang
```

## Development Installation

For development and testing:

```bash
# Clone repository
git clone https://github.com/iamthegreatdestroyer/sigmalang.git
cd sigmalang

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Docker Installation

The fastest way to get started with all services:

```bash
# Clone repository
git clone https://github.com/iamthegreatdestroyer/sigmalang.git
cd sigmalang

# Start all services
docker compose up -d

# Verify services
docker compose ps
```

**Available Services:**
- API Server: `http://localhost:26080`
- Prometheus Metrics: `http://localhost:26900`
- Grafana Dashboard: `http://localhost:26910` (admin/sigmalang)
- Redis Cache: `localhost:26500`

## Verify Installation

### Check CLI
```bash
sigmalang --version
sigmalang --help
```

### Check Python API
```python
from sigmalang.core.encoder import SigmaEncoder
print("ΣLANG imported successfully!")
```

### Quick Test
```bash
sigmalang encode "Hello, World!"
```

## Troubleshooting

### Port Already in Use
```bash
# On macOS/Linux
lsof -i :26080

# On Windows
netstat -ano | findstr :26080

# Change port in docker-compose.yml or use environment variable
export SIGMALANG_API_PORT=8001
```

### Python Version Issue
Ensure you're using Python 3.9+:
```bash
python --version  # Should be 3.9 or higher
```

### Permission Denied (Unix/Linux)
If you get permission errors, use `--user` flag or virtual environment:
```bash
pip install --user sigmalang
# OR
python -m venv venv
source venv/bin/activate
pip install sigmalang
```

## Next Steps

- Read the [Quick Start](quickstart.md) guide
- Explore [Basic Usage](basic-usage.md) examples
- Check the [API Reference](../api/overview.md)
