# ΣLANG Dashboard Setup Guide

Complete guide for setting up and running the ΣLANG Local Dashboard with multiple UI options.

## Overview

Three dashboard options are available for local setup and testing:

1. **Streamlit** - Simple, rapid development (Recommended for beginners)
2. **FastAPI + HTML/TailwindCSS** - Lightweight, production-ready
3. **React + TypeScript** - Modern, full-featured, scalable

## Quick Start (Streamlit)

### Prerequisites

- Docker and Docker Compose installed
- Python 3.9+
- 2GB+ available memory

### Installation & Running

```bash
# Install dependencies
cd dashboard
pip install -r requirements.txt

# Start the dashboard
streamlit run app.py

# Dashboard will open at http://localhost:8501
```

### Features

- 🏠 **Home**: System status and quick actions
- ⚙️ **Setup Wizard**: 4-step configuration and deployment
- 🧪 **Testing**: Interactive API endpoint testing
- 📊 **Metrics**: Real-time performance metrics with charts
- ⚡ **Performance**: Benchmarks and optimization tips
- 📝 **Documentation**: Embedded documentation sections
- 🔧 **Tools**: Test runner, coverage, load testing, logs

---

## Option 2: FastAPI Dashboard

### Prerequisites

- Docker and Docker Compose installed
- Python 3.9+
- pip packages: fastapi, uvicorn, requests

### Installation & Running

```bash
# Install dependencies
cd dashboard
pip install fastapi uvicorn requests

# Start the API server
python api_app.py

# Dashboard will be available at http://localhost:8501
```

### Features

- HTML/TailwindCSS frontend
- RESTful API backend
- Real-time health checks
- Docker service management
- API endpoint testing

### API Endpoints

```
GET  /              - Dashboard home
GET  /setup         - Setup wizard page
GET  /testing       - API testing interface
GET  /api/status    - API health status
POST /api/services/start  - Start Docker services
POST /api/services/stop   - Stop Docker services
GET  /api/services/status - Get service status
POST /api/test/encode     - Test encoding endpoint
POST /api/test/entities   - Test entity extraction
POST /api/test/analogy    - Test analogy solving
GET  /api/prerequisites   - Check prerequisites
POST /api/generate-env    - Generate .env file
GET  /api/verify          - Run verification checks
```

---

## Option 3: React + TypeScript Dashboard

### Prerequisites

- Node.js 18+
- npm or yarn package manager
- Docker and Docker Compose installed

### Installation & Development

```bash
# Navigate to React app directory
cd dashboard/react-app

# Install dependencies
npm install

# Start development server
npm run dev

# Dashboard will open at http://localhost:5173
```

### Building for Production

```bash
# Build the application
npm run build

# Preview production build
npm run preview
```

### Project Structure

```
react-app/
├── src/
│   ├── App.tsx          - Main application component
│   ├── main.tsx         - Entry point
│   ├── pages/
│   │   ├── Home.tsx     - Home page
│   │   ├── Setup.tsx    - Setup wizard
│   │   ├── Testing.tsx  - API testing
│   │   ├── Metrics.tsx  - Metrics dashboard
│   │   ├── Performance.tsx - Performance info
│   │   ├── Documentation.tsx - Documentation
│   │   └── Tools.tsx    - Tools and utilities
│   └── store/
│       └── apiStore.ts  - API client store (Zustand)
├── package.json         - Dependencies and scripts
└── vite.config.ts       - Vite configuration
```

### Key Dependencies

- **React 18**: UI library
- **React Router**: Navigation
- **Zustand**: State management
- **Recharts**: Data visualization
- **TailwindCSS**: Styling
- **Axios**: HTTP client
- **Vite**: Build tool

### Features

- Modern React architecture
- TypeScript for type safety
- Recharts for metrics visualization
- Responsive design
- RESTful API integration
- Multi-page SPA with routing

---

## Configuration

### Environment Setup

All dashboards connect to the ΣLANG API running on `localhost:26080`.

#### Configure Docker Compose

Ensure `docker-compose.yml` has the correct port mappings:

```yaml
services:
  sigmalang:
    ports:
      - "26080:8000"  # API
  grafana:
    ports:
      - "26910:3000"  # Grafana
  prometheus:
    ports:
      - "26900:9090"  # Prometheus
```

#### Start Services

```bash
# Start all services
docker compose up -d

# Verify services are running
docker compose ps

# View logs
docker compose logs -f sigmalang
```

### Environment Variables

Create `.env` file in project root (used by FastAPI dashboard):

```bash
SIGMALANG_API_HOST=http://localhost:26080
GRAFANA_URL=http://localhost:26910
PROMETHEUS_URL=http://localhost:26900
DASHBOARD_PORT=8501
```

---

## Usage Guide

### Streamlit Dashboard

#### Page: Home
- View system status and quick statistics
- One-click service start/stop
- Real-time API health indicator

#### Page: Setup Wizard
1. **Prerequisites**: Check Docker, Python, Git, disk space
2. **Configuration**: Generate and view `.env` file
3. **Docker Setup**: Start/stop services, view status
4. **Verify**: Run all health checks

#### Page: Testing
- **Encoding**: Test text encoding with different optimization levels
- **Entities**: Extract named entities from text
- **Analogy**: Solve word analogies (word1 : word2 :: word3 : ?)
- **Search**: Semantic search interface (coming soon)

#### Page: Metrics
- Real-time performance metrics
- Compression statistics
- Request rate trends
- Links to Prometheus and Grafana

#### Page: Performance
- Performance optimization tips
- Benchmark table (operation, speed, compression ratio)
- Load testing quick start
- Coverage report generation

#### Page: Documentation
- Getting Started guide
- API Reference
- Configuration options
- Deployment options
- Troubleshooting guide

#### Page: Tools
- **Run Tests**: Execute full test suite (1,656 tests)
- **Generate Coverage**: Fast (5 min) or Full (10 min) coverage reports
- **Load Testing**: Run baseline, normal, peak, spike, or endurance tests
- **Generate SDKs**: Create SDKs for TypeScript, Java, Python, Go
- **View Logs**: View logs from Docker services

### FastAPI Dashboard

Access at `http://localhost:8501`:

- Same features as Streamlit but with HTML/TailwindCSS frontend
- Responsive design optimized for desktop and mobile
- Direct REST API backend integration

### React Dashboard

Access at `http://localhost:5173` (development):

- Full-featured modern web interface
- Real-time data visualization with Recharts
- Responsive sidebar navigation
- Professional styling with TailwindCSS
- TypeScript type safety throughout

---

## Troubleshooting

### Services Won't Start

```bash
# Check Docker is running
docker ps

# Check if ports are already in use
lsof -i :26080  # API
lsof -i :26910  # Grafana
lsof -i :26900  # Prometheus

# View Docker logs
docker compose logs sigmalang

# Restart services
docker compose down
docker compose up -d
```

### API Connection Failed

```bash
# Verify API is running
curl http://localhost:26080/health

# Check network connectivity
docker exec sigmalang curl http://localhost:8000/health

# Review API logs
docker compose logs sigmalang --tail 50
```

### Streamlit Issues

```bash
# Clear Streamlit cache
streamlit cache clear

# Run with verbose output
streamlit run app.py --logger.level=debug

# Check logs
ps aux | grep streamlit
```

### React Build Issues

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear build cache
rm -rf dist node_modules/.vite

# Rebuild
npm run build
```

### FastAPI Connection Issues

```bash
# Verify API is accessible
curl http://localhost:8501/api/status

# Check uvicorn process
ps aux | grep uvicorn

# Restart API server
pkill uvicorn
python api_app.py
```

---

## Performance Considerations

### Streamlit
- **Pros**: Easiest to set up, minimal dependencies
- **Cons**: Rerun-based architecture, can be slower for large datasets
- **Best for**: Development, demos, data analysis

### FastAPI
- **Pros**: Fast, scalable, production-ready
- **Cons**: More setup required, HTML creation needed
- **Best for**: Production deployments, API-first applications

### React
- **Pros**: Modern, feature-rich, highly responsive
- **Cons**: Requires Node.js, larger build size
- **Best for**: Complex UIs, real-time applications, teams with frontend expertise

---

## Deployment Options

### Local Development

```bash
# Start services
docker compose up -d

# Run dashboard
streamlit run dashboard/app.py
# OR
python dashboard/api_app.py
# OR
cd dashboard/react-app && npm run dev
```

### Docker Container

```bash
# Build dashboard image
docker build -t sigmalang-dashboard -f Dockerfile.dashboard .

# Run container
docker run -p 8501:8501 \
  -v ~/.env:/app/.env:ro \
  sigmalang-dashboard
```

### Kubernetes

```bash
# Deploy with Helm
helm install sigmalang-dashboard sigmalang/dashboard \
  --set dashboard.type=react \
  --set image.tag=latest
```

---

## Integration with ΣLANG Services

### API Server (Port 26080)

- Health: `GET /health`
- Encoding: `POST /api/encode`
- Entities: `POST /api/entities`
- Analogy: `POST /api/analogy`
- Docs: `GET /docs` (Swagger UI)

### Prometheus (Port 26900)

- Metrics: `http://localhost:26900`
- Targets: `http://localhost:26900/targets`
- Queries: `http://localhost:26900/graph`

### Grafana (Port 26910)

- Dashboard: `http://localhost:26910`
- Default credentials: admin / admin
- Dashboards for: API metrics, compression, cache performance

---

## Advanced Configuration

### SSL/TLS Support

For FastAPI and React dashboards, add nginx reverse proxy:

```yaml
# docker-compose.yml
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
```

### Authentication

Add authentication middleware:

```python
# FastAPI
from fastapi.security import HTTPBearer, HTTPAuthCredentials

security = HTTPBearer()

@app.get("/api/protected")
async def protected_route(credentials: HTTPAuthCredentials = Depends(security)):
    return {"message": "Authenticated"}
```

### Rate Limiting

```python
# FastAPI with SlowAPI
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/test/encode", dependencies=[Depends(limiter.limit("10/minute"))])
async def test_encode_endpoint():
    ...
```

---

## Monitoring & Logging

### View Dashboard Logs

```bash
# Streamlit
streamlit run app.py 2>&1 | tee dashboard.log

# FastAPI
python api_app.py 2>&1 | tee dashboard.log

# React (development)
npm run dev 2>&1 | tee dashboard.log

# React (production)
docker compose logs dashboard -f
```

### Monitor Metrics

```bash
# Check API metrics
curl http://localhost:26900/api/v1/query?query=sigmalang_request_count_total

# View Grafana dashboards
open http://localhost:26910
```

---

## Support & Resources

- **GitHub**: https://github.com/iamthegreatdestroyer/sigmalang
- **Issues**: https://github.com/iamthegreatdestroyer/sigmalang/issues
- **Documentation**: https://sigmalang.io
- **API Docs**: http://localhost:26080/docs

---

**Last Updated**: February 19, 2026
**Version**: 1.0.0
**Status**: Production Ready
