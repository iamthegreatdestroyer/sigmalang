# ΣLANG Dashboard

User-friendly dashboards for local setup, testing, and monitoring of ΣLANG.

**Three implementations available:**
1. 🟠 **Streamlit** - Simplest, perfect for rapid development
2. 🔵 **FastAPI + HTML** - Lightweight, production-ready web app
3. ⚛️ **React + TypeScript** - Modern, full-featured SPA

## Quick Start

### Option 1: Streamlit (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py

# Open browser to http://localhost:8501
```

**Pros:**
- ✅ Easiest to set up
- ✅ Minimal dependencies
- ✅ Great for prototyping
- ✅ Built-in widgets

**Cons:**
- ⚠️ Page reruns on every interaction
- ⚠️ Limited for large-scale apps

---

### Option 2: FastAPI

```bash
# Install dependencies
pip install fastapi uvicorn requests

# Run server
python api_app.py

# Open browser to http://localhost:8501
```

**Pros:**
- ✅ Fast and scalable
- ✅ Production-ready
- ✅ RESTful API
- ✅ Professional HTML/CSS

**Cons:**
- ⚠️ More code to maintain
- ⚠️ Manual HTML templates

---

### Option 3: React

```bash
# Navigate to React app
cd react-app

# Install dependencies
npm install

# Start development server
npm run dev

# Open browser to http://localhost:5173
```

**Pros:**
- ✅ Modern architecture
- ✅ TypeScript safety
- ✅ Rich component ecosystem
- ✅ Scalable and maintainable

**Cons:**
- ⚠️ Requires Node.js
- ⚠️ Larger build size
- ⚠️ Steeper learning curve

---

## Project Structure

```
dashboard/
├── app.py                    # Streamlit application
├── api_app.py              # FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── react-app/             # React TypeScript application
    ├── src/
    │   ├── App.tsx        # Main component
    │   ├── main.tsx       # Entry point
    │   ├── index.css      # Global styles
    │   ├── pages/         # Page components
    │   │   ├── Home.tsx
    │   │   ├── Setup.tsx
    │   │   ├── Testing.tsx
    │   │   ├── Metrics.tsx
    │   │   ├── Performance.tsx
    │   │   ├── Documentation.tsx
    │   │   └── Tools.tsx
    │   └── store/         # State management
    │       └── apiStore.ts
    ├── package.json       # Dependencies
    ├── vite.config.ts     # Vite configuration
    ├── tsconfig.json      # TypeScript config
    └── index.html         # HTML template
```

---

## Features

### All Dashboards Include

- 🏠 **Home** - System status and quick actions
- ⚙️ **Setup Wizard** - 4-step configuration
- 🧪 **Testing** - API endpoint testing interface
- 📊 **Metrics** - Real-time performance metrics
- ⚡ **Performance** - Benchmarks and optimization tips
- 📝 **Documentation** - Integrated documentation
- 🔧 **Tools** - Test runner, coverage, load testing, logs

### Dashboard-Specific Features

#### Streamlit
- Interactive widgets
- Real-time data updates
- Built-in charting
- Simple state management

#### FastAPI
- RESTful API endpoints
- TailwindCSS styling
- HTML5 templates
- Responsive design

#### React
- Component-based architecture
- Advanced routing
- State management with Zustand
- Recharts data visualization
- TypeScript type safety

---

## Configuration

### API Endpoints

All dashboards connect to the ΣLANG API at `localhost:26080`:

```
GET  /health                 - Health check
GET  /health/detailed        - Detailed health info
POST /api/encode             - Test encoding
POST /api/entities           - Test entity extraction
POST /api/analogy            - Test analogy solving
GET  /docs                   - Swagger UI
GET  /metrics                - Prometheus metrics
```

### Related Services

```
API Server    → http://localhost:26080
Prometheus    → http://localhost:26900
Grafana       → http://localhost:26910
Dashboard     → http://localhost:8501 (Streamlit/FastAPI)
              or http://localhost:5173 (React)
```

### Environment Variables

```bash
SIGMALANG_API_HOST=http://localhost:26080
GRAFANA_URL=http://localhost:26910
PROMETHEUS_URL=http://localhost:26900
DASHBOARD_PORT=8501
```

---

## Development

### Streamlit Development

```bash
# Auto-reloads on file changes
streamlit run app.py

# Configuration file at ~/.streamlit/config.toml
# Clear cache: streamlit cache clear
```

### FastAPI Development

```bash
# Auto-reloads with uvicorn --reload
uvicorn api_app:app --reload --host 0.0.0.0 --port 8501

# Swagger UI at http://localhost:8501/docs
# ReDoc at http://localhost:8501/redoc
```

### React Development

```bash
# Start dev server with HMR
npm run dev

# TypeScript checking
npx tsc --noEmit

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## Deployment

### Docker

```dockerfile
# Streamlit
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]

# FastAPI
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["python", "api_app.py"]

# React
FROM node:18 as build
WORKDIR /app
COPY react-app/package*.json .
RUN npm install
COPY react-app .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose

```yaml
services:
  dashboard-streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - SIGMALANG_API_HOST=http://sigmalang:8000

  dashboard-fastapi:
    build:
      context: .
      dockerfile: Dockerfile.fastapi
    ports:
      - "8502:8501"
    environment:
      - SIGMALANG_API_HOST=http://sigmalang:8000

  dashboard-react:
    build:
      context: .
      dockerfile: Dockerfile.react
    ports:
      - "80:80"
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use a different port
streamlit run app.py --server.port 8502
```

### API Connection Error

```bash
# Check API is running
curl http://localhost:26080/health

# Check network
docker network ls
docker inspect bridge

# Restart services
docker compose down
docker compose up -d
```

### Streamlit Issues

```bash
# Clear cache
streamlit cache clear

# Reset configuration
rm -rf ~/.streamlit

# Run in debug mode
streamlit run app.py --logger.level=debug
```

### FastAPI Issues

```bash
# Check uvicorn process
ps aux | grep uvicorn

# View logs
uvicorn api_app:app --reload --log-level debug

# Test endpoint
curl http://localhost:8501/health
```

### React Issues

```bash
# Clear cache
rm -rf node_modules dist
npm install

# TypeScript errors
npx tsc --noEmit

# Build issues
npm run build --verbose
```

---

## Performance

### Streamlit
- Lightweight, ~30MB
- Suitable for small-medium dashboards
- Ideal for development/prototyping

### FastAPI
- Minimal footprint, ~20MB
- High performance, ~10,000 req/s
- Production-ready

### React
- Build size ~100-200KB (gzipped ~30-50KB)
- Fast rendering, ~60fps
- Best for complex UIs

---

## Testing

### Unit Tests

```bash
# Streamlit - Use pytest
pytest tests/

# FastAPI - Use pytest with TestClient
pytest tests/ -v

# React - Use Vitest
npm run test
```

### Integration Tests

```bash
# Test all dashboards together
./tests/integration_test.sh
```

### End-to-End Tests

```bash
# Playwright for React
npx playwright test

# Selenium for Streamlit/FastAPI
pytest tests/e2e/ -v
```

---

## Contributing

1. Choose a dashboard option
2. Make changes in respective directory
3. Test thoroughly
4. Submit PR with description

---

## License

MIT License - See LICENSE file

---

## Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [React Docs](https://react.dev)
- [Vite Docs](https://vitejs.dev)
- [TailwindCSS Docs](https://tailwindcss.com)
- [ΣLANG GitHub](https://github.com/iamthegreatdestroyer/sigmalang)

---

**Created**: February 19, 2026
**Version**: 1.0.0
**Status**: Production Ready
