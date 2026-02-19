# 🚀 ΣLANG Dashboard - Quick Start

Get your ΣLANG dashboard running in 2 minutes!

---

## Prerequisites

✅ Docker and Docker Compose installed
✅ Python 3.9+ (for Streamlit/FastAPI)
✅ Node.js 18+ (for React only)

---

## Option 1: Streamlit Dashboard (⭐ Recommended)

### Start Services First
```bash
cd s:/sigmalang
docker compose up -d
```

### Run Dashboard
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

**Opens:** http://localhost:8501

**Features:**
- 🏠 Home with system status
- ⚙️ Setup Wizard (4 steps)
- 🧪 API Testing interface
- 📊 Real-time Metrics
- ⚡ Performance benchmarks
- 📝 Documentation
- 🔧 Tools (tests, coverage, load testing)

---

## Option 2: FastAPI Dashboard

### Start Services First
```bash
cd s:/sigmalang
docker compose up -d
```

### Run Dashboard
```bash
cd dashboard
pip install fastapi uvicorn requests
python api_app.py
```

**Opens:** http://localhost:8501

**Same features as Streamlit with HTML/TailwindCSS UI**

---

## Option 3: React Dashboard

### Start Services First
```bash
cd s:/sigmalang
docker compose up -d
```

### Run Dashboard
```bash
cd dashboard/react-app
npm install
npm run dev
```

**Opens:** http://localhost:5173

**Modern React SPA with TypeScript, Recharts, and Vite**

---

## Verify Services Are Running

```bash
# Check Docker Compose
docker compose ps

# Expected output:
# sigmalang       8000/tcp
# redis           6379/tcp
# prometheus      9090/tcp
# grafana         3000/tcp
```

---

## Quick Links

Once running, access these endpoints:

- **Dashboard:** http://localhost:8501 (Streamlit/FastAPI) or http://localhost:5173 (React)
- **API Docs:** http://localhost:26080/docs
- **Prometheus:** http://localhost:26900
- **Grafana:** http://localhost:26910 (admin/admin)
- **API Health:** http://localhost:26080/health

---

## Test the Dashboard

1. **Open the Dashboard** (one of the three options above)
2. **Go to Setup Wizard** page
3. **Click "Run Checks"** to verify everything
4. **Go to Testing** page
5. **Try encoding some text** with different optimization levels
6. **Go to Tools** and run the test suite

---

## Docker Compose Services

| Service | Port | Purpose |
|---------|------|---------|
| ΣLANG API | 26080 | Main API server |
| Redis | 26500 | Caching layer |
| Prometheus | 26900 | Metrics collection |
| Grafana | 26910 | Metrics visualization |

---

## Troubleshooting

### Port Already in Use
```bash
# Streamlit - use different port
streamlit run app.py --server.port 8502

# FastAPI - use different port
# Edit api_app.py: DASHBOARD_PORT = 8502
```

### Services Won't Start
```bash
docker compose down
docker compose up -d --force-recreate
```

### API Not Responding
```bash
# Check if API is running
curl http://localhost:26080/health

# View logs
docker compose logs sigmalang
```

---

## Next Steps

- 📚 **Read DASHBOARD_SETUP_GUIDE.md** for detailed setup
- 🔧 **Check IMPLEMENTATION_SUMMARY.md** for architecture details
- 📊 **Visit docs/getting-started/** for ΣLANG usage
- 🚀 **Read PUBLICATION_GUIDE.md** for production deployment

---

## Need Help?

- **GitHub:** https://github.com/iamthegreatdestroyer/sigmalang
- **Issues:** https://github.com/iamthegreatdestroyer/sigmalang/issues
- **Documentation:** See docs/ folder

---

**Ready to go!** Choose your dashboard option above and get started in 2 minutes. 🎉

Version 1.0.0 | February 19, 2026 | Production Ready ✅
