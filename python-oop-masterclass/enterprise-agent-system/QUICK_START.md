# 🚀 Quick Start Guide

Get the Enterprise Agent System up and running in **5 minutes**!

---

## Prerequisites

Before you begin, ensure you have:
- ✅ Python 3.11 or higher
- ✅ OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- ✅ Git installed

**Optional** (for full features):
- Docker Desktop (recommended for easy setup)
- PostgreSQL, Redis, Weaviate (or use Docker Compose)

---

## Option 1: Quick Start with UV (Recommended) ⚡

**Fastest way to get started!**

### Step 1: Clone and Navigate
```bash
git clone https://github.com/your-org/enterprise-agent-system.git
cd enterprise-agent-system/python-oop-masterclass/enterprise-agent-system
```

### Step 2: Install UV (if not installed)
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 3: Set Up Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your favorite editor
```

**Required**: Set your OpenAI API key in `.env`:
```bash
OPENAI_API_KEY="sk-your-actual-api-key-here"
```

### Step 4: Install Dependencies
```bash
# Install with UV (fast!)
uv pip install -e ".[dev]"
```

### Step 5: Start Infrastructure (Docker)
```bash
# Start Redis, PostgreSQL, Weaviate, Kafka using Docker
docker-compose up -d
```

### Step 6: Run the Application
```bash
# Start the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 7: Test It!
Open your browser to:
- **API Docs**: http://localhost:8000/docs
- **API Root**: http://localhost:8000

---

## Option 2: Traditional pip Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/enterprise-agent-system.git
cd enterprise-agent-system/python-oop-masterclass/enterprise-agent-system
```

### Step 2: Create Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Step 5: Start Infrastructure
```bash
docker-compose up -d
```

### Step 6: Run Application
```bash
uvicorn src.api.main:app --reload
```

---

## Option 3: Minimal Setup (Development Only)

If you just want to explore the code without running infrastructure:

```bash
# 1. Clone and navigate
git clone <repo-url>
cd enterprise-agent-system/python-oop-masterclass/enterprise-agent-system

# 2. Install with UV
uv pip install -e "."

# 3. Run code quality checks
mypy src/
pytest tests/
```

---

## 🎯 Verify Installation

### Check API is Running
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "overall_status": "healthy",
  "components": {
    "api": {"status": "healthy"},
    ...
  }
}
```

### Run First Request
```bash
curl -X POST http://localhost:8000/requests \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust-12345",
    "message": "How do I reset my password?",
    "category": "account",
    "priority": 2
  }'
```

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'fastapi'"
**Solution**: Install dependencies
```bash
uv pip install -e ".[dev]"
# or
pip install -r requirements.txt
```

### Issue: "Connection refused" when starting API
**Solution**: Check if infrastructure services are running
```bash
docker-compose ps
# Ensure redis, postgres, weaviate, kafka are "Up"
```

### Issue: "Configuration error: OPENAI_API_KEY missing"
**Solution**: Add API key to .env file
```bash
echo 'OPENAI_API_KEY="sk-your-key-here"' >> .env
```

### Issue: Port 8000 already in use
**Solution**: Use a different port
```bash
uvicorn src.api.main:app --port 8001
```

---

## 📚 Next Steps

Now that you're up and running:

1. **Explore API**: http://localhost:8000/docs
2. **Read API Guide**: See [API_GUIDE.md](./API_GUIDE.md)
3. **Run Tests**: `pytest tests/`
4. **Check Code Quality**: `mypy src/`
5. **View Metrics**: http://localhost:8000/metrics

---

## 🚀 Production Deployment

For production deployment, see:
- [DEPLOYMENT.md](./DEPLOYMENT.md) - Deployment guide
- [CONFIGURATION.md](./CONFIGURATION.md) - Configuration reference

---

## 💡 Quick Tips

### Development Mode
```bash
# Auto-reload on code changes
uvicorn src.api.main:app --reload

# Enable debug logging
export LOG_LEVEL=DEBUG
```

### Type Checking
```bash
# Check all type hints
mypy src/

# Auto-format code
black src/
isort src/
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

---

## 🆘 Need Help?

- **Issues**: [GitHub Issues](https://github.com/your-org/enterprise-agent-system/issues)
- **Documentation**: See full [README.md](./README.md)
- **API Reference**: http://localhost:8000/docs

---

**Congratulations! 🎉** You're now ready to use the Enterprise Agent System!
