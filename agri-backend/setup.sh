#!/usr/bin/env bash
# =============================================================================
# setup.sh — AgriLoan Credit Scorer Backend — One-command setup
# =============================================================================
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# What this script does:
#   1. Creates a Python virtual environment (agri-env/)
#   2. Installs all dependencies from requirements.txt
#   3. Trains the ML model (saves to model/credit_model.pkl)
#   4. Initialises the SQLite database
#   5. Runs a quick health check
#   6. Starts the development server on port 5000

set -e   # exit immediately on any error

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'  # No colour

banner() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    echo -e "${NC}"
}

ok()  { echo -e "${GREEN}  ✅  $1${NC}"; }
err() { echo -e "${RED}  ❌  $1${NC}"; exit 1; }

banner "AgriLoan Credit Scorer — Backend Setup"

# ── 1. Check Python version ───────────────────────────────────────────────────
echo "Checking Python version …"
python3 --version || err "Python 3 not found. Please install Python 3.10+"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python $PYTHON_VERSION detected"

# ── 2. Create virtual environment ─────────────────────────────────────────────
if [ ! -d "agri-env" ]; then
    echo "Creating virtual environment (agri-env/) …"
    python3 -m venv agri-env
    ok "Virtual environment created"
else
    ok "Virtual environment already exists"
fi

# ── 3. Activate virtual environment ───────────────────────────────────────────
# shellcheck disable=SC1091
source agri-env/bin/activate
ok "Virtual environment activated"

# ── 4. Install dependencies ────────────────────────────────────────────────────
echo ""
echo "Installing dependencies from requirements.txt …"
pip install --upgrade pip -q
pip install -r requirements.txt -q
ok "All dependencies installed"

# ── 5. Train ML model ─────────────────────────────────────────────────────────
echo ""
banner "Training ML Model"
python3 train_model.py
ok "Model trained and saved to model/credit_model.pkl"

# ── 6. Initialise database ────────────────────────────────────────────────────
echo ""
echo "Initialising SQLite database …"
python3 -c "from database.init_db import init_db; init_db()"
ok "Database ready at database/database.db"

# ── 7. Quick smoke test ───────────────────────────────────────────────────────
echo ""
echo "Running smoke tests …"
python3 -c "
from app import app
client = app.test_client()
r = client.get('/health')
assert r.status_code == 200, f'Health check failed: {r.status_code}'
data = r.get_json()
assert data['status'] == 'running'
assert data['model_ready'] == True
print('  /health          OK')
r2 = client.post('/evaluate', json={
    'fullName':'Test Farmer','aadhaar':'1234-5678-9012',
    'phone':'9876543210','district':'nashik','village':'TestVillage',
    'surveyNumber':'1/1/A','landSize':5,'irrigation':'drip',
    'ownership':'owner','primaryCrop':'wheat','cropDiversity':2,
    'yield1':20,'yield2':22,'yield3':25,'latitude':19.99,'longitude':73.78
})
assert r2.status_code == 201
d = r2.get_json()
assert 'trust_score' in d
print(f'  /evaluate        OK  (score={d[\"trust_score\"]}, status={d[\"status\"]})')
r3 = client.get('/applications')
assert r3.status_code == 200
print(f'  /applications    OK  (total={r3.get_json()[\"total\"]})')
print('  All smoke tests passed!')
"
ok "Smoke tests passed"

# ── 8. Done ───────────────────────────────────────────────────────────────────
banner "Setup Complete!"
echo ""
echo "  Start the server:"
echo ""
echo "    Development  :  python app.py"
echo "    Production   :  gunicorn -w 4 -b 0.0.0.0:5000 'app:create_app()'"
echo ""
echo "  API will be available at:"
echo "    http://localhost:5000/health"
echo "    http://localhost:5000/evaluate     (POST)"
echo "    http://localhost:5000/applications (GET)"
echo "    http://localhost:5000/stats        (GET)"
echo ""

# ── 9. Start dev server (optional – comment out if you want manual start) ─────
echo "Starting development server …"
echo "(Press Ctrl+C to stop)"
echo ""
python3 app.py