#!/bin/bash
# RIV Protocol Setup Script
# Automated installation for IEEE S&P artifact evaluation

set -e  # Exit on error

echo "======================================"
echo "RIV Protocol Artifact Setup"
echo "======================================"
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "Error: Python 3.8 or higher required"
    exit 1
fi

# Check Rust installation
echo ""
echo "[2/6] Checking Rust toolchain..."
if ! command -v cargo &> /dev/null; then
    echo "Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    rust_version=$(rustc --version)
    echo "Found: $rust_version"
fi

# Create virtual environment
echo ""
echo "[3/6] Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "[4/6] Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Build cryptographic backend
echo ""
echo "[5/6] Building cryptographic backend..."
cd crypto_backend
bash build.sh
cd ..

# Download datasets
echo ""
echo "[6/6] Downloading datasets..."
cd data
bash download_datasets.sh
cd ..

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run a quick demo:"
echo "  python experiments/lenet5_mnist_demo.py"
echo ""
echo "To run full experiments:"
echo "  python experiments/run_all_experiments.py"
echo ""

echo ""
echo "========================================"
echo "Step 4: Building ZK libraries"
echo "========================================"

echo "Building matrix prover (Rust → Python)..."
cd matrix-prover-python
if ! maturin develop --release 2>&1 | tee /tmp/matrix_build.log | grep -E "(Finished|error)"; then
    echo "❌ Failed to build matrix-prover-python"
    echo "Check /tmp/matrix_build.log for details"
    exit 1
fi
cd ..

echo "Building range prover (Rust → Python)..."
cd range-prover-python
if ! maturin develop --release 2>&1 | tee /tmp/range_build.log | grep -E "(Finished|error)"; then
    echo "❌ Failed to build range-prover-python"
    echo "Check /tmp/range_build.log for details"
    exit 1
fi
cd ..

echo "✅ ZK libraries built successfully"

