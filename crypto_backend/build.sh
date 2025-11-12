#!/bin/bash
# Build script for cryptographic backend

set -e

echo "Building cryptographic backend..."

# Check Rust installation
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust toolchain not found"
    echo "Install via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Build Rust library
echo "Compiling Rust library..."
cargo build --release --features python

# Check if maturin is installed
if ! pip show maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Build and install Python bindings
echo "Building Python bindings..."
maturin develop --release --features python

echo ""
echo "Build complete!"
echo ""
echo "To verify installation:"
echo "  python -c 'import crypto_backend_rust; print(\"Success!\")'"
