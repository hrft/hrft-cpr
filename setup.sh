#!/bin/bash
# ============================
# Setup & Run Script for hrft-cpr Project
# ============================

# Exit on error
set -e

echo "🔹 Checking & Creating conda environment if needed..."
if conda env list | grep -q "hrft-cpr"; then
    echo "✅ Environment hrft-cpr found."
else
    echo "⚠️ Environment not found, creating..."
    conda create -n hrft-cpr python=3.10 -y
fi

echo "🔹 Installing required packages..."
conda run -n hrft-cpr pip install --upgrade pip
conda run -n hrft-cpr pip install streamlit yfinance numpy pandas scikit-learn tensorflow

echo "✅ All dependencies installed!"

echo "🔹 Running quick test..."
conda run -n hrft-cpr python - <<EOF
import tensorflow as tf, numpy as np, pandas as pd
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
EOF

echo "✅ Test successful!"

echo "🔹 Starting Streamlit app..."
conda run -n hrft-cpr streamlit run app.py
