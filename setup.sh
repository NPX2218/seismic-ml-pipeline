#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Done! Run:"
echo "  source venv/bin/activate"
echo "  python main.py"
