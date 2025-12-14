#!/bin/bash
# Quick install script for EDA dependencies

echo "Installing EDA dependencies..."

# Try user installation first
echo "Attempting user installation..."
pip install --user matplotlib seaborn scikit-learn pandas numpy 2>&1 | grep -E "(Successfully|already|error)" || true

# Check if installation was successful
python3 -c "import matplotlib; import seaborn; import sklearn" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
    echo "Run: python scripts/05_cleaning_and_eda.py"
else
    echo "⚠️  Installation may have failed or packages not in path"
    echo ""
    echo "Alternative options:"
    echo "1. Create virtual environment:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo ""
    echo "2. Use system packages (if available):"
    echo "   sudo apt install python3-matplotlib python3-seaborn python3-sklearn"
    echo ""
    echo "3. Force install (use with caution):"
    echo "   pip install --break-system-packages matplotlib seaborn scikit-learn"
fi

