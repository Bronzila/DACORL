#!/bin/bash
# Install DACORL environment from packed tarball (OFFLINE)

echo "=========================================="
echo "DACORL Environment Installation (Offline)"
echo "=========================================="

# Check if miniconda/anaconda is installed
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install miniconda first."
    exit 1
fi

echo "✓ Conda found: $(conda --version)"

# Create directory for the environment
ENV_DIR="$HOME/miniconda3/envs/DACORL"  # Adjust if needed
# Or use: ENV_DIR="$HOME/.conda/envs/DACORL"

echo ""
echo "Installing environment to: $ENV_DIR"
echo ""

# Create the directory if it doesn't exist
mkdir -p "$ENV_DIR"

# Extract the packed environment
echo "Extracting environment (this may take a few minutes)..."
tar -xzf DACORL_env.tar.gz -C "$ENV_DIR"

# Activate and fix paths
echo ""
echo "Fixing environment paths..."
cd "$ENV_DIR"
source "$ENV_DIR/bin/activate"
conda-unpack

echo ""
echo "=========================================="
echo "✓ Environment installed successfully!"
echo "=========================================="

# Test activation
echo ""
echo "Testing environment activation..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate DACORL

echo "Python: $(python --version)"
echo "Location: $(which python)"

# Install editable packages (DACBench)
echo ""
echo "Installing DACBench (editable package)..."
cd ../cloud_dacorl/DACBench
pip install -e .

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use the environment:"
echo "  conda activate DACORL"
echo ""
echo "Verify installation:"
echo "  python -c 'import torch; print(torch.__version__)'"
echo "  python -c 'from dacbench import AbstractEnv; print(\"DACBench OK\")'"
echo ""
