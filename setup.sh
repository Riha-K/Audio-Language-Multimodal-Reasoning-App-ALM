#!/bin/bash

# ALM Hackathon Setup Script
# This script sets up the complete ALM environment for the hackathon

echo "🚀 Setting up ALM Hackathon Environment..."

# Create conda environment
echo "📦 Creating conda environment..."
conda create -n alm python=3.10 -y
conda activate alm

# Install PyTorch
echo "🔥 Installing PyTorch..."
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo "📋 Installing requirements..."
pip install -r requirements.txt

# Install additional packages
echo "🔧 Installing additional packages..."
pip install transformers accelerate datasets peft gradio fastapi uvicorn

# Create directories
echo "📁 Creating project directories..."
mkdir -p datasets/asian_audio
mkdir -p checkpoints/alm_model
mkdir -p evaluation_results
mkdir -p logs

# Generate datasets
echo "🎵 Generating Asian audio datasets..."
python src/data/generate_dataset.py

# Download pre-trained models (if available)
echo "🤖 Setting up pre-trained models..."
python setup/download_models.py

# Run initial tests
echo "🧪 Running initial tests..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate environment: conda activate alm"
echo "2. Start training: python src/training/train_alm.py"
echo "3. Run inference: python src/inference/inference_api.py"
echo "4. Launch demo: python demo/app.py"
echo ""
echo "🏆 Good luck with your hackathon!"
