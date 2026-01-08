@echo off
REM ALM Hackathon Setup Script for Windows
REM This script sets up the complete ALM environment for the hackathon

echo 🚀 Setting up ALM Hackathon Environment...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.10+ first.
    pause
    exit /b 1
)

echo ✅ Python found

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv alm_env
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call alm_env\Scripts\activate.bat

REM Install requirements
echo 📋 Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

REM Create directories
echo 📁 Creating project directories...
if not exist "datasets\asian_audio" mkdir datasets\asian_audio
if not exist "checkpoints\alm_model" mkdir checkpoints\alm_model
if not exist "evaluation_results" mkdir evaluation_results
if not exist "logs" mkdir logs
if not exist "models" mkdir models

REM Generate datasets
echo 🎵 Generating Asian audio datasets...
python src\data\generate_dataset.py
if errorlevel 1 (
    echo ⚠️ Dataset generation failed, but continuing...
)

REM Setup models
echo 🤖 Setting up pre-trained models...
python setup\download_models.py
if errorlevel 1 (
    echo ⚠️ Model setup failed, but continuing...
)

REM Run initial tests
echo 🧪 Running initial tests...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo ✅ Setup completed successfully!
echo.
echo 🎯 Next steps:
echo 1. Activate environment: alm_env\Scripts\activate.bat
echo 2. Start training: python run.py train
echo 3. Run inference: python run.py inference
echo 4. Launch demo: python run.py demo
echo.
echo 🏆 Good luck with your hackathon!
pause
