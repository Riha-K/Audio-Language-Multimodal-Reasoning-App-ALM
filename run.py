#!/usr/bin/env python3
"""
ALM Hackathon - Main Runner Script

This script provides a unified interface to run all components of the ALM project.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ALMRunner:
    """Main runner class for ALM project"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            "datasets/asian_audio",
            "checkpoints/alm_model",
            "evaluation_results",
            "logs",
            "models"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def setup_environment(self):
        """Setup the development environment"""
        logger.info("🔧 Setting up ALM environment...")
        
        # Check if conda is available
        try:
            subprocess.run(["conda", "--version"], check=True, capture_output=True)
            logger.info("✅ Conda found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️ Conda not found. Please install Anaconda/Miniconda first.")
            return False
        
        # Create conda environment
        try:
            subprocess.run([
                "conda", "create", "-n", "alm", "python=3.10", "-y"
            ], check=True)
            logger.info("✅ Conda environment created")
        except subprocess.CalledProcessError:
            logger.info("ℹ️ Conda environment already exists")
        
        # Install requirements
        try:
            subprocess.run([
                "conda", "run", "-n", "alm", "pip", "install", "-r", "requirements.txt"
            ], check=True)
            logger.info("✅ Requirements installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install requirements: {e}")
            return False
        
        return True
    
    def generate_datasets(self):
        """Generate training datasets"""
        logger.info("🎵 Generating Asian audio datasets...")
        
        try:
            subprocess.run([
                "python", "src/data/generate_dataset.py"
            ], check=True)
            logger.info("✅ Datasets generated successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to generate datasets: {e}")
            return False
        
        return True
    
    def train_model(self, config_path: str = "configs/training_config.yaml"):
        """Train the ALM model"""
        logger.info("🏋️ Training ALM model...")
        
        try:
            subprocess.run([
                "python", "src/training/train_alm.py",
                "--config", config_path
            ], check=True)
            logger.info("✅ Model training completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Training failed: {e}")
            return False
        
        return True
    
    def run_inference_api(self, model_path: str = "checkpoints/alm_model/checkpoint-epoch-2-step-1000-best", 
                         port: int = 8000, gradio: bool = False):
        """Run the inference API"""
        logger.info("🚀 Starting inference API...")
        
        try:
            cmd = [
                "python", "src/inference/inference_api.py",
                "--model_path", model_path,
                "--port", str(port)
            ]
            
            if gradio:
                cmd.append("--gradio")
            
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Inference API failed: {e}")
            return False
        
        return True
    
    def run_demo(self, model_path: str = "checkpoints/alm_model/checkpoint-epoch-2-step-1000-best", 
                port: int = 7860, share: bool = False):
        """Run the demo application"""
        logger.info("🎮 Starting demo application...")
        
        try:
            cmd = [
                "python", "demo/app.py",
                "--model_path", model_path,
                "--port", str(port)
            ]
            
            if share:
                cmd.append("--share")
            
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Demo failed: {e}")
            return False
        
        return True
    
    def evaluate_model(self, model_path: str = "checkpoints/alm_model/checkpoint-epoch-2-step-1000-best",
                      test_data_path: str = "datasets/asian_audio/test_asian_audio_dataset.json"):
        """Evaluate the model"""
        logger.info("📊 Evaluating model...")
        
        try:
            subprocess.run([
                "python", "src/evaluation/evaluate_alm.py",
                "--model_path", model_path,
                "--test_data_path", test_data_path
            ], check=True)
            logger.info("✅ Evaluation completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return False
        
        return True
    
    def run_full_pipeline(self):
        """Run the complete ALM pipeline"""
        logger.info("🚀 Running complete ALM pipeline...")
        
        steps = [
            ("Setup Environment", self.setup_environment),
            ("Generate Datasets", self.generate_datasets),
            ("Train Model", self.train_model),
            ("Evaluate Model", self.evaluate_model),
        ]
        
        for step_name, step_func in steps:
            logger.info(f"📋 Step: {step_name}")
            if not step_func():
                logger.error(f"❌ Pipeline failed at step: {step_name}")
                return False
            logger.info(f"✅ {step_name} completed")
        
        logger.info("🎉 Complete pipeline finished successfully!")
        return True
    
    def show_status(self):
        """Show project status"""
        logger.info("📊 ALM Project Status:")
        
        # Check directories
        directories = [
            "datasets/asian_audio",
            "checkpoints/alm_model",
            "evaluation_results",
            "models"
        ]
        
        for directory in directories:
            path = Path(directory)
            if path.exists():
                files = list(path.glob("*"))
                logger.info(f"✅ {directory}: {len(files)} files")
            else:
                logger.info(f"❌ {directory}: Not found")
        
        # Check datasets
        dataset_files = [
            "datasets/asian_audio/train_asian_audio_dataset.json",
            "datasets/asian_audio/val_asian_audio_dataset.json",
            "datasets/asian_audio/test_asian_audio_dataset.json"
        ]
        
        logger.info("\n📁 Dataset Status:")
        for dataset_file in dataset_files:
            path = Path(dataset_file)
            if path.exists():
                size = path.stat().st_size / 1024 / 1024  # MB
                logger.info(f"✅ {dataset_file}: {size:.2f} MB")
            else:
                logger.info(f"❌ {dataset_file}: Not found")
        
        # Check model checkpoints
        checkpoint_dir = Path("checkpoints/alm_model")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
            logger.info(f"\n🤖 Model Checkpoints: {len(checkpoints)} found")
            for checkpoint in checkpoints:
                logger.info(f"  - {checkpoint.name}")
        else:
            logger.info("\n🤖 Model Checkpoints: None found")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ALM Hackathon Runner")
    parser.add_argument("command", choices=[
        "setup", "datasets", "train", "inference", "demo", "evaluate", 
        "pipeline", "status"
    ], help="Command to run")
    
    parser.add_argument("--model_path", type=str, 
                       default="checkpoints/alm_model/checkpoint-epoch-2-step-1000-best",
                       help="Path to model checkpoint")
    parser.add_argument("--port", type=int, default=8000, help="Port for API/Demo")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--gradio", action="store_true", help="Use Gradio interface")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Training config file")
    
    args = parser.parse_args()
    
    runner = ALMRunner()
    
    if args.command == "setup":
        runner.setup_environment()
    elif args.command == "datasets":
        runner.generate_datasets()
    elif args.command == "train":
        runner.train_model(args.config)
    elif args.command == "inference":
        runner.run_inference_api(args.model_path, args.port, args.gradio)
    elif args.command == "demo":
        runner.run_demo(args.model_path, args.port, args.share)
    elif args.command == "evaluate":
        runner.evaluate_model(args.model_path)
    elif args.command == "pipeline":
        runner.run_full_pipeline()
    elif args.command == "status":
        runner.show_status()
    else:
        logger.error(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()
