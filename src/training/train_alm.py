"""
ALM Training Pipeline

This module implements the complete training pipeline for the ALM model,
including data loading, preprocessing, training loops, and checkpointing.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
import wandb
from pathlib import Path
import yaml
import random
from dataclasses import dataclass

from src.models.alm_model import ALMModel, ALMTrainer
from src.data.generate_dataset import AsianAudioDatasetGenerator

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model config
    base_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    audio_dim: int = 64
    hidden_dim: int = 768
    num_audio_queries: int = 32
    num_audio_layers: int = 6
    num_aggregator_layers: int = 3
    use_lora: bool = True
    
    # Training config
    batch_size: int = 4
    micro_batch_size: int = 1
    num_epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    
    # Data config
    train_data_path: str = "datasets/asian_audio/train_asian_audio_dataset.json"
    val_data_path: str = "datasets/asian_audio/val_asian_audio_dataset.json"
    test_data_path: str = "datasets/asian_audio/test_asian_audio_dataset.json"
    
    # Output config
    output_dir: str = "checkpoints/alm_model"
    save_steps: int = 100
    eval_steps: int = 200
    logging_steps: int = 10
    
    # Wandb config
    use_wandb: bool = True
    wandb_project: str = "alm-hackathon"
    wandb_run_name: str = "alm-training"
    
    # Audio config
    sample_rate: int = 16000
    max_audio_length: float = 30.0  # seconds
    
    # LoRA config
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

class ALMDataset(Dataset):
    """Dataset class for ALM training"""
    
    def __init__(self, 
                 data_path: str,
                 tokenizer,
                 max_length: int = 512,
                 sample_rate: int = 16000,
                 max_audio_length: float = 30.0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        
        # Load dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Create prompt
        prompt = self._create_prompt(sample)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Generate synthetic audio data for training
        audio_data = self._generate_synthetic_audio(sample['duration'])
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'audio_data': audio_data,
            'labels': inputs['input_ids'].squeeze(0),  # For language modeling
            'sample_info': sample
        }
    
    def _create_prompt(self, sample: Dict[str, Any]) -> str:
        """Create training prompt from sample"""
        prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
Audio recording in {language} language, duration: {duration:.1f}s

### Response:
{output}"""
        
        return prompt_template.format(
            instruction=sample['instruction'],
            language=sample['language'],
            duration=sample['duration'],
            output=sample['output']
        )
    
    def _generate_synthetic_audio(self, duration: float) -> torch.Tensor:
        """Generate synthetic audio data for training"""
        # Generate random audio data
        audio_length = int(duration * self.sample_rate)
        audio_data = torch.randn(audio_length) * 0.1
        
        # Add some structure to make it more realistic
        t = torch.linspace(0, duration, audio_length)
        audio_data += 0.1 * torch.sin(2 * torch.pi * 440 * t)  # A4 note
        audio_data += 0.05 * torch.sin(2 * torch.pi * 880 * t)  # A5 note
        
        # Add noise
        audio_data += torch.randn_like(audio_data) * 0.01
        
        return audio_data

class ALMTrainingPipeline:
    """Main training pipeline for ALM"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Setup wandb
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__
            )
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._create_model()
        self.trainer = ALMTrainer(self.model, config.learning_rate)
        
        # Initialize datasets
        self.train_dataset = self._create_dataset(config.train_data_path)
        self.val_dataset = self._create_dataset(config.val_data_path)
        
        # Initialize data loaders
        self.train_loader = self._create_dataloader(self.train_dataset, is_training=True)
        self.val_loader = self._create_dataloader(self.val_dataset, is_training=False)
        
        logger.info("Training pipeline initialized successfully!")
    
    def _create_model(self) -> ALMModel:
        """Create ALM model"""
        model_config = {
            'base_model_name': self.config.base_model_name,
            'audio_dim': self.config.audio_dim,
            'hidden_dim': self.config.hidden_dim,
            'num_audio_queries': self.config.num_audio_queries,
            'num_audio_layers': self.config.num_audio_layers,
            'num_aggregator_layers': self.config.num_aggregator_layers,
            'use_lora': self.config.use_lora,
            'lora_config': {
                'r': self.config.lora_r,
                'lora_alpha': self.config.lora_alpha,
                'target_modules': self.config.lora_target_modules,
                'lora_dropout': self.config.lora_dropout,
                'bias': 'none',
                'task_type': 'CAUSAL_LM'
            }
        }
        
        model = ALMModel(**model_config)
        model.to(self.device)
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
    
    def _create_dataset(self, data_path: str) -> ALMDataset:
        """Create dataset"""
        return ALMDataset(
            data_path=data_path,
            tokenizer=self.model.tokenizer,
            max_length=self.config.max_length,
            sample_rate=self.config.sample_rate,
            max_audio_length=self.config.max_audio_length
        )
    
    def _create_dataloader(self, dataset: ALMDataset, is_training: bool) -> DataLoader:
        """Create data loader"""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size if is_training else self.config.batch_size * 2,
            shuffle=is_training,
            num_workers=4,
            pin_memory=True,
            drop_last=is_training
        )
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training phase
            train_loss = self._train_epoch(global_step)
            
            # Evaluation phase
            eval_loss = self._evaluate_epoch()
            
            # Logging
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
            
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                    'learning_rate': self.trainer.scheduler.get_last_lr()[0]
                })
            
            # Save checkpoint if best
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self._save_checkpoint(epoch, global_step, is_best=True)
            
            # Regular checkpoint saving
            if (epoch + 1) % 2 == 0:  # Save every 2 epochs
                self._save_checkpoint(epoch, global_step, is_best=False)
        
        logger.info("Training completed!")
    
    def _train_epoch(self, global_step: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Training step
            metrics = self.trainer.train_step(batch)
            
            total_loss += metrics['loss']
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'lr': f"{metrics['learning_rate']:.2e}"
            })
            
            # Logging
            if global_step % self.config.logging_steps == 0:
                logger.info(f"Step {global_step}: Loss = {metrics['loss']:.4f}")
                
                if self.config.use_wandb:
                    wandb.log({
                        'train_loss': metrics['loss'],
                        'learning_rate': metrics['learning_rate'],
                        'global_step': global_step
                    })
            
            # Save checkpoint
            if global_step % self.config.save_steps == 0:
                self._save_checkpoint(0, global_step, is_best=False)
        
        return total_loss / num_batches
    
    def _evaluate_epoch(self) -> float:
        """Evaluate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Evaluating")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    audio_data=batch.get('audio_data'),
                    labels=batch.get('labels')
                )
                
                total_loss += outputs['loss'].item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f"{outputs['loss'].item():.4f}"})
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, epoch: int, global_step: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_name = f"checkpoint-epoch-{epoch}-step-{global_step}"
        if is_best:
            checkpoint_name += "-best"
        
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / "pytorch_model.bin"
        torch.save(self.model.state_dict(), model_path)
        
        # Save config
        config_path = checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save tokenizer
        self.model.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def generate_sample_outputs(self, num_samples: int = 5):
        """Generate sample outputs for evaluation"""
        self.model.eval()
        
        logger.info("Generating sample outputs...")
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_samples:
                    break
                
                # Get first sample from batch
                sample = {k: v[0:1] for k, v in batch.items()}
                sample = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in sample.items()}
                
                # Generate response
                generated_ids = self.model.generate(
                    input_ids=sample['input_ids'],
                    audio_data=sample.get('audio_data'),
                    max_length=self.config.max_length,
                    temperature=0.7,
                    do_sample=True
                )
                
                # Decode
                input_text = self.model.tokenizer.decode(sample['input_ids'][0], skip_special_tokens=True)
                generated_text = self.model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                logger.info(f"\n--- Sample {i + 1} ---")
                logger.info(f"Input: {input_text[:200]}...")
                logger.info(f"Generated: {generated_text[:200]}...")

def main():
    """Main training function"""
    # Load config
    config_path = "configs/training_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Generate datasets if they don't exist
    dataset_dir = Path("datasets/asian_audio")
    if not dataset_dir.exists():
        logger.info("Generating datasets...")
        generator = AsianAudioDatasetGenerator(str(dataset_dir))
        
        # Generate datasets
        train_dataset = generator.generate_dataset(num_samples=2000)
        val_dataset = generator.generate_dataset(num_samples=500)
        test_dataset = generator.generate_dataset(num_samples=300)
        
        generator.save_dataset(train_dataset, "train_asian_audio_dataset.json")
        generator.save_dataset(val_dataset, "val_asian_audio_dataset.json")
        generator.save_dataset(test_dataset, "test_asian_audio_dataset.json")
    
    # Initialize training pipeline
    pipeline = ALMTrainingPipeline(config)
    
    # Start training
    pipeline.train()
    
    # Generate sample outputs
    pipeline.generate_sample_outputs()

if __name__ == "__main__":
    main()
