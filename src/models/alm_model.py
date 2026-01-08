"""
ALM (Audio Language Model) Architecture Implementation

This module implements the core ALM model architecture based on GAMA,
with enhancements for Asian language support and multi-modal audio understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    LlamaConfig,
    BertTokenizerFast
)
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import torchaudio
import numpy as np
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class AudioQFormer(nn.Module):
    """Custom Audio Q-Former for audio feature extraction"""
    
    def __init__(self, 
                 audio_dim: int = 64,
                 hidden_dim: int = 768,
                 num_queries: int = 32,
                 num_layers: int = 6):
        super().__init__()
        
        self.num_queries = num_queries
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        
        # Audio feature projection
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        
        # Query embeddings
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, hidden_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, seq_len, audio_dim)
        Returns:
            query_features: (batch_size, num_queries, hidden_dim)
        """
        batch_size = audio_features.size(0)
        
        # Project audio features
        audio_proj = self.audio_projection(audio_features)  # (batch_size, seq_len, hidden_dim)
        
        # Expand query embeddings
        query_embeds = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate queries and audio features
        combined = torch.cat([query_embeds, audio_proj], dim=1)  # (batch_size, num_queries + seq_len, hidden_dim)
        
        # Apply transformer
        output = self.transformer(combined)
        
        # Extract query features
        query_features = output[:, :self.num_queries, :]  # (batch_size, num_queries, hidden_dim)
        
        # Apply layer normalization
        query_features = self.layer_norm(query_features)
        
        return query_features

class MultiLayerAggregator(nn.Module):
    """Multi-layer aggregator for audio encoder features"""
    
    def __init__(self, 
                 input_dim: int = 768,
                 hidden_dim: int = 768,
                 num_layers: int = 3):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Layer-wise projections
        self.layer_projections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Attention mechanism for layer fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Final projection
        self.final_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            layer_features: List of (batch_size, seq_len, input_dim) tensors
        Returns:
            aggregated_features: (batch_size, seq_len, hidden_dim)
        """
        batch_size = layer_features[0].size(0)
        seq_len = layer_features[0].size(1)
        
        # Project each layer
        projected_features = []
        for i, features in enumerate(layer_features):
            projected = self.layer_projections[i](features)
            projected_features.append(projected)
        
        # Stack features
        stacked_features = torch.stack(projected_features, dim=1)  # (batch_size, num_layers, seq_len, hidden_dim)
        
        # Reshape for attention
        stacked_features = stacked_features.view(batch_size, self.num_layers, seq_len * stacked_features.size(-1))
        
        # Apply attention
        attended_features, _ = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Reshape back
        attended_features = attended_features.view(batch_size, seq_len, -1)
        
        # Final projection
        aggregated_features = self.final_projection(attended_features)
        
        return aggregated_features

class ALMModel(nn.Module):
    """Main ALM Model Architecture"""
    
    def __init__(self,
                 base_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 audio_dim: int = 64,
                 hidden_dim: int = 768,
                 num_audio_queries: int = 32,
                 num_audio_layers: int = 6,
                 num_aggregator_layers: int = 3,
                 use_lora: bool = True,
                 lora_config: Optional[Dict] = None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_lora = use_lora
        
        # Load base language model
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = LlamaForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Audio processing components
        self.audio_qformer = AudioQFormer(
            audio_dim=audio_dim,
            hidden_dim=hidden_dim,
            num_queries=num_audio_queries,
            num_layers=num_audio_layers
        )
        
        self.audio_aggregator = MultiLayerAggregator(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_aggregator_layers
        )
        
        # Audio-to-text projection
        self.audio_to_text_projection = nn.Linear(hidden_dim, self.llm.config.hidden_size)
        
        # Soft prompt for enhanced reasoning
        self.soft_prompt = nn.Parameter(torch.randn(10, self.llm.config.hidden_size))
        
        # Apply LoRA if specified
        if use_lora:
            if lora_config is None:
                lora_config = {
                    "r": 8,
                    "lora_alpha": 16,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM"
                }
            
            lora_config_obj = LoraConfig(**lora_config)
            self.llm = get_peft_model(self.llm, lora_config_obj)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize audio components
        for module in [self.audio_qformer, self.audio_aggregator, self.audio_to_text_projection]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # Initialize soft prompt
        nn.init.normal_(self.soft_prompt, std=0.02)
    
    def get_audio_features(self, audio_data: torch.Tensor) -> torch.Tensor:
        """Extract audio features using mel spectrogram"""
        # Convert to mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=480,
            n_mels=64,
            f_min=50,
            f_max=8000
        ).to(audio_data.device)
        
        mel = mel_transform(audio_data)
        mel = torchaudio.transforms.AmplitudeToDB()(mel)
        
        # Transpose to (batch_size, seq_len, n_mels)
        mel = mel.transpose(1, 2)
        
        return mel
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                audio_data: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the ALM model
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            audio_data: (batch_size, audio_seq_len) - optional
            labels: (batch_size, seq_len) - optional
            
        Returns:
            Dictionary containing loss and logits
        """
        batch_size = input_ids.size(0)
        
        # Process audio if provided
        audio_features = None
        if audio_data is not None:
            # Extract audio features
            mel_features = self.get_audio_features(audio_data)  # (batch_size, seq_len, 64)
            
            # Apply audio Q-Former
            audio_query_features = self.audio_qformer(mel_features)  # (batch_size, num_queries, hidden_dim)
            
            # Apply multi-layer aggregator
            audio_features = self.audio_aggregator([audio_query_features])  # (batch_size, seq_len, hidden_dim)
            
            # Project to LLM hidden size
            audio_features = self.audio_to_text_projection(audio_features)  # (batch_size, seq_len, llm_hidden_size)
        
        # Prepare inputs for LLM
        if audio_features is not None:
            # Create extended input embeddings
            text_embeddings = self.llm.get_input_embeddings()(input_ids)
            
            # Add soft prompt
            soft_prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Concatenate soft prompt, audio features, and text embeddings
            extended_embeddings = torch.cat([
                soft_prompt_embeds,
                audio_features,
                text_embeddings
            ], dim=1)
            
            # Update attention mask
            soft_prompt_len = self.soft_prompt.size(0)
            audio_len = audio_features.size(1) if audio_features is not None else 0
            
            extended_attention_mask = torch.cat([
                torch.ones(batch_size, soft_prompt_len + audio_len, device=input_ids.device),
                attention_mask
            ], dim=1)
            
            # Create custom inputs
            inputs_embeds = extended_embeddings
            attention_mask = extended_attention_mask
            
            # Forward pass through LLM
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        else:
            # Standard text-only forward pass
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states
        }
    
    def generate(self,
                 input_ids: torch.Tensor,
                 audio_data: Optional[torch.Tensor] = None,
                 max_length: int = 256,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 do_sample: bool = True) -> torch.Tensor:
        """Generate text with optional audio conditioning"""
        
        with torch.no_grad():
            if audio_data is not None:
                # Process audio
                mel_features = self.get_audio_features(audio_data)
                audio_query_features = self.audio_qformer(mel_features)
                audio_features = self.audio_aggregator([audio_query_features])
                audio_features = self.audio_to_text_projection(audio_features)
                
                # Create extended embeddings
                batch_size = input_ids.size(0)
                text_embeddings = self.llm.get_input_embeddings()(input_ids)
                soft_prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
                
                extended_embeddings = torch.cat([
                    soft_prompt_embeds,
                    audio_features,
                    text_embeddings
                ], dim=1)
                
                # Generate with extended inputs
                generated_ids = self.llm.generate(
                    inputs_embeds=extended_embeddings,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                # Standard text generation
                generated_ids = self.llm.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
        return generated_ids
    
    def save_model(self, save_path: str):
        """Save the model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'use_lora': self.use_lora
            }
        }, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load the model"""
        checkpoint = torch.load(load_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {load_path}")

class ALMTrainer:
    """Trainer class for ALM model"""
    
    def __init__(self, model: ALMModel, learning_rate: float = 3e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            audio_data=batch.get('audio_data'),
            labels=batch.get('labels')
        )
        
        loss = outputs['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    audio_data=batch.get('audio_data'),
                    labels=batch.get('labels')
                )
                
                total_loss += outputs['loss'].item()
                num_batches += 1
        
        return {
            'eval_loss': total_loss / num_batches
        }

def create_alm_model(config: Dict[str, Any]) -> ALMModel:
    """Factory function to create ALM model"""
    return ALMModel(
        base_model_name=config.get('base_model_name', 'meta-llama/Llama-2-7b-chat-hf'),
        audio_dim=config.get('audio_dim', 64),
        hidden_dim=config.get('hidden_dim', 768),
        num_audio_queries=config.get('num_audio_queries', 32),
        num_audio_layers=config.get('num_audio_layers', 6),
        num_aggregator_layers=config.get('num_aggregator_layers', 3),
        use_lora=config.get('use_lora', True),
        lora_config=config.get('lora_config')
    )

if __name__ == "__main__":
    # Test the model
    config = {
        'base_model_name': 'meta-llama/Llama-2-7b-chat-hf',
        'audio_dim': 64,
        'hidden_dim': 768,
        'use_lora': True
    }
    
    model = create_alm_model(config)
    print(f"ALM Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
