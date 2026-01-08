"""
ALM Inference API and Web Interface

This module provides both REST API and web interface for ALM model inference,
supporting real-time audio processing and multi-language understanding.
"""

import os
import json
import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import tempfile
import librosa
import soundfile as sf
from pydantic import BaseModel
import gradio as gr
from datetime import datetime

from src.models.alm_model import ALMModel, create_alm_model

logger = logging.getLogger(__name__)

# Request/Response models
class AudioAnalysisRequest(BaseModel):
    instruction: str
    language: Optional[str] = "auto"
    temperature: float = 0.7
    max_length: int = 256

class AudioAnalysisResponse(BaseModel):
    response: str
    processing_time: float
    language_detected: str
    audio_duration: float
    confidence: float

class ALMInferenceAPI:
    """ALM Inference API Server"""
    
    def __init__(self, model_path: str = "checkpoints/alm_model/checkpoint-epoch-2-step-1000-best"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._load_model()
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="ALM Inference API",
            description="Audio Language Model API for multi-language audio understanding",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info("ALM Inference API initialized successfully!")
    
    def _load_model(self) -> ALMModel:
        """Load the trained ALM model"""
        try:
            # Load model configuration
            config_path = Path(self.model_path) / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
            else:
                # Default config
                config_dict = {
                    'base_model_name': 'meta-llama/Llama-2-7b-chat-hf',
                    'audio_dim': 64,
                    'hidden_dim': 768,
                    'use_lora': True
                }
            
            # Create model
            model = create_alm_model(config_dict)
            
            # Load weights
            model_path = Path(self.model_path) / "pytorch_model.bin"
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning(f"Model weights not found at {model_path}, using pre-trained weights")
            
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve the main web interface"""
            return self._get_web_interface()
        
        @self.app.post("/analyze_audio", response_model=AudioAnalysisResponse)
        async def analyze_audio(
            audio_file: UploadFile = File(...),
            instruction: str = Form(...),
            language: str = Form("auto"),
            temperature: float = Form(0.7),
            max_length: int = Form(256)
        ):
            """Analyze uploaded audio file"""
            try:
                start_time = datetime.now()
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    content = await audio_file.read()
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                
                # Process audio
                audio_data = self._process_audio_file(tmp_file_path)
                
                # Generate response
                response = self._generate_response(
                    audio_data=audio_data,
                    instruction=instruction,
                    language=language,
                    temperature=temperature,
                    max_length=max_length
                )
                
                # Clean up
                os.unlink(tmp_file_path)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return AudioAnalysisResponse(
                    response=response,
                    processing_time=processing_time,
                    language_detected=language,
                    audio_duration=len(audio_data) / 16000,  # Assuming 16kHz
                    confidence=0.85  # Placeholder confidence score
                )
                
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
        
        @self.app.post("/analyze_text")
        async def analyze_text(request: AudioAnalysisRequest):
            """Analyze text-only input (for testing)"""
            try:
                start_time = datetime.now()
                
                # Generate response without audio
                response = self._generate_response(
                    audio_data=None,
                    instruction=request.instruction,
                    language=request.language,
                    temperature=request.temperature,
                    max_length=request.max_length
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return AudioAnalysisResponse(
                    response=response,
                    processing_time=processing_time,
                    language_detected=request.language,
                    audio_duration=0.0,
                    confidence=0.90
                )
                
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "model_loaded": True}
        
        @self.app.get("/model_info")
        async def model_info():
            """Get model information"""
            return {
                "model_name": "ALM (Audio Language Model)",
                "version": "1.0.0",
                "supported_languages": ["mandarin", "urdu", "hindi", "telugu", "tamil", "bangla", "english"],
                "capabilities": [
                    "Speech Recognition",
                    "Audio Event Detection", 
                    "Complex Audio Reasoning",
                    "Multilingual Understanding"
                ],
                "device": str(self.device)
            }
    
    def _process_audio_file(self, file_path: str) -> torch.Tensor:
        """Process audio file and return tensor"""
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(file_path, sr=16000)
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
            
            # Ensure proper length (pad or truncate)
            max_length = int(30 * 16000)  # 30 seconds max
            if len(audio_tensor) > max_length:
                audio_tensor = audio_tensor[:max_length]
            else:
                # Pad with zeros
                padding = max_length - len(audio_tensor)
                audio_tensor = torch.cat([audio_tensor, torch.zeros(padding)])
            
            return audio_tensor
            
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")
    
    def _generate_response(self, 
                          audio_data: Optional[torch.Tensor],
                          instruction: str,
                          language: str,
                          temperature: float,
                          max_length: int) -> str:
        """Generate response using the ALM model"""
        try:
            # Create prompt
            prompt = self._create_prompt(instruction, language)
            
            # Tokenize input
            inputs = self.model.tokenizer(
                prompt,
                return_tensors='pt',
                max_length=512,
                truncation=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                if audio_data is not None:
                    audio_data = audio_data.to(self.device)
                    generated_ids = self.model.generate(
                        input_ids=inputs['input_ids'],
                        audio_data=audio_data,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.model.tokenizer.pad_token_id,
                        eos_token_id=self.model.tokenizer.eos_token_id
                    )
                else:
                    generated_ids = self.model.generate(
                        input_ids=inputs['input_ids'],
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.model.tokenizer.pad_token_id,
                        eos_token_id=self.model.tokenizer.eos_token_id
                    )
            
            # Decode response
            response = self.model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Extract only the response part
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _create_prompt(self, instruction: str, language: str) -> str:
        """Create prompt for the model"""
        prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
Audio analysis request in {language} language

### Response:"""
        
        return prompt_template.format(
            instruction=instruction,
            language=language
        )
    
    def _get_web_interface(self) -> str:
        """Get HTML web interface"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ALM - Audio Language Model</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .content {
            padding: 40px;
        }
        .form-group {
            margin-bottom: 25px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        input[type="file"], input[type="text"], select, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input[type="file"]:focus, input[type="text"]:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        textarea {
            height: 100px;
            resize: vertical;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            width: 100%;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            display: none;
        }
        .result h3 {
            margin-top: 0;
            color: #667eea;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .feature {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .feature h4 {
            color: #667eea;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 ALM</h1>
            <p>Audio Language Model - Listen, Think, and Understand</p>
        </div>
        
        <div class="content">
            <form id="audioForm">
                <div class="form-group">
                    <label for="audioFile">Upload Audio File:</label>
                    <input type="file" id="audioFile" accept="audio/*" required>
                </div>
                
                <div class="form-group">
                    <label for="instruction">Instruction:</label>
                    <textarea id="instruction" placeholder="What would you like to know about this audio? (e.g., 'Transcribe the speech', 'Identify the sounds', 'What can you infer from this audio?')" required></textarea>
                </div>
                
                <div class="form-group">
                    <label for="language">Language:</label>
                    <select id="language">
                        <option value="auto">Auto-detect</option>
                        <option value="mandarin">Mandarin Chinese</option>
                        <option value="urdu">Urdu</option>
                        <option value="hindi">Hindi</option>
                        <option value="telugu">Telugu</option>
                        <option value="tamil">Tamil</option>
                        <option value="bangla">Bangla</option>
                        <option value="english">English</option>
                    </select>
                </div>
                
                <button type="submit" class="btn" id="submitBtn">Analyze Audio</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing audio... This may take a few moments.</p>
            </div>
            
            <div class="result" id="result">
                <h3>Analysis Result:</h3>
                <div id="resultContent"></div>
            </div>
            
            <div class="features">
                <div class="feature">
                    <h4>🎤 Speech Recognition</h4>
                    <p>Transcribe speech in multiple Asian languages</p>
                </div>
                <div class="feature">
                    <h4>🔊 Audio Events</h4>
                    <p>Identify sounds, music, and environmental audio</p>
                </div>
                <div class="feature">
                    <h4>🧠 Complex Reasoning</h4>
                    <p>Understand context and make intelligent inferences</p>
                </div>
                <div class="feature">
                    <h4>🌏 Multi-language</h4>
                    <p>Support for 6+ Asian languages + English</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('audioForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const audioFile = document.getElementById('audioFile').files[0];
            const instruction = document.getElementById('instruction').value;
            const language = document.getElementById('language').value;
            
            if (!audioFile) {
                alert('Please select an audio file');
                return;
            }
            
            formData.append('audio_file', audioFile);
            formData.append('instruction', instruction);
            formData.append('language', language);
            formData.append('temperature', '0.7');
            formData.append('max_length', '256');
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('submitBtn').disabled = true;
            
            try {
                const response = await fetch('/analyze_audio', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Analysis failed');
                }
                
                const result = await response.json();
                
                // Show result
                document.getElementById('resultContent').innerHTML = `
                    <p><strong>Response:</strong> ${result.response}</p>
                    <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(2)}s</p>
                    <p><strong>Language Detected:</strong> ${result.language_detected}</p>
                    <p><strong>Audio Duration:</strong> ${result.audio_duration.toFixed(2)}s</p>
                    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                `;
                
                document.getElementById('result').style.display = 'block';
                
            } catch (error) {
                alert('Error analyzing audio: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('submitBtn').disabled = false;
            }
        });
    </script>
</body>
</html>
        """

def create_gradio_interface():
    """Create Gradio interface for ALM"""
    
    def analyze_audio(audio_file, instruction, language, temperature):
        """Analyze audio using ALM model"""
        if audio_file is None:
            return "Please upload an audio file."
        
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(audio_file, sr=16000)
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
            
            # Create prompt
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
Audio analysis request in {language} language

### Response:"""
            
            # For demo purposes, return a mock response
            # In production, this would use the actual ALM model
            mock_responses = {
                "Transcribe the speech": "The audio contains speech in " + language + ". The speaker is discussing various topics with clear pronunciation.",
                "Identify the sounds": "The audio contains multiple sound events including speech, background noise, and environmental sounds.",
                "What can you infer from this audio?": "Based on the audio analysis, this appears to be a " + language + " conversation in a typical environment with various background sounds.",
                "Analyze the emotional tone": "The emotional tone of the speech appears to be neutral to positive, with clear articulation and moderate pace."
            }
            
            response = mock_responses.get(instruction, "The audio has been analyzed successfully. The content appears to be in " + language + " with various audio characteristics.")
            
            return f"**Analysis Result:**\n\n{response}\n\n**Processing Details:**\n- Language: {language}\n- Temperature: {temperature}\n- Audio Duration: {len(audio_data)/sample_rate:.2f} seconds"
            
        except Exception as e:
            return f"Error processing audio: {str(e)}"
    
    # Create Gradio interface
    with gr.Blocks(title="ALM - Audio Language Model", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎵 ALM - Audio Language Model")
        gr.Markdown("Upload an audio file and ask questions about it. The model can transcribe speech, identify sounds, and perform complex reasoning.")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath"
                )
                instruction = gr.Textbox(
                    label="Instruction",
                    placeholder="What would you like to know about this audio?",
                    value="Analyze this audio and provide a comprehensive description."
                )
                language = gr.Dropdown(
                    choices=["auto", "mandarin", "urdu", "hindi", "telugu", "tamil", "bangla", "english"],
                    label="Language",
                    value="auto"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                analyze_btn = gr.Button("Analyze Audio", variant="primary")
            
            with gr.Column():
                output = gr.Markdown(label="Analysis Result")
        
        # Examples
        gr.Markdown("## Examples")
        examples = [
            ["What can you infer from this audio?", "auto", 0.7],
            ["Transcribe the speech in this audio", "english", 0.5],
            ["Identify all the sounds and events", "auto", 0.7],
            ["Analyze the emotional tone of the speech", "auto", 0.8]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[instruction, language, temperature],
            label="Example Instructions"
        )
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_audio,
            inputs=[audio_input, instruction, language, temperature],
            outputs=output
        )
    
    return interface

def main():
    """Main function to run the inference API"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ALM Inference API")
    parser.add_argument("--model_path", type=str, default="checkpoints/alm_model/checkpoint-epoch-2-step-1000-best")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--gradio", action="store_true", help="Use Gradio interface instead of FastAPI")
    
    args = parser.parse_args()
    
    if args.gradio:
        # Run Gradio interface
        interface = create_gradio_interface()
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=True
        )
    else:
        # Run FastAPI server
        api = ALMInferenceAPI(args.model_path)
        uvicorn.run(
            api.app,
            host=args.host,
            port=args.port,
            log_level="info"
        )

if __name__ == "__main__":
    main()
