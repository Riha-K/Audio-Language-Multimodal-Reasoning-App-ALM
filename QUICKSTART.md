# ALM Hackathon Project - Quick Start Guide

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Or use Python runner
python run.py setup
```

### 2. Generate Datasets
```bash
python run.py datasets
```

### 3. Train Model
```bash
python run.py train
```

### 4. Run Demo
```bash
python run.py demo --share
```

### 5. Run Inference API
```bash
python run.py inference --gradio
```

## 📋 Available Commands

- `python run.py setup` - Setup development environment
- `python run.py datasets` - Generate training datasets
- `python run.py train` - Train the ALM model
- `python run.py inference` - Run inference API
- `python run.py demo` - Launch interactive demo
- `python run.py evaluate` - Evaluate model performance
- `python run.py pipeline` - Run complete pipeline
- `python run.py status` - Show project status

## 🎯 Hackathon Goals

### Primary Objectives
1. **Multi-language Support**: Support for 6+ Asian languages
2. **Audio Understanding**: Speech + non-speech audio processing
3. **Complex Reasoning**: Advanced reasoning capabilities
4. **Real-time Processing**: Fast inference and response

### Performance Targets
- Speech Recognition: >95% accuracy
- Audio Event Detection: >90% F1-score
- Complex Reasoning: >85% accuracy
- Multi-language Support: Native support for Asian languages

## 🏗️ Architecture Overview

```
ALM Model Architecture:
├── Audio Q-Former (Custom audio feature extraction)
├── Multi-layer Aggregator (Feature aggregation)
├── LLaMA-2-7B (Language model)
├── LoRA Fine-tuning (Efficient adaptation)
└── Soft Prompt (Enhanced reasoning)
```

## 📊 Evaluation Metrics

### Tasks Evaluated
1. **Speech Recognition**: Transcribe speech in multiple languages
2. **Audio Event Detection**: Identify sounds and audio events
3. **Complex Reasoning**: Understand context and make inferences
4. **Multilingual Understanding**: Process multiple languages simultaneously

### Metrics Used
- Accuracy
- Precision
- Recall
- F1-Score
- Language-specific performance
- Error analysis

## 🌏 Supported Languages

- **Mandarin Chinese** (中文)
- **Urdu** (اردو)
- **Hindi** (हिन्दी)
- **Telugu** (తెలుగు)
- **Tamil** (தமிழ்)
- **Bangla** (বাংলা)
- **English**

## 🎮 Demo Features

### Interactive Demo
- Real-time audio upload and analysis
- Multi-language support
- Visual waveform and spectrogram display
- Sample scenarios and instructions
- Benchmark results visualization

### API Endpoints
- `/analyze_audio` - Analyze uploaded audio
- `/analyze_text` - Text-only analysis
- `/health` - Health check
- `/model_info` - Model information

## 📁 Project Structure

```
ALM_Hackathon/
├── src/
│   ├── models/           # Model implementations
│   ├── data/            # Dataset generation
│   ├── training/        # Training scripts
│   ├── inference/       # Inference API
│   └── evaluation/      # Evaluation metrics
├── datasets/            # Generated datasets
├── checkpoints/         # Model checkpoints
├── demo/               # Interactive demo
├── configs/            # Configuration files
├── setup/              # Setup scripts
├── requirements.txt    # Dependencies
├── run.py             # Main runner script
└── README.md          # This file
```

## 🔧 Configuration

### Training Configuration
- Model: LLaMA-2-7B with LoRA
- Batch Size: 4
- Learning Rate: 3e-4
- Epochs: 3
- Max Length: 512

### Audio Configuration
- Sample Rate: 16kHz
- Max Duration: 30 seconds
- Mel Spectrogram: 64 mel bins
- Audio Q-Former: 32 queries

## 🏆 Hackathon Success Criteria

### Technical Excellence
- [ ] Multi-language audio understanding
- [ ] Complex reasoning capabilities
- [ ] Real-time processing
- [ ] Comprehensive evaluation

### Innovation
- [ ] Novel architecture improvements
- [ ] Asian language optimization
- [ ] Advanced reasoning features
- [ ] User-friendly interface

### Impact
- [ ] Practical applications
- [ ] Scalable solution
- [ ] Open-source contribution
- [ ] Community engagement

## 🚀 Deployment

### Local Development
```bash
# Start development server
python run.py demo --port 7860

# Start API server
python run.py inference --port 8000
```

### Production Deployment
```bash
# Build Docker image
docker build -t alm-hackathon .

# Run container
docker run -p 8000:8000 alm-hackathon
```

## 📞 Support

For questions or issues:
- Check the logs in `logs/` directory
- Review the evaluation results in `evaluation_results/`
- Use `python run.py status` to check project status

## 🎉 Good Luck!

This is your complete ALM hackathon project. You have everything you need to build a winning audio language model!
