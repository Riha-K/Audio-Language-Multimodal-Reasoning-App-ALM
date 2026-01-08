# ALM Architecture – Listen • Think • Understand

## High-Level Diagram
```mermaid
graph TD
  A[Raw Audio Waveform] --> B1[Speech Encoder\n(Whisper/Wav2Vec2/HuBERT)]
  A --> B2[Non‑Speech Encoder\n(BEATs/AudioMAE/PANNs)]
  B1 --> C[Joint Audio Embedding]
  B2 --> C
  C --> D[Cross‑Modal Fusion Transformer\n(Cross‑Attention)]
  D --> E[LLM Head\n(LLaMA‑family, LoRA/QLoRA)]
  E --> F[Audio QA Response]
```

## Modules
- Speech Encoder: multilingual speech representation.
- Non‑Speech Encoder: environmental events/scene acoustics.
- Joint Embedding: concatenation + projection + normalization.
- Fusion: cross‑attention letting text tokens query audio semantics.
- Reasoning Head: instruction‑tuned LLM for AQA generation.

## Training Phases
1. Pretraining: self‑supervised encoders on mixed corpora.
2. Alignment: contrastive audio–text (Audio‑CLIP‑style).
3. Instruction Tuning: Audio QA (AQA) with curated QA pairs.
4. Multi‑Task Aux Heads: diarization, emotion.
5. Evaluation & Safety: metrics, ablations, guardrails.

## Inference Paths
- With Question: audio + question → fused reasoning → text answer.
- Without Question: audio only → scene summary and salient events.

## Deployment
- FastAPI microservice; optional Gradio front-end.
- GPU preferred; CPU fallback with quantization.
- Telemetry: latency, confidence, safety flags.
