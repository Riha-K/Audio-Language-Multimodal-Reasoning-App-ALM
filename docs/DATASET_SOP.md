# Dataset Preparation SOP – ALM (Audio‑Language Model)

## 1. Scope
Define a robust, ethical, multilingual dataset for joint speech + non‑speech reasoning across Asian languages (hi, ur, te, ta, bn, zh, en).

## 2. Sources & Acquisition
- Speech: Mozilla Common Voice; domain recordings (public consented data).
- Non‑speech: AudioSet, ESC‑50, UrbanSound8K, TAU Urban Acoustic Scenes.
- Affect: RAVDESS, CREMA‑D for emotion proxies.
- Contextual: YouTube/field recordings (airports, markets, transit) with logging of license/consent.

## 3. Storage & Formats
- Audio: 16 kHz WAV (mono preferred), normalized gain.
- Metadata: JSONL records (one object per audio file) with schema below.
- Versioning: DVC or Git‑LFS; checksums for integrity.

## 4. Canonical Record Schema
```
{
  "audio_file": "relative/path.wav",
  "duration_sec": 13.4,
  "language": "hi-IN",
  "transcript": "…",
  "audio_events": ["airplane", "announcement"],
  "speaker_spans": [
    {"id": "spk1", "start": 0.40, "end": 5.20},
    {"id": "spk2", "start": 6.10, "end": 12.80}
  ],
  "emotion": "excited",
  "scene": "airport",
  "qa_pairs": [
    {"question": "Where is the person?", "answer": "Airport waiting area"},
    {"question": "What event is happening?", "answer": "Boarding announcement"}
  ],
  "license": "CC‑BY/consented",
  "source": "CommonVoice/YouTube/Field",
  "split": "train/val/test"
}
```

## 5. Pipelines
### 5.1 Pre‑processing
- Resample → 16 kHz mono; normalize; trim leading/trailing silence (threshold −40 dB).
- Loudness normalization (e.g., LUFS target).

### 5.2 Auto‑Labeling
- Transcription: Whisper (medium/base) → language‑tag + punctuation.
- Event Tags: AudioSet taggers/BEATs; top‑k events with confidences.
- Diarization: pyannote.audio → speaker turns.
- Emotion: Pretrained emotion classifier (target: arousal/valence + categorical tag).
- QA Generation: LLM prompt with metadata (scene hints, events, transcript) to produce 2–4 QA pairs per file. Human review for a sample.

### 5.3 Human QA & Curation
- Sampling plan (e.g., 10% random per language) for human validation.
- Checks: transcript WER vs human, event tag plausibility, QA factuality, language correctness.
- Fix policy: correct labels; mark uncertain items; exclude if confidence < threshold.

### 5.4 Balancing & Splits
- Balance by language, scene, duration bins, speaker counts.
- Stratified split into train/val/test (e.g., 80/10/10) ensuring speaker and source disjointness across splits.

## 6. Governance & Ethics
- Record consent, licenses, and provenance. Exclude sensitive PII content.
- Provide opt‑out mechanism and takedown workflow.
- Maintain a data statement and model card.

## 7. Quality Metrics
- ASR quality: WER/CER per language.
- Event quality: Precision@k on a labeled subset.
- QA quality: Human Likert and GPT‑judge agreement.
- Coverage: per‑language hours, scene diversity, duration bands.

## 8. Delivery Artifacts
- Data repository with JSONL metadata and audio.
- Stats report (hours per language, label distributions, QA counts).
- Labeling logs (tool versions, prompts, thresholds).
- README with scripts to reproduce pipelines.
