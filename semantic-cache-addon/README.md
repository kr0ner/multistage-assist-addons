# Semantic Cache Add-on

Home Assistant addon providing semantic cache lookup and CrossEncoder reranking for Multi-Stage Assist.

## Features

- **Hybrid Search** - Combines BM25 keyword search + vector similarity for robust retrieval
- **CrossEncoder Reranking** - Precise semantic validation with INT8 quantization
- **NUC Optimized** - INT8 quantization enables ~35ms reranking on CPU (vs ~600ms unoptimized)
- **GPU Acceleration** - NVIDIA CUDA, Apple Silicon (MPS), Intel OpenVINO support
- **Local Models** - No external API dependencies

## Architecture

The addon implements a state-of-the-art "Funnel" architecture optimized for Intel NUC and similar hardware:

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
│                  "Mach es dunkel im Wohnzimmer"                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Fast Retrieval (Wide & Fast)           ~20ms total    │
│  ┌────────────────────┐    ┌────────────────────┐               │
│  │   Vector Search    │    │   BM25 Keywords    │               │
│  │   (MiniLM-L12)     │    │   (Exact Match)    │               │
│  │   ~15ms            │    │   <5ms             │               │
│  │                    │    │                    │               │
│  │ Finds: "dunkel"    │    │ Finds: "Wohnzimmer"│               │
│  │ → "Rollladen"      │    │ exact word match   │               │
│  └────────────────────┘    └────────────────────┘               │
│              │                      │                           │
│              └──────────┬───────────┘                           │
│                         ▼                                       │
│              Top 10 Candidates (deduplicated)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Precise Reranking (Narrow & Smart)     ~35ms          │
│  ┌────────────────────────────────────────────┐                 │
│  │     CrossEncoder (BGE-M3 INT8 Quantized)   │                 │
│  │                                            │                 │
│  │  Compares query against each candidate:    │                 │
│  │  • "Mach es dunkel" vs "Rollladen runter"  │  → 0.94 ✓       │
│  │  • "Mach es dunkel" vs "Licht aus"         │  → 0.72         │
│  │  • "Mach es dunkel" vs "Rollladen hoch"    │  → 0.31         │
│  └────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  RESULT: Best match above threshold                             │
│  Intent: HassCloseCover, Entity: cover.wohnzimmer_rollladen     │
│  Total Latency: ~55ms                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

| Problem | Solution |
|---------|----------|
| "an" vs "aus" confusion | BM25 catches exact keywords |
| Abstract commands ("Mach es dunkel") | Vector search finds semantic matches |
| False positives from either method | CrossEncoder validates the final answer |
| CPU too slow for large models | INT8 quantization: 560M params in 35ms |

## Installation

1. Add this repository to your Home Assistant addon store
2. Install "Semantic Cache Add-on"
3. Configure and start

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `reranker_model` | `BAAI/bge-reranker-v2-m3` | CrossEncoder model (560M, INT8 quantized on CPU) |
| `embedding_model` | `paraphrase-multilingual-MiniLM-L12-v2` | Fast embedding model (118M) |
| `device` | `cpu` | Device: `cpu`, `cuda`, `mps`, `openvino:GPU` |
| `port` | `9876` | API port |
| `anchors_file` | `/homeassistant/.storage/multistage_assist_anchors.json` | Pre-generated anchors |
| `user_cache_file` | `/homeassistant/.storage/multistage_assist_semantic_cache.json` | User-learned cache |
| `hybrid_alpha` | `0.6` | Semantic vs BM25 weight (0=all BM25, 1=all vector) |
| `vector_threshold` | `0.5` | Min hybrid score for candidates |
| `vector_top_k` | `10` | Candidates to rerank |
| `reranker_threshold` | `0.73` | Min reranker score for cache hit |

### Model Selection Guide

| Hardware | Recommended Config |
|----------|-------------------|
| Intel NUC (CPU) | `device: cpu` (uses INT8 quantization automatically) |
| Intel NUC (iGPU) | `device: openvino:GPU` |
| NVIDIA GPU | `device: cuda` |
| Apple Silicon | `device: mps` |

### Performance Tuning

For **faster response** (may reduce accuracy):
```yaml
embedding_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2"
vector_top_k: 5
```

For **higher accuracy** (slower, requires good hardware):
```yaml
embedding_model: "BAAI/bge-m3"
vector_top_k: 15
reranker_threshold: 0.8
```

## API

### Health Check
```
GET /health
```

Response:
```json
{
  "status": "ok",
  "reranker_model": "BAAI/bge-reranker-v2-m3",
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "device": "cpu",
  "cache_entries": 1234,
  "last_reload": 1706900000.0
}
```

### Cache Lookup
```
POST /lookup
{
  "query": "Schalte das Licht im Bad an"
}
```

Response (hit):
```json
{
  "found": true,
  "intent": "HassTurnOn",
  "entity_ids": ["light.bad"],
  "slots": {"area": "Bad", "domain": "light"},
  "score": 0.85,
  "original_text": "Schalte Licht in Bad an",
  "reranked": true
}
```

Response (miss):
```json
{
  "found": false,
  "score": 0.0
}
```

### Rerank (standalone)
```
POST /rerank
{
  "query": "Turn on the kitchen light",
  "candidates": [
    "Switch on the lamp in kitchen",
    "Turn off bedroom light"
  ]
}
```

Response:
```json
{
  "scores": [0.89, 0.32],
  "best_index": 0,
  "best_score": 0.89
}
```

### Embed (generate cache entries)
```
POST /embed
{
  "entries": [
    {
      "text": "Schalte Licht im Bad an",
      "intent": "HassTurnOn",
      "entity_ids": ["light.bad"],
      "slots": {"area": "Bad", "domain": "light"}
    }
  ]
}
```

Response:
```json
{
  "entries": [
    {
      "text": "Schalte Licht im Bad an",
      "intent": "HassTurnOn",
      "entity_ids": ["light.bad"],
      "slots": {"area": "Bad", "domain": "light"},
      "embedding": [0.123, -0.456, ...],
      "generated": true
    }
  ],
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "embedding_dim": 384
}
```

### Embed Text (single text embedding)
```
POST /embed/text
{
  "text": "Schalte das Licht an"
}
```

Response:
```json
{
  "text": "Schalte das Licht an",
  "embedding": [0.123, -0.456, ...],
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "embedding_dim": 384
}
```

## Multi-Stage Assist Configuration

In your Multi-Stage Assist config:
```yaml
reranker_ip: "localhost"  # or addon hostname
reranker_port: 9876
reranker_enabled: true
reranker_mode: "api"
```

## INT8 Quantization (NUC Optimization)

When running on CPU (`device: cpu`), the addon automatically applies **INT8 dynamic quantization** to the reranker model. This compresses the model's Linear layers from 32-bit floats to 8-bit integers, providing:

- **~10-20x speedup**: 600ms → 35ms inference time
- **~90%+ accuracy retained**: Minimal quality loss for semantic matching
- **Lower memory usage**: Reduced RAM footprint

This optimization is applied automatically—no configuration needed. The logs will show:
```
⚡ NUC MODE: Loading BAAI/bge-reranker-v2-m3 with INT8 Quantization...
✅ Model Quantized & Ready. Expect ~30-40ms latency.
```

## Latency Budget (Intel NUC)

| Stage | Component | Latency |
|-------|-----------|---------|
| 1a | Embedding (MiniLM-L12) | ~15ms |
| 1b | BM25 + Vector Search | ~5ms |
| 2 | Reranking (BGE-M3 INT8) | ~35ms |
| **Total** | | **~55ms** |
