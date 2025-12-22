# Semantic Cache Add-on

Home Assistant addon providing semantic cache lookup and CrossEncoder reranking for Multi-Stage Assist.

## Features

- **Semantic Cache Lookup** - Fast command resolution via vector + BM25 hybrid search
- **CrossEncoder Reranking** - Precise semantic validation
- **GPU Acceleration** - NVIDIA CUDA, Apple Silicon (MPS) support
- **Local Models** - No external API dependencies

## Installation

1. Add this repository to your Home Assistant addon store
2. Install "Semantic Cache Add-on"
3. Configure and start

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `reranker_model` | `BAAI/bge-reranker-base` | CrossEncoder model |
| `embedding_model` | `BAAI/bge-m3` | Embedding model |
| `device` | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` |
| `port` | `9876` | API port |
| `anchors_file` | `/homeassistant/.storage/multistage_assist_anchors.json` | Pre-generated anchors |
| `user_cache_file` | `/homeassistant/.storage/multistage_assist_semantic_cache.json` | User-learned cache |
| `hybrid_alpha` | `0.7` | Semantic vs BM25 weight (0-1) |
| `vector_threshold` | `0.5` | Min score for candidates |
| `vector_top_k` | `10` | Candidates to rerank |
| `reranker_threshold` | `0.73` | Min reranker score for hit |

## API

### Health Check
```
GET /health
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
  "embedding_model": "BAAI/bge-m3",
  "embedding_dim": 1024
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
