# ExperteaseAI Portable GPU Worker

This package contains GPU inference only. It intentionally does not contain
Agent orchestration, RAG retrieval, Qdrant, tool logic, or business APIs.

## Endpoints

- `GET /health`
- `POST /embed` -> BGE-M3
- `POST /decompose` -> shared small chat model
- `POST /generate` with `profile="decision"` -> shared small chat model
- `POST /generate` with `profile="response"` -> response model

The small chat model is loaded once and is shared by the RAG decomposer and the
Agent decision model.

## Local machine

Configure `.env` to load only models that fit locally, for example:

```env
GPU_LOAD_EMBEDDING=true
GPU_LOAD_SMALL_LLM=true
GPU_LOAD_RESPONSE_LLM=false
```

## AWS GPU

Use the same source code but point the environment variables at AWS model
directories and enable all required profiles.

## Run

From the repository root:

```bash
uvicorn gpu_worker.server:app --host 0.0.0.0 --port 8001
```

`/health` reports which profiles are actually available. The future automatic
router can use this information to route unsupported local profiles directly
to AWS rather than intentionally causing an OOM.
