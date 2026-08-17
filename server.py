"""Portable local/AWS GPU inference service.

Endpoints
---------
GET  /health
POST /embed
POST /decompose
POST /generate

The same server can run on a local GPU or AWS. Individual model groups are
enabled through environment variables so a smaller local device can advertise
only the profiles it can actually serve.

Physical model mapping
----------------------
BGE-M3:
    embedding

Small chat model (for example Qwen 4B):
    decomposer
    decision

Large chat model (for example Qwen 30B):
    response
"""

from __future__ import annotations

import asyncio
import hmac
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Callable, Literal
from uuid import uuid4

import torch
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Response,
)
from pydantic import BaseModel, Field

from .config import GPUWorkerSettings, load_settings
from .models import (
    BGEM3EmbeddingModel,
    ChatGeneration,
    ChatModel,
    QueryDecomposer,
)


SETTINGS: GPUWorkerSettings = load_settings()

REQUEST_ID_HEADER = "X-Request-ID"
TRACE_ID_HEADER = "X-Langfuse-Trace-Id"
PARENT_SPAN_ID_HEADER = "X-Langfuse-Parent-Span-Id"


# ---------------------------------------------------------------------------
# Loaded models
# ---------------------------------------------------------------------------

embedding_model: BGEM3EmbeddingModel | None = None
small_llm: ChatModel | None = None
decomposer: QueryDecomposer | None = None
response_llm: ChatModel | None = None


# ---------------------------------------------------------------------------
# Shared GPU queue
# ---------------------------------------------------------------------------

GPUCallable = Callable[[float], Any]


@dataclass(slots=True)
class GPUJob:
    request_id: str
    job_type: str
    function: GPUCallable
    future: asyncio.Future[Any]
    submitted_at: float


@dataclass(slots=True)
class GPUJobResult:
    request_id: str
    value: Any
    queue_wait_ms: float
    execution_ms: float


gpu_queue: asyncio.Queue[GPUJob] | None = None
gpu_worker_task: asyncio.Task[None] | None = None
gpu_executor: ThreadPoolExecutor | None = None

current_gpu_job_type: str | None = None
current_gpu_request_id: str | None = None
completed_gpu_jobs = 0


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------

class RequestContext(BaseModel):
    user_id: str | None = None
    session_id: str | None = None
    collection_name: str | None = None
    pipeline_request_id: str | None = None


class EmbedRequest(RequestContext):
    texts: list[str] = Field(
        min_length=1,
        max_length=64,
    )


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dimension: int


class DecomposeRequest(RequestContext):
    userquery: str = Field(min_length=1)
    max_questions: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class DecomposeResponse(BaseModel):
    questions: list[str]


class ChatMessage(BaseModel):
    role: Literal[
        "system",
        "user",
        "assistant",
        "tool",
    ]
    content: str = Field(min_length=1)


class GenerateRequest(BaseModel):
    profile: Literal["decision", "response"]
    messages: list[ChatMessage] = Field(
        min_length=1,
        max_length=100,
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
    )


class GenerateResponse(BaseModel):
    text: str
    model_name: str
    finish_reason: str | None = None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    generation_seconds: float = Field(ge=0)
    tokens_per_second: float | None = Field(
        default=None,
        ge=0,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request_id(incoming: str | None) -> str:
    if incoming and incoming.strip():
        return incoming.strip()[:200]
    return uuid4().hex


def _authorize(authorization: str | None) -> None:
    configured = SETTINGS.api_key

    if not configured:
        return

    prefix = "Bearer "
    if (
        not authorization
        or not authorization.startswith(prefix)
    ):
        raise HTTPException(
            status_code=401,
            detail="Missing GPU service bearer token.",
        )

    provided = authorization[len(prefix):].strip()

    if not hmac.compare_digest(
        provided,
        configured,
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid GPU service bearer token.",
        )


def _set_job_headers(
    response: Response,
    result: GPUJobResult,
) -> None:
    response.headers[REQUEST_ID_HEADER] = (
        result.request_id
    )
    response.headers["X-GPU-Queue-Wait-Ms"] = (
        f"{result.queue_wait_ms:.2f}"
    )
    response.headers["X-GPU-Execution-Ms"] = (
        f"{result.execution_ms:.2f}"
    )


def _gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "gpu": None,
            "free_vram_gb": None,
            "total_vram_gb": None,
        }

    free, total = torch.cuda.mem_get_info()

    return {
        "cuda_available": True,
        "gpu": torch.cuda.get_device_name(0),
        "free_vram_gb": round(
            free / 1024**3,
            2,
        ),
        "total_vram_gb": round(
            total / 1024**3,
            2,
        ),
    }


def _loaded_profiles() -> dict[str, bool]:
    return {
        "embedding": embedding_model is not None,
        "decomposer": decomposer is not None,
        "decision": small_llm is not None,
        "response": response_llm is not None,
    }


def _model_labels() -> dict[str, str | None]:
    return {
        "embedding": (
            embedding_model.model_name
            if embedding_model is not None
            else None
        ),
        "small_llm": (
            small_llm.model_name
            if small_llm is not None
            else None
        ),
        "response_llm": (
            response_llm.model_name
            if response_llm is not None
            else None
        ),
    }


def _execute_gpu_callable(
    function: GPUCallable,
    queue_wait_ms: float,
) -> Any:
    try:
        with torch.inference_mode():
            return function(queue_wait_ms)
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


async def _gpu_worker() -> None:
    global current_gpu_job_type
    global current_gpu_request_id
    global completed_gpu_jobs

    assert gpu_queue is not None
    assert gpu_executor is not None

    loop = asyncio.get_running_loop()

    while True:
        job = await gpu_queue.get()

        try:
            if job.future.cancelled():
                continue

            current_gpu_job_type = job.job_type
            current_gpu_request_id = job.request_id

            queue_wait_ms = (
                time.perf_counter() - job.submitted_at
            ) * 1000

            execution_started = time.perf_counter()

            value = await loop.run_in_executor(
                gpu_executor,
                _execute_gpu_callable,
                job.function,
                queue_wait_ms,
            )

            execution_ms = (
                time.perf_counter() - execution_started
            ) * 1000

            if not job.future.cancelled():
                job.future.set_result(
                    GPUJobResult(
                        request_id=job.request_id,
                        value=value,
                        queue_wait_ms=queue_wait_ms,
                        execution_ms=execution_ms,
                    )
                )

            completed_gpu_jobs += 1

        except asyncio.CancelledError:
            if not job.future.done():
                job.future.cancel()
            raise

        except Exception as exc:
            if (
                not job.future.cancelled()
                and not job.future.done()
            ):
                job.future.set_exception(exc)

        finally:
            current_gpu_job_type = None
            current_gpu_request_id = None
            gpu_queue.task_done()


async def _submit_gpu_job(
    *,
    request_id: str,
    job_type: str,
    function: GPUCallable,
    timeout_seconds: float,
) -> GPUJobResult:
    if (
        gpu_queue is None
        or gpu_worker_task is None
        or gpu_worker_task.done()
    ):
        raise HTTPException(
            status_code=503,
            detail="GPU worker is not ready.",
        )

    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = (
        loop.create_future()
    )

    job = GPUJob(
        request_id=request_id,
        job_type=job_type,
        function=function,
        future=future,
        submitted_at=time.perf_counter(),
    )

    try:
        gpu_queue.put_nowait(job)
    except asyncio.QueueFull as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "The GPU server is currently "
                    "at capacity."
                ),
                "queue_capacity": SETTINGS.queue_maxsize,
            },
            headers={"Retry-After": "3"},
        ) from exc

    try:
        return await asyncio.wait_for(
            future,
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        future.cancel()
        raise HTTPException(
            status_code=504,
            detail={
                "message": (
                    "The model request exceeded "
                    "its timeout."
                ),
                "job_type": job_type,
                "timeout_seconds": timeout_seconds,
            },
        ) from exc


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def _load_models() -> None:
    global embedding_model
    global small_llm
    global decomposer
    global response_llm

    print(
        "Starting ExperteaseAI GPU worker...",
        flush=True,
    )
    print(
        "GPU before loading:",
        _gpu_info(),
        flush=True,
    )

    if SETTINGS.load_embedding:
        assert SETTINGS.bge_model_path is not None

        print("Loading BGE-M3...", flush=True)
        embedding_model = BGEM3EmbeddingModel(
            model_path=SETTINGS.bge_model_path,
            max_seq_length=(
                SETTINGS.bge_max_seq_length
            ),
            trust_remote_code=(
                SETTINGS.trust_remote_code
            ),
        )
        embedding_model.warm_up()
        print("BGE-M3 loaded.", flush=True)

    if SETTINGS.load_small_llm:
        assert (
            SETTINGS.small_llm_model_path
            is not None
        )

        print(
            "Loading shared small chat model...",
            flush=True,
        )
        small_llm = ChatModel(
            model_path=(
                SETTINGS.small_llm_model_path
            ),
            enable_thinking=(
                SETTINGS.small_llm_enable_thinking
            ),
            trust_remote_code=(
                SETTINGS.trust_remote_code
            ),
            device_map=SETTINGS.device_map,
        )

        decomposer = QueryDecomposer(
            model=small_llm,
            max_new_tokens=(
                SETTINGS.decompose_max_new_tokens
            ),
        )

        print(
            "Shared small model loaded "
            "for decision + decomposition.",
            flush=True,
        )

    if SETTINGS.load_response_llm:
        assert (
            SETTINGS.response_llm_model_path
            is not None
        )

        # If both roles intentionally point to the exact
        # same physical model, share it rather than loading
        # a duplicate copy.
        if (
            small_llm is not None
            and SETTINGS.response_llm_model_path
            == SETTINGS.small_llm_model_path
        ):
            response_llm = small_llm
        else:
            print(
                "Loading response chat model...",
                flush=True,
            )
            response_llm = ChatModel(
                model_path=(
                    SETTINGS.response_llm_model_path
                ),
                enable_thinking=(
                    SETTINGS.response_llm_enable_thinking
                ),
                trust_remote_code=(
                    SETTINGS.trust_remote_code
                ),
                device_map=SETTINGS.device_map,
            )

        print(
            "Response model loaded.",
            flush=True,
        )

    print(
        "GPU after loading:",
        _gpu_info(),
        flush=True,
    )
    print(
        "Loaded profiles:",
        _loaded_profiles(),
        flush=True,
    )


def _unload_models() -> None:
    global embedding_model
    global small_llm
    global decomposer
    global response_llm

    embedding_model = None
    decomposer = None
    response_llm = None
    small_llm = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gpu_queue
    global gpu_worker_task
    global gpu_executor

    gpu_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="expertease-gpu",
    )

    loop = asyncio.get_running_loop()

    try:
        await loop.run_in_executor(
            gpu_executor,
            _load_models,
        )

        gpu_queue = asyncio.Queue(
            maxsize=SETTINGS.queue_maxsize
        )
        gpu_worker_task = asyncio.create_task(
            _gpu_worker(),
            name="expertease-gpu-queue-worker",
        )

        yield

    finally:
        if gpu_worker_task is not None:
            gpu_worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await gpu_worker_task

        if gpu_executor is not None:
            try:
                await loop.run_in_executor(
                    gpu_executor,
                    _unload_models,
                )
            finally:
                await asyncio.to_thread(
                    gpu_executor.shutdown,
                    wait=True,
                    cancel_futures=True,
                )


app = FastAPI(
    title="ExperteaseAI GPU Worker",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "profiles": _loaded_profiles(),
        "configured_profiles": (
            SETTINGS.profile_configuration
        ),
        "models": _model_labels(),
        "gpu_worker_running": (
            gpu_worker_task is not None
            and not gpu_worker_task.done()
        ),
        "gpu_busy": (
            current_gpu_job_type is not None
        ),
        "current_gpu_job_type": (
            current_gpu_job_type
        ),
        "queue_waiting": (
            gpu_queue.qsize()
            if gpu_queue is not None
            else 0
        ),
        "queue_capacity": SETTINGS.queue_maxsize,
        "completed_gpu_jobs": completed_gpu_jobs,
        **_gpu_info(),
    }


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@app.post(
    "/embed",
    response_model=EmbedResponse,
)
async def embed(
    req: EmbedRequest,
    response: Response,
    authorization: str | None = Header(
        default=None,
        alias="Authorization",
    ),
    incoming_request_id: str | None = Header(
        default=None,
        alias=REQUEST_ID_HEADER,
    ),
):
    _authorize(authorization)

    if embedding_model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "Embedding profile is not "
                    "available on this worker."
                ),
                "profile": "embedding",
            },
        )

    request_id = _request_id(
        incoming_request_id
        or req.pipeline_request_id
    )

    try:
        result = await _submit_gpu_job(
            request_id=request_id,
            job_type="embedding",
            function=lambda _queue_wait_ms: (
                embedding_model.encode(
                    req.texts,
                    batch_size=(
                        SETTINGS.bge_batch_size
                    ),
                )
            ),
            timeout_seconds=(
                SETTINGS.embed_timeout_seconds
            ),
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise HTTPException(
            status_code=507,
            detail="CUDA out of memory during embedding.",
        ) from exc

    _set_job_headers(response, result)

    vectors = result.value
    return {
        "vectors": vectors,
        "dimension": (
            len(vectors[0]) if vectors else 0
        ),
    }


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

@app.post(
    "/decompose",
    response_model=DecomposeResponse,
)
async def decompose(
    req: DecomposeRequest,
    response: Response,
    authorization: str | None = Header(
        default=None,
        alias="Authorization",
    ),
    incoming_request_id: str | None = Header(
        default=None,
        alias=REQUEST_ID_HEADER,
    ),
):
    _authorize(authorization)

    if decomposer is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "Decomposer profile is not "
                    "available on this worker."
                ),
                "profile": "decomposer",
            },
        )

    request_id = _request_id(
        incoming_request_id
        or req.pipeline_request_id
    )

    try:
        result = await _submit_gpu_job(
            request_id=request_id,
            job_type="decomposition",
            function=lambda _queue_wait_ms: (
                decomposer.decompose(
                    req.userquery,
                    max_questions=req.max_questions,
                )
            ),
            timeout_seconds=(
                SETTINGS.decompose_timeout_seconds
            ),
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise HTTPException(
            status_code=507,
            detail=(
                "CUDA out of memory during "
                "query decomposition."
            ),
        ) from exc

    _set_job_headers(response, result)

    return {"questions": result.value}


# ---------------------------------------------------------------------------
# Generic Agent LLM generation
# ---------------------------------------------------------------------------

def _generate(
    req: GenerateRequest,
) -> ChatGeneration:
    if req.profile == "decision":
        model = small_llm
        default_max = (
            SETTINGS.decision_max_output_tokens
        )
    else:
        model = response_llm
        default_max = (
            SETTINGS.response_max_output_tokens
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    f"Profile {req.profile!r} is not "
                    "available on this worker."
                ),
                "profile": req.profile,
            },
        )

    requested_max = (
        req.max_output_tokens
        if req.max_output_tokens is not None
        else default_max
    )

    # The worker owns the hard profile ceiling even if
    # a client asks for an unexpectedly large generation.
    max_output_tokens = min(
        requested_max,
        default_max,
    )

    return model.generate(
        messages=[
            message.model_dump(mode="json")
            for message in req.messages
        ],
        max_new_tokens=max_output_tokens,
        temperature=req.temperature,
    )


@app.post(
    "/generate",
    response_model=GenerateResponse,
)
async def generate(
    req: GenerateRequest,
    response: Response,
    authorization: str | None = Header(
        default=None,
        alias="Authorization",
    ),
    incoming_request_id: str | None = Header(
        default=None,
        alias=REQUEST_ID_HEADER,
    ),
):
    _authorize(authorization)

    # Check availability before occupying a queue slot.
    if (
        req.profile == "decision"
        and small_llm is None
    ):
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "Decision profile is not "
                    "available on this worker."
                ),
                "profile": "decision",
            },
        )

    if (
        req.profile == "response"
        and response_llm is None
    ):
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "Response profile is not "
                    "available on this worker."
                ),
                "profile": "response",
            },
        )

    request_id = _request_id(
        incoming_request_id
    )

    try:
        result = await _submit_gpu_job(
            request_id=request_id,
            job_type=f"generation:{req.profile}",
            function=lambda _queue_wait_ms: (
                _generate(req)
            ),
            timeout_seconds=(
                SETTINGS.generate_timeout_seconds
            ),
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise HTTPException(
            status_code=507,
            detail={
                "message": (
                    "CUDA out of memory during "
                    "LLM generation."
                ),
                "profile": req.profile,
            },
        ) from exc

    _set_job_headers(response, result)

    generation: ChatGeneration = result.value

    return GenerateResponse(
        text=generation.text,
        model_name=generation.model_name,
        finish_reason=generation.finish_reason,
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
        generation_seconds=(
            generation.generation_seconds
        ),
        tokens_per_second=(
            generation.tokens_per_second
        ),
        metadata={
            "profile": req.profile,
        },
    )
