# ACE-Step 1.5 XL RunPod Serverless Endpoint
# GPU: NVIDIA A40 (48GB GDDR6, Ampere sm_86)
# - bfloat16 inference (native Ampere support)
# - SDPA attention (flash-attn skipped due to torch 2.4.0 ABI mismatch)
# - XL DiT (~9 GB bf16) + 1.7B LM fit well under 48GB; no offload needed
# - Weights live on RunPod network volume at /runpod-volume (not baked)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /app/ACE-Step-1.5 && \
    rm -rf /app/ACE-Step-1.5/.cache /app/ACE-Step-1.5/.git

WORKDIR /app/ACE-Step-1.5

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "safetensors==0.7.0" \
    "transformers>=4.51.0,<4.58.0" \
    "diffusers>=0.27.0,<0.33.0" \
    "scipy>=1.10.1" \
    "soundfile>=0.13.1" \
    "loguru>=0.7.3" \
    "einops>=0.8.1" \
    "accelerate>=1.12.0" \
    "numba>=0.63.1" \
    "vector-quantize-pytorch>=1.27.15" \
    "peft>=0.18.0" \
    "toml" \
    "xxhash" \
    "diskcache" \
    "modelscope" \
    "typer-slim>=0.21.1"

RUN pip install --no-cache-dir \
    torch==2.4.0+cu124 \
    torchaudio==2.4.0+cu124 \
    torchvision==0.19.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir torchcodec || true

RUN pip install --no-cache-dir "triton>=3.0.0" || true

RUN cd acestep/third_parts/nano-vllm && pip install --no-cache-dir --no-deps -e .

RUN pip install --no-cache-dir --no-deps -e .

RUN pip install --no-cache-dir runpod requests

RUN pip uninstall -y torchao flash-attn 2>/dev/null || true

WORKDIR /app
COPY handler.py /app/handler.py

# ── XL-specific environment ──
ENV ACESTEP_PROJECT_ROOT=/app/ACE-Step-1.5
ENV ACESTEP_CONFIG_PATH=acestep-v15-xl-base
ENV ACESTEP_CHECKPOINTS_DIR=/runpod-volume/checkpoints
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ENV ACESTEP_LM_BACKEND=vllm
ENV PYTHONUNBUFFERED=1

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV ACESTEP_COMPILE_MODEL=1
ENV ACESTEP_MAX_BATCH_SIZE=4
ENV ACESTEP_INFERENCE_STEPS_DEFAULT=50
ENV ACESTEP_GUIDANCE_SCALE_DEFAULT=7.0
ENV ACESTEP_OFFLOAD_DIT_TO_CPU=0

CMD ["python", "-u", "/app/handler.py"]
