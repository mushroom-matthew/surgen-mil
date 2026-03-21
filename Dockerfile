# SurGen MIL — Docker image
#
# Supported commands:
#   Smoke test (no real data needed):
#     docker run --rm surgen-mil python scripts/smoke_test.py
#
#   Evaluate a checkpoint:
#     docker run --rm \
#       -v /path/to/data:/mnt/data-surgen:ro \
#       -v /path/to/outputs:/app/outputs:ro \
#       surgen-mil \
#       python scripts/evaluate.py \
#         --config configs/uni_mean_fair.yaml \
#         --checkpoint outputs/uni_mean_fair/runs/latest/checkpoint.pt \
#         --split test
#
#   Inference API (single model via env vars):
#     docker run --rm -p 8000:8000 \
#       -v /path/to/data:/mnt/data-surgen:ro \
#       -v /path/to/outputs:/app/outputs:ro \
#       -e MODEL_CONFIG=configs/uni_mean_fair.yaml \
#       -e MODEL_CHECKPOINT=outputs/uni_mean_fair/runs/latest/checkpoint.pt \
#       surgen-mil \
#       uvicorn api:app --host 0.0.0.0 --port 8000

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# git is needed for git_commit.txt artifact and pip VCS installs
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source (layer cache)
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt \
    && pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.29"

# Copy source and install the package
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Default: smoke test (validates the pipeline without real data)
CMD ["python", "scripts/smoke_test.py"]
