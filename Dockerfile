FROM python:3.13-slim-trixie

WORKDIR /wrk

# Install make
RUN apt update && apt install -y make

# Install uv by copying the binary from the official distroless Docker image
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_DEV=0
ENV UV_PYTHON_DOWNLOADS=0

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project into the image
COPY unified-watermarking/ /wrk/unified-watermarking
COPY README.md /wrk/
COPY Makefile /wrk/
COPY pyproject.toml /wrk/
COPY uv.lock /wrk/
COPY src/ /wrk/src

# Sync the project into a new environment
RUN uv sync --locked

# Install Unified Watermarking Framework submodule into venv
# RUN . .venv/bin/activate && \
#     cd unified-watermarking && \
#     uv pip install vllm --torch-backend=auto && \
#     uv pip install -e .
