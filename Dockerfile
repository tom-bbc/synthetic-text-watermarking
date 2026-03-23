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
COPY . /wrk

# Sync the project into a new environment
RUN uv sync --locked
