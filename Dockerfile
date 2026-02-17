FROM python:3.13-slim-trixie

WORKDIR /wrk

# Install make
RUN apt update && apt install -y make

# Install uv by copying the binary from the official distroless Docker image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project into the image
COPY . /wrk

# Sync the project into a new environment, asserting the lockfile is up to date
RUN uv sync --locked
