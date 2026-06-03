# DEFAULT VALUES
# ---------------------------------------------------------------------------

MODEL_NAME ?= ai/gemma3-vllm:270M


# BUILD TARGETS
# ---------------------------------------------------------------------------

.PHONY: build
build:
	docker build -t synthetic-text-watermarking --platform linux/arm64 .


.PHONY: model
model:
	@docker model pull $(MODEL_NAME)
	@docker model configure \
        --hf_overrides '{"dtype": "auto", "logits_processors": "lm_wm_tools.watermarks:WatermarkLogitsProcessor", "generation_config": "vllm"}' \
		$(MODEL_NAME)


# RUN TARGETS
# ---------------------------------------------------------------------------

.PHONY: demo-up
demo-up:
	docker compose up --build --detach


.PHONY: demo-down
demo-down:
	docker compose down web


# TEST TARGETS
# ---------------------------------------------------------------------------

.PHONY: test
test:
	uv run ruff check --fix src
