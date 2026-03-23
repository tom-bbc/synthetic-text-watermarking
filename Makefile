# BUILD TARGETS
.PHONY: build
build:
	docker build -t synthetic-text-watermarking --platform linux/arm64 .

.PHONY: demo-up
demo-up:
	docker compose up --build --detach

.PHONY: demo-down
demo-down:
	docker compose down web


# RUN TARGETS
.PHONY: web
web:
	uv run streamlit run src/synthetic_text_watermarking/web/Home.py


# TEST TARGETS
.PHONY: test
test:
	uv run ruff check --fix src
