# BUILD TARGETS
.PHONY: build
build:
	docker build -t synthetic-text-watermarking .

.PHONY: demo
demo:
	docker compose up --build


# RUN TARGETS
.PHONY: web
web:
	uv run streamlit run src/synthetic_text_watermarking/web/Home.py


# TEST TARGETS
.PHONY: test
test:
	uv run ruff check --fix src
