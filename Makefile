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
	uv run streamlit run src/synthetic_text_watermarking/web/homepage.py --server.port 4040


# TEST TARGETS
.PHONY: test
test:
	uv run ruff check --fix src
