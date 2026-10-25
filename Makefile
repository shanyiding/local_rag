.PHONY: install test ingest chat serve eval

install:
	pip install -e ".[dev]"

test:
	pytest -q

ingest:
	python -m localrag ingest

chat:
	python -m localrag chat

serve:
	python -m localrag serve

eval:
	python -m localrag eval
