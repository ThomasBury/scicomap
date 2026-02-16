set shell := ["bash", "-cu"]

venv := ".venv.just"

default:
  @just --list

sync:
  UV_PROJECT_ENVIRONMENT={{venv}} uv sync --extra lint --extra test --extra docs

sync-docs:
  UV_PROJECT_ENVIRONMENT={{venv}} uv sync --extra docs

check: sync
  UV_PROJECT_ENVIRONMENT={{venv}} uv run python -m pytest
  UV_PROJECT_ENVIRONMENT={{venv}} uv run ruff check src tests
  UV_PROJECT_ENVIRONMENT={{venv}} uv run ruff format --check src tests

docs: sync-docs
  rm -rf docs/build/html
  UV_PROJECT_ENVIRONMENT={{venv}} uv run sphinx-build -n -b html docs/source docs/build/html
  UV_PROJECT_ENVIRONMENT={{venv}} uv run python scripts/build_llm_assets.py

marimo: sync-docs
  UV_PROJECT_ENVIRONMENT={{venv}} uv run marimo export html-wasm docs/marimo/tutorial_app_lite.py -o docs/build/html/marimo --mode run
  touch docs/build/html/.nojekyll

validate-doc-artifacts:
  test -f docs/build/html/marimo/index.html
  test -f docs/build/html/.nojekyll
  test -f docs/build/html/marimo/.nojekyll
  test -f docs/build/html/llms.txt
  UV_PROJECT_ENVIRONMENT={{venv}} uv run python -c "from pathlib import Path; md=list((Path('docs/build/html/llm')).rglob('*.md')); assert md, 'No markdown mirrors generated'"

validate-pages-artifacts: validate-doc-artifacts
  UV_PROJECT_ENVIRONMENT={{venv}} uv run python -c "from pathlib import Path; text=Path('docs/build/html/llms.txt').read_text(encoding='utf-8'); assert 'getting-started' in text, 'llms.txt missing getting-started'; assert 'user-guide' in text, 'llms.txt missing user-guide'"

build: sync
  rm -rf dist
  UV_PROJECT_ENVIRONMENT={{venv}} uv run --with build python -m build
  UV_PROJECT_ENVIRONMENT={{venv}} uv run --with twine python -m twine check dist/*

release-check: check docs marimo validate-doc-artifacts build

smoke-testpypi version:
  rm -rf .venv.testpypi
  uv venv .venv.testpypi
  uv pip install --python .venv.testpypi/bin/python -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple scicomap=={{version}}
  .venv.testpypi/bin/python -c "import scicomap; print(scicomap.__version__)"
  .venv.testpypi/bin/scicomap version

tag version:
  git tag {{version}}

push-tag version:
  git push origin {{version}}

tag-rc version:
  git tag {{version}}rc1

push-tag-rc version:
  git push origin {{version}}rc1
