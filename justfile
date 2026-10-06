set shell := ["bash", "-cu"]

default:
  @just --list

sync:
  uv sync --locked --extra lint --extra test

sync-docs:
  uv sync --locked --extra lint --extra test --extra docs

check: sync
  uv run --locked python -m pytest -m "not docs"
  uv run --locked ruff check src tests scripts
  uv run --locked ruff format --check src tests scripts
  uv run --locked ty check --error-on-warning

check-docs: sync-docs
  uv run --locked python -m pytest -m docs

docs: sync-docs
  rm -rf docs/build/html
  uv run --locked sphinx-build -n -W -b html docs/source docs/build/html
  uv run --locked python scripts/build_llm_assets.py

marimo: sync-docs
  uv run --locked marimo check docs/marimo/tutorial_app.py docs/marimo/tutorial_app_lite.py
  uv run --locked marimo export html-wasm docs/marimo/tutorial_app_lite.py -o docs/build/html/marimo --mode run
  mkdir -p docs/build/html/marimo/public
  uv build --wheel --out-dir docs/build/html/marimo/public
  touch docs/build/html/.nojekyll

validate-doc-artifacts:
  test -f docs/build/html/marimo/public/scicomap-2.0.0-py3-none-any.whl
  test -f docs/build/html/marimo/index.html
  test -f docs/build/html/.nojekyll
  test -f docs/build/html/marimo/.nojekyll
  test -f docs/build/html/llms.txt
  uv run --locked python -c "from pathlib import Path; md=list((Path('docs/build/html/llm')).rglob('*.md')); assert md, 'No markdown mirrors generated'"

validate-pages-artifacts: validate-doc-artifacts
  uv run --locked python -c "from pathlib import Path; text=Path('docs/build/html/llms.txt').read_text(encoding='utf-8'); assert 'getting-started' in text, 'llms.txt missing getting-started'; assert 'user-guide' in text, 'llms.txt missing user-guide'"

build: sync
  rm -rf dist
  uv run --locked --with build python -m build
  uv run --locked --with twine python -m twine check dist/*

release-check: check docs check-docs marimo validate-doc-artifacts build
  uv run --isolated --no-project --with ./dist/scicomap-2.0.0-py3-none-any.whl python scripts/smoke_wheel.py

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
