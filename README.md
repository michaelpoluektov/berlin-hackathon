# BERLIN

Install [`uv`](https://github.com/astral-sh/uv) to manage the Python stuff

## Prerequisites

- Python installed (check with `python --version` or `python3 --version`)
- `uv` installed

If you don’t have `uv` yet, install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell or follow any instructions printed by the installer.

You can verify the installation with:

```bash
uv --version
```

Setup virtual environment:

```bash
uv sync
uv venv
source .venv/bin/activate
```

Setup playwright:

```bash
python -m playwright install
```

Test that everything works:

```bash
python test_playwright.py
```
