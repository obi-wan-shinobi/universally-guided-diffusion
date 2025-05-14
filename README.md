# UGDiff – Universally Guided Diffusion

## Installation

### Option 1: Using Poetry (recommended)

1. **Install Poetry** (if not already installed):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies and the project

```bash
poetry install
```

3. Activate the virtual environment created by poetry

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Option 2: Without Poetry (using `venv` + `pip`... boo!)

I would strongly recommend you try poetry because it is great for package management and
for project management as a whole but if you like to make life difficult for yourself, then you can do this.

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install the project in editable mode (with `pip`):

```bash
pip install -e .
```

## Running stable diffusion scripts

You can run the generate script through:

```bash
python3 scripts/generate.py
```
