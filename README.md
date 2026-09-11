# Artificial Neural Networks & Deep Learning — portfolio

Reports for the Insper ANN-DL course, published at <https://thiago-gouveia.github.io/ann-dl/>.

| Exercise | Report | Code |
|---|---|---|
| Data | [docs/exercises/data/index.md](docs/exercises/data/index.md) | [docs/exercises/data/code/data.py](docs/exercises/data/code/data.py) |

## Setup

```shell
python3 -m venv env
source ./env/bin/activate
python3 -m pip install -r requirements.txt --upgrade
```

## Re-running an exercise

```shell
python docs/exercises/data/code/data.py
```

The script regenerates every figure in `docs/exercises/data/figures/` and the numbers in
`docs/exercises/data/figures/results.json` (fixed seed, fully reproducible).

## Building the site

```shell
mkdocs serve -o
```

Every push to `main` is published to GitHub Pages by `.github/workflows/main.yaml`.
