# Hyperparameters in Population Health Modeling

Hyperparameter tuning taught on a real population-health problem: predicting
**county-level chronic disease prevalence from health-related social needs**, using
CDC PLACES data.

Built for VS Code — plain `.py` files with `# %%` cell markers, so you get
notebook-style cell execution in the Interactive Window while the files stay
diffable, importable, and testable. No `.ipynb` anywhere.

## Why this framing

Most hyperparameter tutorials demonstrate regularization on data where it does
nothing visible, and stop before the question of whether the model means anything.
This one uses a dataset where the statistical problem is real: county-level social
need measures — food insecurity, housing insecurity, lack of transportation,
loneliness — are strongly correlated with each other, and an unpenalized regression
handles that badly.

It also shows the result you are supposed to get and rarely see written down: with
~3,100 counties and 8 predictors, **the penalty barely matters**. The script says so,
then builds the regime where it does matter (120 counties, 44 terms after a degree-2
expansion) and shows the U-shaped curve there. Knowing when a hyperparameter cannot
change your answer is worth more than knowing how to grid-search it.

## What's covered

| Script | Concept | Population-health anchor |
| --- | --- | --- |
| `01_assignment_operators.py` | `+=`, `-=`, `*=`; absolute vs relative change; scientific notation; float equality | Tracking learning rate and validation loss while modeling diabetes prevalence |
| `02_learning_rate_schedules.py` | Step / exponential / cosine decay; what learning rate does to convergence | SGD fits on county data at five learning rates, from too-small to divergent |
| `03_regularization_sdoh.py` | Ridge vs lasso, tuning `alpha`, regularization paths, the 1-SE rule | Collinear social-need predictors; state-grouped cross-validation |

Cross-cutting points the scripts make explicitly:

- **Absolute change → `+=` / `-=`; relative change → `*=`.** "Down 10%" is `*= 0.9`.
- **Relative changes compound.** `*= 1.05` then `*= 0.95` is a net *decrease* — the
  same arithmetic that makes "rose 20%, then fell 20%" a net decline in a rate report.
- **Search penalty strength on a log grid.** A linear grid spends its budget where
  nothing changes.
- **Scale inside the fold.** Standardization is a pipeline step, so fold statistics
  never leak from test to train.
- **Group folds by state.** Counties in a state share BRFSS sampling design and
  policy; a random split puts near-siblings on both sides and flatters the score.
- **Never compare floats with `==`** outside a controlled exercise.

## Data

**Source:** CDC PLACES, *Local Data for Better Health, County Data* (Socrata dataset
`swc5-untb`, model-based small-area estimates from BRFSS). No API key required.

- **Outcome:** diagnosed diabetes prevalence, crude, adults.
- **Predictors:** the seven health-related social needs measures (food insecurity,
  food stamp receipt, housing insecurity, utility shutoff threat, lack of reliable
  transportation, loneliness, lack of social/emotional support) plus uninsured share.

```bash
python scripts/download_data.py          # downloads and builds data/processed/county_analytic.csv
python scripts/download_data.py --list   # print every measure ID in the live release
```

CDC re-publishes PLACES annually and has changed measure IDs between releases, so the
download validates requested IDs against the live dataset first and prints what is
actually available if one has moved. Change the outcome, predictors, or crude vs
age-adjusted in `src/sdoh_prevalence/config.py`.

**Bringing your own data:** point `load_dataset()` at any CSV and describe its columns
with a `ColumnMap`. Everything downstream is unchanged.

```python
from sdoh_prevalence.config import ColumnMap
from sdoh_prevalence.data import load_dataset

columns = ColumnMap(
    outcome="dm_prevalence",
    predictors=("food_insecurity", "no_transport", "uninsured"),
    geo_id="fips", geo_name="county", group="state",
)
df = load_dataset(path="data/raw/my_extract.csv", columns=columns)
```

`data/` is gitignored except for `.gitkeep` files — **no dataset, public or client,
is ever committed**. If nothing has been downloaded, the loader generates synthetic
data with a loud warning and labels the frame `df.attrs["source"]`, so the repo runs
on a fresh clone and no figure is ever silently built on fake numbers.

## Setup

Python 3.10+.

```bash
git clone <repo-url>
cd hyperparameters-sdoh-prevalence

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # editable install; puts sdoh_prevalence on the path
```

Open the folder in VS Code and accept the recommended extensions (Python, Pylance,
Jupyter, Ruff) — `.vscode/` is committed and configures the interpreter paths, pytest,
and the Interactive Window root.

## Run it

**Cell by cell (the intended way):** open any script in `scripts/`, put the cursor in a
`# %%` cell and press **Shift+Enter**, or click **Run Cell** above the marker. The
Interactive Window keeps state between cells exactly like a notebook.

**Top to bottom:**

```bash
python scripts/download_data.py             # optional; synthetic fallback otherwise
python scripts/01_assignment_operators.py
python scripts/02_learning_rate_schedules.py
python scripts/03_regularization_sdoh.py
```

**Debug configurations:** each script has an entry in the Run and Debug panel.

Figures are written to `figures/` (gitignored — regenerate by running the scripts).

**Tests:**

```bash
pytest          # 35 tests
```

They cover the arithmetic with tolerances rather than `==`, the long→wide reshape,
the loader's three sources and its error messages, and the modeling invariants that
matter: grouped folds never split a state, scaling happens inside the pipeline, a
stronger penalty monotonically shrinks the coefficient norm, and a large learning rate
diverges.

## Layout

```
hyperparameters-sdoh-prevalence/
├── .vscode/                       # settings, extensions, launch configs (committed)
├── scripts/
│   ├── download_data.py           # CDC PLACES downloader, with measure validation
│   ├── 01_assignment_operators.py # # %% cells
│   ├── 02_learning_rate_schedules.py
│   └── 03_regularization_sdoh.py
├── src/sdoh_prevalence/
│   ├── config.py                  # paths, dataset IDs, measures, ColumnMap
│   ├── data.py                    # download, reshape, synthetic, loader
│   ├── hyperparameters.py         # operators, decay schedules, alpha grids
│   ├── models.py                  # pipelines, grouped CV, tuning, paths
│   └── plots.py                   # figures
├── tests/
├── data/                          # gitignored
├── figures/                       # gitignored
└── pyproject.toml
```

## Tech used

Python 3.10+ · pandas · scikit-learn · matplotlib · requests · pytest · ruff. No ML
framework — the lessons are about the knobs, not about deep learning.

## Interpretation caveats

Stated at the end of script 03 as well, because a tuned model invites over-reading:

1. **Ecological fallacy.** County-level associations say nothing about individual risk.
2. **PLACES values are modeled, not measured.** Outcome and predictors both come from
   small-area estimation over BRFSS with shared demographic and geographic inputs, so
   they partly agree with each other by construction. Apparent fit is inflated.
3. **Nothing here is causal.** This describes covariation — useful for deciding where
   to look, not for claiming what to change.
4. **Crude, not age-adjusted**, by default. Switch `DATA_VALUE_TYPE_ID` in `config.py`
   for between-county comparison. That study-design choice moves the coefficients more
   than any `alpha` will.

## Notes

- Released under the MIT License — see [`LICENSE`](LICENSE).
- CDC PLACES is US federal government work and is in the public domain; cite it as
  CDC, PLACES: Local Data for Better Health, County Data.
