# Knowledge Tracing Model Comparison on EDNet-KT1

Research code, experimental outputs, dissertation material, and a deployed adaptive-learning prototype for the MSc Data Science dissertation:

> **A Comparative Study of Knowledge Tracing Models: Failure Modes and Cross-Domain Transfer with EDNet-KT1**

[Open the live SFLA adaptive learning platform](https://sfla-adaptive-learning.streamlit.app/)

## Project overview

This project compares three knowledge tracing approaches:

- Bayesian Knowledge Tracing (BKT)
- Deep Knowledge Tracing (DKT)
- causal Self-Attentive Knowledge Tracing (SAKT)

The evaluation goes beyond a single headline metric. It examines discrimination, calibration, learner-level uncertainty, model disagreement, sequence-length sensitivity, temporal leakage, and performance across learner activity, skill frequency, and available-history strata.

The repository also contains a cross-domain configurability case study for the University of the West of England module **Sets, Functions and Linear Algebra (SFLA)**. The case study is presented through a Streamlit proof-of-concept platform that uses BKT for adaptive question selection and exposes DKT and causal SAKT predictions only for research comparison.

## Repository contents

| Path | Contents |
|---|---|
| [`notebooks/`](notebooks/) | Initial pipeline work and the main comparative modelling workflow |
| [`adaptive_learning_platform/`](adaptive_learning_platform/) | Streamlit application, BKT engine, recommendation logic, database layer, tests, and evaluation scripts |
| [`data/platform/`](data/platform/) | The 27-question SFLA prototype question bank |
| [`data/processed/`](data/processed/) | The SFLA item-to-knowledge-component register |
| [`models/sfla/`](models/sfla/) | Exported BKT parameters, DKT and SAKT SavedModels, and model metadata |
| [`results/tables/`](results/tables/) | Aggregate metrics, calibration, disagreement, ablation, and failure-mode tables |
| [`results/figures/`](results/figures/) | Figures generated for the model comparison and diagnostic analyses |
| [`results/predictions/`](results/predictions/) | Saved predictions and common evaluation targets |
| [`results/cross_domain/sfla/`](results/cross_domain/sfla/) | Frozen SFLA configuration, simulated interactions, mappings, and configurability evidence |
| [`results/adaptive_platform/`](results/adaptive_platform/) | Platform functional scorecard and machine-readable summary |
| [`docs/`](docs/) | Proposal, dissertation chapters, literature review, and ethics material |
| [`meeting_minutes/`](meeting_minutes/) | Supervisor meeting records |

## Run the adaptive learning platform

The platform can be run without the raw EDNet-KT1 dataset because its question bank and exported SFLA model artefacts are committed to the repository. Python 3.12 is recommended.

Clone the repository, then create a virtual environment:

```bash
git clone https://github.com/tolulopesoetan/msc_kt_ednet.git
cd msc_kt_ednet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r adaptive_learning_platform/requirements.txt
python -m streamlit run adaptive_learning_platform/app.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The local Streamlit address is normally `http://localhost:8501`.

For application behaviour, configuration, exports, privacy constraints, and deployment guidance, see the [adaptive learning platform README](adaptive_learning_platform/README.md).

## Run the checks

Install the platform dependencies, then run:

```bash
python -m pytest adaptive_learning_platform/tests
python adaptive_learning_platform/scripts/run_platform_evaluation.py
```

The unit tests cover BKT updates, question-bank validation, persistence, learner-scoped exports, session isolation, and recommendation behaviour. The functional evaluation checks the integrated question bank, parameters, adaptive rules, persistence, and export queries.

The committed [functional summary](results/adaptive_platform/platform_functional_summary.json) records **11 passed checks, 0 warnings, and 0 failures**. Running the evaluation script refreshes that summary and the associated [scorecard](results/adaptive_platform/platform_functional_scorecard.csv).

To regenerate the committed 27-question prototype bank:

```bash
python adaptive_learning_platform/scripts/create_question_bank.py
```

This command overwrites `data/platform/sfla_question_bank.csv` after validating coverage of all nine knowledge components and all three difficulty levels.

## Work with the research notebooks

The notebooks are research records rather than a packaged training command. Run them from the repository root or from `notebooks/` after supplying the required source data.

The principal recorded environment is Python 3.12.13 with:

```text
numpy==1.26.4
pandas==2.2.3
scipy==1.13.1
scikit-learn==1.6.1
pyBKT==1.4.2
tensorflow==2.21.0
```

The notebooks also use Jupyter, Matplotlib, Seaborn, PyYAML, and the Kaggle API where indicated. The authoritative recorded versions are available in [`results/tables/software_environment.csv`](results/tables/software_environment.csv).

### Data notice

The raw educational datasets are not committed. Keep downloaded source data under `data/raw/` and do not add it to version control. The initial EDNet pipeline expects user interaction files below `data/raw/` and question metadata at `data/contents/questions.csv`; the notebook data-access cells provide the final path checks for each workflow.

Raw files, Streamlit secrets, local databases, caches, and virtual environments are excluded through [`.gitignore`](.gitignore).

## Reproducibility and evidence

- Learners are isolated across model partitions to reduce identity leakage.
- DKT and SAKT use next-response prediction contracts, and SAKT applies causal masking.
- Neural-model results are aggregated across seeds 42, 43, and 44.
- Learner-level bootstrap intervals are retained with the evaluation tables.
- A separate SAKT leakage diagnostic demonstrates the effect of exposing future information.
- Model predictions, thresholds, training histories, calibration outputs, and disagreement tables are committed under `results/`.
- The SFLA cross-domain configuration and its scope boundaries are frozen under `results/cross_domain/sfla/`.

## Important research boundaries

- The SFLA learner interactions and fitted SFLA model parameters are **simulated**.
- The SFLA platform demonstrates technical integration and configurability; it does not establish target-domain predictive validity or improved learning outcomes.
- BKT is the only model that controls adaptation in the platform. DKT and causal SAKT are optional research displays and never control recommendations.
- The 27 SFLA questions are original prototype practice questions, not an official assessment bank.
- The hosted demonstration uses local SQLite storage, which may be reset when the service restarts or is redeployed.
- Only pseudonymous learner identifiers should be entered into the platform.


## Author

Tolulope David Soetan

MSc Data Science, University of the West of England (UWE Bristol)
