# Vector-Borne Risk Monitoring System

An environmental mosquito-suitability assessment project combining satellite imagery, weather information, image feature extraction, and machine-learning-based environmental risk modelling.

The system includes a data-generation application, dataset preparation utilities, image preprocessing, ResNet-18 feature extraction, Random Forest modelling, multimodal fusion, and a Streamlit visualization dashboard.

> **Repository scope:** This repository contains source code, configuration, and documentation. Large source datasets, generated intermediate datasets, trained model binaries, extracted features, predictions, and generated reports are excluded. They must be obtained or generated separately before executing the complete modelling pipeline.

## 1. Project overview

The project follows a multimodal environmental assessment workflow:

1. Collect or obtain satellite imagery and weather-related information.
2. Prepare and validate environmental datasets.
3. Generate Environmental Risk Index (ERI) and Environmental Mosquito Suitability Score (EMSS) targets.
4. Process satellite images and create consistent dataset partitions.
5. Extract image representations using ResNet-18.
6. Train the first Random Forest model (RF1).
7. Verify alignment between environmental records, image features, and predictions.
8. Train the second Random Forest fusion model (RF2).
9. Visualize supplied environmental suitability records using Streamlit.

See [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) for the detailed architecture and processing flow.

## 2. Project components

| Directory | Responsibility |
|---|---|
| `app/` | Satellite and weather data collection application |
| `config/` | Dataset and image-processing configuration |
| `scripts/data/` | Dataset preparation and environmental target generation |
| `scripts/preprocessing/` | Image processing, splitting, and alignment verification |
| `src/models/` | ResNet-18, RF1, and RF2 modelling |
| `dashboard/` | Streamlit environmental suitability dashboard |
| `docs/` | Dataset, integration, and project documentation |

## 3. Technology stack

- Python 3.11
- PyTorch and Torchvision
- ResNet-18
- Scikit-learn
- Random Forest
- Pandas and NumPy
- OpenCV and Pillow
- FastAPI
- Streamlit
- Plotly and PyDeck
- YAML and JSON configuration

## 4. Repository contents and generated data

The following directories are used by the application and modelling pipeline but are intentionally excluded from the source repository.

| Runtime directory | Purpose |
|---|---|
| `data/raw/` | Original images and input datasets |
| `data/interim/` | Prepared datasets and environmental targets |
| `data/processed/` | Processed images and dataset partitions |
| `artifacts/features/` | Extracted image features |
| `artifacts/models/` | Trained model files |
| `artifacts/predictions/` | Model predictions |
| `reports/` | Generated evaluation and processing reports |

These directories are created or populated by the corresponding scripts when required. Their absence from a fresh clone is intentional.

The dashboard's existing data files are retained under `dashboard/data/` because the dashboard reads them directly.

## 5. Installation

### Requirements

- Windows, Linux, or macOS
- Python 3.11 recommended
- Git
- Internet access for applicable data-provider requests and pretrained model downloads

### Windows setup

Run the following commands from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r dashboard/requirements.txt
```

If PowerShell prevents virtual-environment activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux and macOS setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r dashboard/requirements.txt
```

### Environment configuration

Use `.env.example` as the reference for optional environment settings.

On Windows:

```powershell
Copy-Item .env.example .env
```

On Linux or macOS:

```bash
cp .env.example .env
```

Configure applicable provider credentials and settings when required.

Never commit `.env` files, API keys, credentials, or other secrets.

## 6. Required input data

The modelling pipeline requires source data that is not bundled in this repository.

The default dataset preparation workflow expects the following structure:

```text
data/
└── raw/
    └── bhopal_satellite/
        ├── images/
        └── metadata/
            ├── dataset.csv
            ├── weather.csv
            └── images.csv
```

| Input | Expected location |
|---|---|
| Satellite images | `data/raw/bhopal_satellite/images/` |
| Dataset metadata | `data/raw/bhopal_satellite/metadata/dataset.csv` |
| Weather metadata | `data/raw/bhopal_satellite/metadata/weather.csv` |
| Image metadata | `data/raw/bhopal_satellite/metadata/images.csv` |

The default paths are used by the dataset preparation workflow. Review the relevant script arguments and `config/image_config.yaml` when using a different dataset location or directory structure.

The dataset-generation application provides an interface for generating or obtaining applicable inputs:

```bash
python scripts/data/generate_dataset.py --help
```

This displays the available command-line options. Some operations require network connectivity and provider configuration.

Do not assume that a successful application launch means all required modelling inputs have been generated.

## 7. Complete modelling pipeline

Execute the pipeline from the repository root and preserve the execution order.

### Stage 1 — Generate or obtain source datasets

```bash
python scripts/data/generate_dataset.py --help
```

Use the available options to understand and execute dataset generation.

The required input datasets must be available before proceeding.

### Stage 2 — Prepare the machine-learning dataset

```bash
python scripts/data/prepare_ml_dataset.py
```

This stage prepares the environmental machine-learning dataset from the available input tables.

### Stage 3 — Generate environmental targets

Generate the ERI target:

```bash
python scripts/data/generate_eri_target.py
```

Generate the EMSS target:

```bash
python scripts/data/generate_emss_target.py
```

These stages generate the environmental target information required by subsequent modelling steps.

### Stage 4 — Process images and generate dataset splits

```bash
python scripts/preprocessing/image_pipeline.py split
```

This stage performs the configured image-processing and partitioning workflow.

Review `config/image_config.yaml` before execution. Ensure the configured input and output paths match the available data.

### Stage 5 — Extract image features using ResNet-18

```bash
python src/models/resnet18.py
```

This stage extracts image representations using ResNet-18.

The model produces 512-dimensional feature vectors associated with the corresponding sample identifiers.

Pretrained weights may need to be downloaded during the first execution.

### Stage 6 — Train the RF1 model

```bash
python src/models/rf1.py
```

RF1 performs the environmental-risk modelling stage using the prepared environmental features.

Its generated outputs include the trained model, predictions, feature-importance information, and metrics.

### Stage 7 — Verify multimodal alignment

```bash
python scripts/preprocessing/verify_alignment.py
```

Run this after generating the relevant dataset partitions, ResNet features, and RF1 predictions.

The verification stage checks whether the required sample identifiers and corresponding records are aligned.

Resolve alignment problems before proceeding to RF2.

### Stage 8 — Train the RF2 fusion model

```bash
python src/models/rf2.py
```

RF2 performs multimodal fusion using the available aligned RF1 outputs and ResNet image features.

The script supports:

```bash
python src/models/rf2.py --help
```

The optional `--evaluate-test` argument enables final test evaluation.

Keep the test partition excluded from model selection and parameter-tuning decisions. Use it for final evaluation only.

### Pipeline execution summary

| Order | Script | Purpose |
|---:|---|---|
| 1 | `generate_dataset.py` | Generate or obtain source datasets |
| 2 | `prepare_ml_dataset.py` | Prepare environmental data |
| 3 | `generate_eri_target.py` | Generate ERI |
| 4 | `generate_emss_target.py` | Generate EMSS |
| 5 | `image_pipeline.py split` | Process images and create splits |
| 6 | `resnet18.py` | Extract image features |
| 7 | `rf1.py` | Train RF1 |
| 8 | `verify_alignment.py` | Verify sample alignment |
| 9 | `rf2.py` | Train multimodal fusion model |

> **Important:** These commands document the intended workflow and default execution sequence. A fresh clone is not immediately runnable without the required source datasets and intermediate outputs. Individual scripts may also require configuration adjustments depending on the input data and execution environment.

## 8. Understanding the pipeline outputs

The pipeline uses several runtime directories that are not committed to GitHub.

```text
data/
├── raw/
│   └── bhopal_satellite/
│       ├── images/
│       └── metadata/
├── interim/
│   └── environmental_risk/
└── processed/
    └── multimodal_splits/

artifacts/
├── features/
├── models/
└── predictions/

reports/
├── document_assets/
├── document_render/
├── image_pipeline/
└── model_metrics/
```

The actual output filenames and locations are determined by the relevant scripts and configuration.

Regenerate downstream outputs whenever upstream datasets, image processing settings, or dataset partitions change.

## 9. Running the dashboard

The Streamlit dashboard is a separate, read-only visualization component.

Install its dependencies:

```bash
pip install -r dashboard/requirements.txt
```

Start the dashboard from the repository root:

```bash
streamlit run dashboard/app.py
```

The dashboard reads:

```text
dashboard/data/dashboard_records.csv
dashboard/data/dashboard_metadata.json
```

These records are retained in the repository because the current dashboard depends on them.

The dashboard does not automatically regenerate its records when RF1 or RF2 is retrained.

See [dashboard/README.md](dashboard/README.md) for dashboard features, execution details, data requirements, and interpretation limitations.

## 10. Scientific scope and limitations

- EMSS is a formula-derived environmental suitability proxy.
- The system is intended to support preventive environmental inspection prioritization.
- It does not diagnose dengue or predict confirmed disease outbreaks.
- It does not establish confirmed mosquito presence or breeding-ground detection.
- Stored Bhopal records are historical data, not a live monitoring feed.
- Historical climate observations from Iquitos and San Juan (2000–2008) do not have a real location/date match with the Bhopal image collection (2024–2025).
- Synthetic or simulated climate-image pairings must not be represented as observed joint outcomes or used to make predictive accuracy claims.

Auxiliary research datasets and scripts from earlier experiments are not included in this repository.

## 11. Reproducibility and modelling practices

1. Preserve `sample_id` values across all relevant pipeline stages.
2. Keep train, validation, and test partitions consistent.
3. Regenerate downstream features and predictions when upstream data or splits change.
4. Run alignment verification before RF2 training.
5. Use training data for fitting and validation data for model selection.
6. Reserve the test partition for final evaluation.
7. Keep synthetic demonstration data separate from real-world observations.
8. Record configuration and dependency changes when reproducing experiments.

## 12. Troubleshooting

| Problem | Suggested check |
|---|---|
| Missing input dataset | Verify the required `data/raw/` structure |
| Missing image files | Check the image directory and image configuration |
| Missing intermediate output | Execute the preceding pipeline stage |
| Missing ResNet features | Run the ResNet-18 extraction stage |
| Alignment verification failure | Check sample identifiers and regenerate inconsistent downstream outputs |
| Missing Python module | Activate the correct environment and install the relevant requirements |
| Pretrained model download failure | Check network connectivity and model-cache availability |
| Dashboard cannot load records | Verify the two files under `dashboard/data/` |

Consult the individual scripts and configuration files for stage-specific options.

## 13. Project documentation

Additional documentation is available under `docs/`, including:

- Dataset summaries and data roles
- Project context
- Integration planning
- Workflow verification
- Model and target documentation

## 14. Contributors

This project was developed collaboratively by a six-member team, with different members responsible for different modules.

Individual names, GitHub profiles, and module responsibilities will be documented in `CONTRIBUTORS.md`.

## 15. License

No license has been declared in this repository.

Unless a license is added, the project remains subject to the default copyright rules. Contact the project owners before reusing or redistributing its contents.
