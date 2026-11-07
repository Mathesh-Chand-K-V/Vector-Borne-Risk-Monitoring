# Project Architecture and Workflow

## 1. System overview

The Vector-Borne Risk Monitoring System is organized around a multimodal environmental assessment pipeline.

It combines satellite imagery and weather-related information with image feature extraction and machine-learning models.

The repository also contains a separate Streamlit dashboard for visualizing supplied environmental suitability records.

This document describes the source repository. Generated datasets, trained models, feature files, predictions, and reports are runtime outputs and are not included in the GitHub publishing copy.

## 2. Source repository structure

```text
Vector-Borne-Risk-Monitoring-GitHub/
│
├── app/
│   ├── satellite/
│   │   ├── base.py
│   │   └── provider.py
│   ├── utils/
│   │   ├── geo.py
│   │   ├── images.py
│   │   └── logging.py
│   ├── weather/
│   │   ├── base.py
│   │   └── provider.py
│   ├── web/
│   │   ├── server.py
│   │   └── static/
│   ├── config.py
│   ├── generator.py
│   ├── sampler.py
│   ├── validators.py
│   └── __init__.py
│
├── config/
│   ├── dataset_config.json
│   └── image_config.yaml
│
├── dashboard/
│   ├── data/
│   │   ├── dashboard_metadata.json
│   │   └── dashboard_records.csv
│   ├── app.py
│   ├── README.md
│   ├── requirements.txt
│   └── run_dashboard.bat
│
├── docs/
│   ├── context/
│   └── integration/
│
├── scripts/
│   ├── data/
│   │   ├── generate_dataset.py
│   │   ├── generate_emss_target.py
│   │   ├── generate_eri_target.py
│   │   ├── merge_new_target.py
│   │   └── prepare_ml_dataset.py
│   ├── preprocessing/
│   │   ├── image_pipeline.py
│   │   └── verify_alignment.py
│   └── generate_model_classification_report.py
│
├── src/
│   └── models/
│       ├── resnet18.py
│       ├── rf1.py
│       └── rf2.py
│
├── .env.example
├── .gitignore
├── PROJECT_ARCHITECTURE.md
├── README.md
└── requirements.txt
```

## 3. High-level pipeline architecture

```text
       Satellite imagery + Weather information
                         |
                         v
                Data collection
                         |
                         v
                 Dataset preparation
                         |
                         v
                Environmental dataset
                         |
                         v
                ERI / EMSS generation
                         |
                         v
                 Image preprocessing
                         |
                         v
              Train / Validation / Test
                         |
             +-----------+-----------+
             |                       |
             v                       v
          ResNet-18                  RF1
       Image feature            Environmental
        extraction                modelling
             |                       |
             v                       v
       512-dimensional          RF1 predictions
       image features                 |
             |                       |
             +-----------+-----------+
                         |
                         v
               Alignment verification
                         |
                         v
                    RF2 fusion
                         |
                         v
               Combined model outputs
```

## 4. Component responsibilities

### 4.1 Data collection application

Location: `app/`

Responsible for the dataset-generation application and its supporting components.

Responsibilities include:

- Satellite data provider integration
- Weather data provider integration
- Geographic utilities
- Image utilities
- Dataset generation and validation
- Logging and supporting web application components

### 4.2 Configuration

Location: `config/`

Contains dataset and image-processing configuration.

Important files:

| File | Purpose |
|---|---|
| `dataset_config.json` | Dataset-generation configuration |
| `image_config.yaml` | Image-processing configuration |

Review configured paths before executing the pipeline in another environment.

### 4.3 Dataset preparation and target generation

Location: `scripts/data/`

Responsibilities include:

- Dataset generation
- Machine-learning dataset preparation
- ERI target generation
- EMSS target generation
- Additional target merging

### 4.4 Image preprocessing

Location: `scripts/preprocessing/`

Responsibilities include:

- Image processing
- Image auditing
- Dataset splitting
- Split metadata generation
- Alignment verification

### 4.5 Machine-learning models

Location: `src/models/`

| File | Responsibility |
|---|---|
| `resnet18.py` | Extracts image features using ResNet-18 |
| `rf1.py` | Trains the first Random Forest model |
| `rf2.py` | Performs multimodal fusion modelling |

The ResNet-18 stage produces 512-dimensional image feature vectors associated with sample identifiers.

RF1 performs environmental-risk modelling and produces predictions.

RF2 combines the available aligned RF1 outputs and ResNet image features.

### 4.6 Dashboard

Location: `dashboard/`

The dashboard is a separate read-only visualization layer.

It reads the supplied dashboard records and presents environmental suitability information through Streamlit.

It does not generate environmental targets or train the machine-learning models.

## 5. Required input data

The default dataset preparation workflow expects:

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

These input datasets are not distributed in the source-only repository.

The actual input requirements and available command-line options should be checked in the relevant scripts before execution.

## 6. Generated runtime directories

The following directories are used by the modelling workflow but excluded from the source repository.

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

These directories are created or populated by the corresponding scripts when the required inputs and dependencies are available.

Their absence from a fresh clone is intentional.

## 7. Detailed pipeline execution order

Run the main pipeline from the repository root.

### Stage 1: Source data generation

```bash
python scripts/data/generate_dataset.py --help
```

Inspect the available options and obtain the required source data.

### Stage 2: Dataset preparation

```bash
python scripts/data/prepare_ml_dataset.py
```

Prepare the environmental machine-learning dataset.

### Stage 3: Environmental target generation

```bash
python scripts/data/generate_eri_target.py
python scripts/data/generate_emss_target.py
```

Generate the environmental targets required by the modelling stages.

### Stage 4: Image preprocessing and splitting

```bash
python scripts/preprocessing/image_pipeline.py split
```

Process the image dataset and create the configured partitions.

### Stage 5: Image feature extraction

```bash
python src/models/resnet18.py
```

Extract 512-dimensional ResNet-18 image features.

### Stage 6: RF1 training

```bash
python src/models/rf1.py
```

Train the environmental-risk Random Forest model.

### Stage 7: Alignment verification

```bash
python scripts/preprocessing/verify_alignment.py
```

Verify the relevant sample identifiers and alignment across the pipeline outputs.

### Stage 8: RF2 fusion

```bash
python src/models/rf2.py
```

Train the multimodal fusion model using the aligned outputs.

Use `--evaluate-test` only for final test evaluation after model decisions are complete.

## 8. Data alignment and modelling rules

1. Preserve `sample_id` across all relevant datasets and outputs.
2. Keep train, validation, and test partitions consistent.
3. Regenerate downstream features and predictions whenever upstream inputs or splits change.
4. Verify sample alignment before RF2 training.
5. Use training data for model fitting.
6. Use validation data for model selection.
7. Reserve the test partition for final evaluation.
8. Keep synthetic demonstration data separate from real-world observations.

## 9. Dashboard execution

Install dashboard dependencies:

```bash
pip install -r dashboard/requirements.txt
```

From the repository root:

```bash
streamlit run dashboard/app.py
```

The dashboard reads its retained records from:

```text
dashboard/data/dashboard_records.csv
dashboard/data/dashboard_metadata.json
```

It is not automatically updated by retraining RF1 or RF2.

## 10. Scientific limitations

The project uses historical Bhopal-area records and environmental suitability modelling.

EMSS is a formula-derived environmental proxy. It is not a diagnosis, confirmed outbreak prediction, confirmed mosquito-presence result, or confirmed breeding-ground detection system.

Historical climate observations from Iquitos and San Juan (2000–2008) do not have a real geographic or temporal match with the Bhopal image collection (2024–2025).

Earlier synthetic and simulated multimodal experiments are not included in this source repository. Their outputs must not be represented as observed climate-image relationships or used to make predictive accuracy claims.

## 11. Repository maintenance

- Keep source code, configuration, and essential documentation under version control.
- Keep secrets and local environment files out of version control.
- Do not commit generated datasets, model binaries, temporary files, or reproducible reports unless there is a documented reason.
- Record each contributor's actual responsibilities in `CONTRIBUTORS.md`.
- Preserve the original project history separately when publishing a clean collaborative repository.
