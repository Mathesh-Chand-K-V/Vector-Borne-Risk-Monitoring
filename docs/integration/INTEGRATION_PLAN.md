# Integration Plan

## Recommended structure

```text
Project_Integration/
  docs/                         # architecture, data roles, setup, exhibition story
  integration_reports/          # inventories, duplicate report, merge manifest
  auxiliary_research/
    climate_lag_experiment/     # separate Iquitos/San Juan baseline
    mosquito_image_experiment/ # separate breeding-image experiment
  dashboard/                    # read-only Bhopal prediction display
  main_workflow/                # Bhopal aligned RF1 + ResNet-18 + RF2 components
```

`main_workflow/` should reference or later copy only the aligned Bhopal components needed for RF1, ResNet-18 and RF2. Keep climate-lag and MosquitoFusion/YOLO experiments in their separate auxiliary directories. The dashboard must consume only approved Bhopal main-model output. The current FastAPI app is a dataset generator, not a prediction API. No files or data were copied during this audit.

## Priority gates

1. Resolve provenance, map bounds and the `config/dataset_config.json` 5000 km radius versus the web default 50 km.
2. Review ERI/EMSS thresholds and units. Since the target is derived from environmental RF1 inputs, do not describe current metrics as disease-incidence prediction without an independent observed outcome.
3. Freeze source hashes, target code, feature schema and split IDs. Confirm group-aware isolation, test-set closure and stored RF1 OOF provenance.
4. Account for the two omitted climate city-weeks (`sj|2008|17`, `iq|2010|25`) and five orphan MosquitoFusion test annotation files.
5. Run `verify_alignment.py` after controlled setup and review; it was not run during this audit. Require a pass before RF2.
6. Reproduce artifacts only after approval. Define a read-only Bhopal API contract before adapting a dashboard; do not expose data-generation endpoints.

## File-by-file later-copy review

All listed items require manual review. “Safe later” is conditional, not permission to copy during this audit. Path/dependency notes are included per item and the merge-risk field is in the CSV manifest.

| Source path | Proposed destination | Purpose | Manual review | Safe to copy later? | Dependency/path update needed |
|---|---|---|---|---|---|
| `Vector_Borne_Risk_Project/src/models/rf1.py` | `main_workflow/src/models/rf1.py` | Environmental Random Forest; grouped out-of-fold train predictions feed fusion. | Yes | Only after target/code/path review; not copied in audit | Manual review required; update project-root paths and install only pinned workflow dependencies. Update project-root assumptions and install pinned workflow dependencies. |
| `Vector_Borne_Risk_Project/src/models/resnet18.py` | `main_workflow/src/models/resnet18.py` | Frozen pretrained satellite feature extraction to 512 dimensions by sample_id. | Yes | Only after target/code/path review; not copied in audit | Manual review required; update project-root paths and install only pinned workflow dependencies. Update project-root assumptions and install pinned workflow dependencies. |
| `Vector_Borne_Risk_Project/src/models/rf2.py` | `main_workflow/src/models/rf2.py` | Fusion Random Forest on aligned RF1 output and ResNet features. | Yes | Only after target/code/path review; not copied in audit | Manual review required; update project-root paths and install only pinned workflow dependencies. Update project-root assumptions and install pinned workflow dependencies. |
| `Vector_Borne_Risk_Project/scripts/data/prepare_ml_dataset.py` | `main_workflow/scripts/data/prepare_ml_dataset.py` | Prepare the Bhopal environmental master table. | Yes | Only after target/code/path review; not copied in audit | Manual review required; update project-root paths and install only pinned workflow dependencies. Update project-root assumptions and install pinned workflow dependencies. |
| `Vector_Borne_Risk_Project/scripts/data/generate_emss_target.py` | `main_workflow/scripts/data/generate_emss_target.py` | Generate current EMSS proxy target; scientific review required. | Yes | Only after target/code/path review; not copied in audit | Manual review required; update project-root paths and install only pinned workflow dependencies. Update project-root assumptions and install pinned workflow dependencies. |
| `Vector_Borne_Risk_Project/scripts/preprocessing/image_pipeline.py` | `main_workflow/scripts/preprocessing/image_pipeline.py` | Audit and split Bhopal imagery with configured preprocessing. | Yes | Only after target/code/path review; not copied in audit | Manual review required; update project-root paths and install only pinned workflow dependencies. Update project-root assumptions and install pinned workflow dependencies. |
| `Vector_Borne_Risk_Project/scripts/preprocessing/verify_alignment.py` | `main_workflow/scripts/preprocessing/verify_alignment.py` | Gate fusion on sample_id, row-count, feature-shape and null checks. | Yes | Only after target/code/path review; not copied in audit | Manual review required; update project-root paths and install only pinned workflow dependencies. Update project-root assumptions and install pinned workflow dependencies. |
| `Vector_Borne_Risk_Project/config/image_config.yaml` | `main_workflow/config/image_config.yaml` | Image size, split fractions, columns and paths. | Yes | No until explicit review/approval; not copied in audit | Prune packages and update paths/settings; verify pinned versions. |
| `Vector_Borne_Risk_Project/config/dataset_config.json` | `main_workflow/config/dataset_config.json` | Bhopal sampling config; reconcile the 5000 km radius with the 50 km web default. | Yes | No until explicit review/approval; not copied in audit | Prune packages and update paths/settings; verify pinned versions. |
| `Vector_Borne_Risk_Project/requirements.txt` | `main_workflow/requirements.txt` | Base dependency source for a pruned reproducible environment. | Yes | No until explicit review/approval; not copied in audit | Prune packages and update paths/settings; verify pinned versions. |
| `Vector_Borne_Risk_Project/data/raw/bhopal_satellite/metadata/dataset.csv` | `main_workflow/data/raw/bhopal_satellite/metadata/dataset.csv` | Bhopal dataset metadata and sample_id provenance. | Yes | No until explicit review/approval; not copied in audit | Verify provenance, target, IDs, license and path; no data copied during audit. |
| `Vector_Borne_Risk_Project/data/raw/bhopal_satellite/metadata/images.csv` | `main_workflow/data/raw/bhopal_satellite/metadata/images.csv` | Image path and sample_id index. | Yes | No until explicit review/approval; not copied in audit | Verify provenance, target, IDs, license and path; no data copied during audit. |
| `Vector_Borne_Risk_Project/data/raw/bhopal_satellite/metadata/weather.csv` | `main_workflow/data/raw/bhopal_satellite/metadata/weather.csv` | Weather records associated with Bhopal sample IDs. | Yes | No until explicit review/approval; not copied in audit | Verify provenance, target, IDs, license and path; no data copied during audit. |
| `Vector_Borne_Risk_Project/data/processed/multimodal_splits/train.csv` | `main_workflow/data/processed/multimodal_splits/train.csv` | Frozen aligned training partition. | Yes | No until explicit review/approval; not copied in audit | Verify provenance, target, IDs, license and path; no data copied during audit. |
| `Vector_Borne_Risk_Project/data/processed/multimodal_splits/validation.csv` | `main_workflow/data/processed/multimodal_splits/validation.csv` | Frozen aligned validation partition. | Yes | No until explicit review/approval; not copied in audit | Verify provenance, target, IDs, license and path; no data copied during audit. |
| `Vector_Borne_Risk_Project/data/processed/multimodal_splits/test.csv` | `main_workflow/data/processed/multimodal_splits/test.csv` | Held-out aligned partition for final evaluation. | Yes | No until explicit review/approval; not copied in audit | Verify provenance, target, IDs, license and path; no data copied during audit. |
| `Vector_Borne_Risk_Project/data/raw/bhopal_satellite/images/` | `external reference preferred (or main_workflow/data/raw/bhopal_satellite/images/)` | Bhopal satellite imagery for main workflow. | Yes | No until explicit review/approval; not copied in audit | Large binary data; review storage, license, checksums and paths. |
| `Vector_Borne_Risk_Project/artifacts/models/rf1_model.joblib` | `main_workflow/artifacts/models/rf1_model.joblib` | Potential frozen model after target and training provenance approval. | Yes | No until explicit review/approval; not copied in audit | Binary artifact not validated; reproduce from locked data before relying on it. |
| `Vector_Borne_Risk_Project/artifacts/models/rf2_model.joblib` | `main_workflow/artifacts/models/rf2_model.joblib` | Potential frozen model after target and training provenance approval. | Yes | No until explicit review/approval; not copied in audit | Binary artifact not validated; reproduce from locked data before relying on it. |
| `Vector_Borne_Risk_Project/artifacts/models/rf2_feature_columns.json` | `main_workflow/artifacts/models/rf2_feature_columns.json` | Record exact RF2 input features with a validated model. | Yes | No until explicit review/approval; not copied in audit | Verify against final model and schema. |
| `Coding_Part/Tabular processing/final_climate_lag_dataset.csv` | `auxiliary_research/climate_lag_experiment/data/final_climate_lag_dataset.csv` | Historical Iquitos/San Juan climate-lag baseline. | Yes | Only after provenance/split/license/path review; not copied in audit | Review provenance; distinct case target and no Bhopal join. |
| `Coding_Part/data/climate_train.csv` | `auxiliary_research/climate_lag_experiment/data/splits/climate_train.csv` | Climate-lag train partition. | Yes | Only after provenance/split/license/path review; not copied in audit | Resolve two city-week omissions from the 1,178-row master; verify chronology and disjointness. |
| `Coding_Part/data/climate_val.csv` | `auxiliary_research/climate_lag_experiment/data/splits/climate_val.csv` | Climate-lag val partition. | Yes | Only after provenance/split/license/path review; not copied in audit | Resolve two city-week omissions from the 1,178-row master; verify chronology and disjointness. |
| `Coding_Part/data/climate_test.csv` | `auxiliary_research/climate_lag_experiment/data/splits/climate_test.csv` | Climate-lag test partition. | Yes | Only after provenance/split/license/path review; not copied in audit | Resolve two city-week omissions from the 1,178-row master; verify chronology and disjointness. |
| `Coding_Part/Tabular processing/01_dataset_cleaning.ipynb` | `auxiliary_research/climate_lag_experiment/notebooks/01_dataset_cleaning.ipynb` | Climate-lag cleaning notebook. | Yes | Only after provenance/split/license/path review; not copied in audit | Review paths and preserve target distinction; notebook not executed. |
| `Coding_Part/data/data_dictionary.md` | `auxiliary_research/climate_lag_experiment/docs/data_dictionary.md` | Climate feature and target definitions. | Yes | Only after provenance/split/license/path review; not copied in audit | Review paths and preserve target distinction; notebook not executed. |
| `Coding_Part/DATASET_SUMMARY.md` | `auxiliary_research/climate_lag_experiment/docs/DATASET_SUMMARY.md` | Dataset roles and split description. | Yes | Only after provenance/split/license/path review; not copied in audit | Review paths and preserve target distinction; notebook not executed. |
| `Coding_Part/Image processing/Res-18 Image processing/MosquitoFusion Dataset/` | `external reference preferred (or auxiliary_research/mosquito_image_experiment/data/)` | Separate breeding/non-breeding image experiment. | Yes | Only after provenance/split/license/path review; not copied in audit | Review license, labels, duplicate leakage and mapping; no Bhopal sample/time/location link. |
| `Coding_Part/data/yolo_train_split.csv` | `auxiliary_research/mosquito_image_experiment/metadata/yolo_train_split.csv` | Breeding/non-breeding train IDs and labels. | Yes | Only after provenance/split/license/path review; not copied in audit | Verify actual image/path pairing and label definitions; do not join to Bhopal. |
| `Coding_Part/data/yolo_val_split.csv` | `auxiliary_research/mosquito_image_experiment/metadata/yolo_val_split.csv` | Breeding/non-breeding val IDs and labels. | Yes | Only after provenance/split/license/path review; not copied in audit | Verify actual image/path pairing and label definitions; do not join to Bhopal. |
| `Coding_Part/data/yolo_test_split.csv` | `auxiliary_research/mosquito_image_experiment/metadata/yolo_test_split.csv` | Breeding/non-breeding test IDs and labels. | Yes | Only after provenance/split/license/path review; not copied in audit | Verify actual image/path pairing and label definitions; do not join to Bhopal. |
| `Coding_Part/data/yolo_class_mapping.txt` | `auxiliary_research/mosquito_image_experiment/metadata/yolo_class_mapping.txt` | Provisional YOLO class mapping. | Yes | Only after provenance/split/license/path review; not copied in audit | Verify mapping against dataset source. |
| `Vector_Borne_Risk_Project/app/web/static/index.html` | `dashboard/index.html` | Potential UI starting point; current app controls data generation. | Yes | No until explicit review/approval; not copied in audit | Remove generation controls; connect only to read-only approved Bhopal outputs. Replace generation routes with read-only Bhopal output interface. |
| `Vector_Borne_Risk_Project/app/web/static/app.js` | `dashboard/app.js` | Potential UI starting point; current app controls data generation. | Yes | No until explicit review/approval; not copied in audit | Remove generation controls; connect only to read-only approved Bhopal outputs. Replace generation routes with read-only Bhopal output interface. |
| `Vector_Borne_Risk_Project/app/web/static/styles.css` | `dashboard/styles.css` | Potential UI starting point; current app controls data generation. | Yes | No until explicit review/approval; not copied in audit | Remove generation controls; connect only to read-only approved Bhopal outputs. Replace generation routes with read-only Bhopal output interface. |
| `Vector_Borne_Risk_Project/app/web/server.py` | `dashboard/api/ (new read-only service; do not copy this generator API)` | Current API controls dataset collection and is not model inference. | Yes | No until explicit review/approval; not copied in audit | Define versioned Bhopal output schema, safe paths and failure states. Replace generation routes with read-only Bhopal output interface. |

## Destination rules

- `main_workflow/`: minimum Bhopal aligned code; external-reference large image data and avoid copying generated artifacts by default.
- `auxiliary_research/climate_lag_experiment/`: climate-lag baseline only.
- `auxiliary_research/mosquito_image_experiment/`: MosquitoFusion/YOLO and breeding-image work only.
- `dashboard/`: adapt compatible frontend components only after defining an approved Bhopal output contract.
- `docs/`: architecture summary, dataset roles, setup instructions, provenance/limitations and exhibition explanation.

This is a plan only. Do not modify, copy, move, merge or delete files in either original project. No training, downloads, or artifact regeneration were performed.
