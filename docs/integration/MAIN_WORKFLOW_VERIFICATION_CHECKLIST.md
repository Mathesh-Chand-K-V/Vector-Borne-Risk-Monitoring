# Main Workflow Verification Checklist

## Executive Summary

**Decision: NOT READY.** The stored Bhopal tables and model inputs appear structurally complete for an alignment-gate run: 4,745 samples split into 3,321/712/712, with unique and non-overlapping sample IDs, resolvable image paths, and matching RF1/ResNet output IDs and row counts. The split report also records zero sample-ID and spatial-group overlap.

That is not sufficient for official exhibition readiness. The configured 5,000 km radius conflicts with 50 km generator defaults and the stored manifest; the final EMSS target is a formula built from predictors RF1 also receives; model artifacts conflict with README statements about RF2 execution and lack run provenance; no read-only prediction API/dashboard exists; and external-source licensing/attribution still needs confirmation. The alignment script was inspected but deliberately not run.

**Status totals (39 checks):** 20 PASS, 7 FAIL, 4 UNKNOWN, 8 MANUAL_CONFIRMATION_REQUIRED. Counts are calculated from `integration_reports/main_workflow_readiness.csv`.

## 1. Bhopal Data Provenance

### Stated providers and attribution

- Satellite imagery: `Sentinel2PlanetaryProvider` identifies Copernicus Sentinel-2 L2A through Microsoft Planetary Computer. The code queries the Planetary Computer STAC search endpoint and renders RGB imagery through its TiTiler item/bbox endpoint. Stored metadata names `Copernicus Sentinel-2 L2A (Microsoft Planetary Computer)` and the image rows have `satellite` and `source` fields.
- Weather/environmental data: `OpenMeteoHistoricalProvider` identifies the Open-Meteo Historical Weather Archive and requests daily fields from ECMWF ERA5/ERA5-Land. Stored weather rows have a source field.
- URLs in source: `https://planetarycomputer.microsoft.com/api/stac/v1/search`, `https://planetarycomputer.microsoft.com/api/data/v1/item/bbox`, and `https://archive-api.open-meteo.com/v1/archive`. The satellite search uses collection `sentinel-2-l2a`. Nominatim is called by the generator only if center coordinates are absent.
- The inspected project files do not establish an image-source citation record, dataset-level license, or complete attribution text. Open-access comments and provider names are not a substitute for confirming applicable terms and citations.

### Stored bounds and Bhopal-area evidence

`data/raw/bhopal_satellite/metadata/dataset_manifest.json` records location name Bhopal, India; center latitude 23.2584857 and longitude 77.401989; radius 50 km; recorded source names; date window 2024-01-01 through 2025-06-30; WGS 84 / EPSG:4326; creation timestamp 2026-09-24; and software version 1.1.0. The quality report describes 4,745 rows, latitude 22.809708..23.705719, longitude 76.913320..77.890032, and requested-date range 2024-01-01..2025-06-19. A metadata-only haversine check found all stored coordinates within 49.99 km of the manifest center. The observed `actual_acquisition_date` range is 2024-01-01..2025-06-16.

These records support the claim that the *stored coordinates and provider metadata* describe a Bhopal-area sample. They do not independently authenticate that every downloaded image corresponds to its recorded coordinate/date or establish source/license chain of custody. Human confirmation is required before making that exhibition claim.

### Radius/search settings and discrepancy

| Location | Setting/evidence | Audit interpretation |
|---|---|---|
| `config/dataset_config.json` | `location=Bhopal, India`, center 23.2599/77.4126, `radius_km=5000.0`, requested target 1,500, date end 2025-12-31, cloud max 20, image size 256. | The 5,000 km value conflicts with the saved 50 km manifest and exceeds the validated generator maximum. |
| `app/config.py` | `GeneratorConfig.radius_km` defaults to 50 and constrains the accepted range to 1..500 km; default center is 23.2599/77.4126. | A 5,000 km config cannot be instantiated through this Pydantic model as written. |
| `app/web/server.py` | `StartRequest.radius_km=50`; location text defaults to Bhopal; date, cloud and image defaults are 2024-01-01..2025-12-31, 20%, 256. | `/api/start` passes the radius to `GeneratorConfig`, so the model limit applies. It does not pass explicit center coordinates; the model's fixed default center remains in use. |
| `app/web/static/index.html` and `app.js` | UI radius field defaults to 50 and caps at 500; it posts location text and radius. | UI also starts at 50 km. Editing the location text alone does not supply coordinates to the API. |
| `scripts/data/generate_dataset.py` | CLI radius defaults to 50 km; accepts explicit center latitude/longitude, target count, dates, cloud max and image size. | This is another plausible generation path, but its defaults also differ from the saved manifest's target 5,000, cloud 50%, image size 512 and end date 2025-06-30. |
| `app/sampler.py` / `app/utils/geo.py` | Candidate points are sampled inside a circle using the supplied radius; haversine distance is checked. The image patch bbox is independently calculated from pixel size and metres-per-pixel. | Spatial search radius is not the same as the satellite patch footprint. |
| `app/validators.py` | Recomputes haversine distance to manifest center; allows a 0.5 km tolerance beyond manifest radius and records a geographic bounding box. | This is the generator's validation rule. It was not executed during this audit; the metadata coordinate calculation was independent. |
| `app/satellite/provider.py` | STAC point query uses a bbox of ±0.01° around each candidate coordinate (commented as about 1 km); downloaded RGB patch uses the image-pixel bbox. | These are per-sample search/render bounds, not the overall Bhopal collection radius. |
| `app/utils/geo.py` plus stored image size | The render bbox uses 10 m/pixel; the stored manifest says 512x512, a nominal 5.12 km square patch. The helper doc's 2.56 km example describes 256x256. | Patch footprint is not the circle used to select sample centers. |
| `prepare_ml_dataset.py` / `image_pipeline.py` | The report/preparation spatial grid defaults to 0.05°; the split code assigns each such grid cell to one partition. | This is a grouping/leakage-control grid, not a radius or provider query bbox. |
| Stored metadata | `dataset_manifest.json` says 50 km; computed maximum distance is 49.99 km. | Strongly consistent with a 50 km collection. Metadata alone cannot prove which code/config revision and run produced it. |

**Radius finding: FAIL / MANUAL CONFIRMATION REQUIRED.** The most plausible explanation for the saved rows is a 50 km run, since the manifest records 50 and every recorded point is within that circle. The manifest center (23.2584857, 77.401989) differs from the fixed `GeneratorConfig` default (23.2599, 77.4126); the CLI can geocode the Bhopal text when coordinates are passed as `None`, while the web request does not pass center coordinates and therefore uses fixed defaults. The 5,000 km config was not valid under the current `GeneratorConfig` range and the code path consuming that JSON is not established. Other saved manifest settings differ from current defaults, so the exact historical generation path and settings cannot be confirmed from metadata alone. Before exhibition, the Data Team must reconcile the historical run log/config, confirm which center was used and retain the corresponding provider item/source records.

Statuses: provider names/source fields are present; external source authenticity, applicable license and attribution require manual confirmation. Stored coordinates support Bhopal-area sampling; the 5,000/50 km configuration conflict fails readiness.

## 2. ERI / EMSS Target Review

Two scripts exist; `generate_emss_target.py` reads the ERI output and replaces `ERI_target` with EMSS while keeping the historical column name for pipeline compatibility.

### Legacy ERI calculation

`generate_eri_target.py` uses `temperature_c`, `humidity`, `precipitation`, and `wind_speed`, with weights 0.35, 0.30, 0.25, and 0.10. It coerces values to numeric, median-imputes missing values, min-max normalizes each variable over the full input table, then sums the weighted normalized values. This produces a nominal 0..1 `ERI_target`; `ERI_percentile` is its rank percentile on a 0..100 scale. The code does not clip the weighted result separately; with nonnegative weights summing to one and normalized components in [0,1], its theoretical range is [0,1].

### Final EMSS calculation

`generate_emss_target.py` uses only `temperature_c`, `humidity`, and `precipitation`; weights are 0.40, 0.30, and 0.30. Inputs are coerced to numeric and median-imputed. Suitability components are clipped to [0,1]:

- Temperature: trapezoid with zero below 10 C, linear rise to 1 from 10 to 20 C, plateau through 28 C, linear fall to zero at 40 C, and zero above 40 C.
- Humidity: `clip((humidity - 30) / (85 - 30), 0, 1)`.
- Precipitation: trapezoid with knots 0, 5, 25, and 60 in the dataset's `precipitation` units; values above 60 are set to a 0.35 floor. The script warns that these thresholds must match the actual precipitation units.
- Final `ERI_target` column: `100 * (0.40 * temperature_suitability + 0.30 * humidity_suitability + 0.30 * precipitation_suitability)`, nominal scale 0..100. `ERI_percentile` is an additional rank percentile on 0..100.

This is a formula/rule-derived environmental suitability proxy. It is not observed mosquito abundance, observed breeding sites, or a dengue-case outcome; it is not itself synthetic or simulated. Separate auxiliary synthetic/simulated files exist and must not be confused with this Bhopal target.

RF1 explicitly uses `temperature_c`, `humidity`, and `precipitation` (as well as related environmental fields) among its 17 predictors. Therefore RF1 can learn to reconstruct the formula that generated its target. High R2 values in the stored RF1/RF2 metrics are not evidence of independent disease forecasting or confirmed breeding-ground detection.

**Recommended exhibition wording:** “Environmental Mosquito Suitability Score” (formula-derived environmental proxy). “Potential Environmental Breeding-Risk Score” is an acceptable alternative only with the same caveat. Do not say “confirmed dengue prediction,” “disease diagnosis,” or “confirmed breeding-ground detection.” A Project Lead/domain expert must approve the wording and scientific assumptions.

## 3. Splits, Sample IDs, and Leakage Controls

| Split | Rows | Required ID | Unique IDs | Cross-split ID overlap | `image_path` exists under configured root | Spatial groups |
|---|---:|---|---:|---:|---|---:|
| Train | 3,321 | Present | 3,321 | 0 | Yes, all checked paths resolve | 187 |
| Validation | 712 | Present | 712 | 0 | Yes, all checked paths resolve | 59 |
| Test | 712 | Present | 712 | 0 | Yes, all checked paths resolve | 59 |

The split CSVs total 4,745 rows. `reports/image_pipeline/split_report.txt` records 305 total 0.05-degree grid groups and zero sample-ID and spatial-group overlap. The saved `spatial_group` values were also checked across all three current CSVs: no group overlap.

`image_pipeline.py` allocates complete spatial grid groups to one partition and retains `spatial_group` in output. `rf1.py` uses `GroupKFold(n_splits=5)` over `spatial_group` to create out-of-fold predictions for RF1 training rows; validation/test predictions come from the final RF1 fit on training data. This is leakage-aware in design, but does not fix the target's dependence on its own RF1 predictor inputs.

RF1 uses an explicit 17-column allowlist: latitude, longitude, cloud coverage, temperature/apparent/min/max temperature, humidity, pressure, dew point, wind speed/direction/gusts, precipitation, rain, cloud cover, and weather code. Its explicit exclusion set names `ERI_target`, `ERI_percentile`, `sample_id`, `image_path`, hashes, image/audit fields, `spatial_group`, and `date`. `requested_date`, `actual_acquisition_date`, `date_difference_days`, and `image_path_audit` are not RF1 inputs because they are absent from the allowlist, even though not all are named in the exclusion set.

**Conclusion:** split and ID leakage controls are PASS in the stored split report and metadata checks. The RF1 design excludes direct target/ID/path/audit columns and uses spatial OOF, but the shared target-source environmental variables are a critical scientific validity limitation. No model-training code was executed for this audit.

## 4. Artifact and Model Status

Binary model files were checked for existence and byte size only; neither was unpickled or executed. CSVs were streamed for headers, row counts, ID uniqueness, blank fields, and ID-set comparison.

| Artifact | Status and metadata |
|---|---|
| `artifacts/models/rf1_model.joblib` | Exists; 79,332,490 bytes. Binary contents not inspected. |
| `artifacts/models/rf1_feature_importance.csv` | Exists; 614 bytes; 17 rows x 2 columns (`feature`, `importance`); zero blank rows. |
| `artifacts/models/rf2_model.joblib` | Exists; 83,857,937 bytes. Binary contents not inspected. README conflicts with this file and the prediction/metric outputs by stating RF2 training has not yet run in this copy. Manual provenance reconciliation required. |
| `artifacts/models/rf2_feature_columns.json` | Exists; 6,168 bytes; records 513 RF2 inputs (one RF1 prediction plus 512 ResNet features). |
| `artifacts/predictions/rf1_train_predictions.csv` | Exists; 3,321 rows x 3 columns (`sample_id`, `ERI_target`, `rf1_prediction`); unique IDs; zero blank cells; ID set matches train split. |
| `artifacts/predictions/rf1_validation_predictions.csv` | Exists; 712 x 3; unique IDs; zero blank cells; ID set matches validation split. |
| `artifacts/predictions/rf1_test_predictions.csv` | Exists; 712 x 3; unique IDs; zero blank cells; ID set matches test split. |
| `artifacts/features/resnet_train_features.csv` | Exists; 3,321 x 513 (`sample_id` + 512 features); unique IDs; zero blank cells; ID set matches train split. |
| `artifacts/features/resnet_validation_features.csv` | Exists; 712 x 513; unique IDs; zero blank cells; ID set matches validation split. |
| `artifacts/features/resnet_test_features.csv` | Exists; 712 x 513; unique IDs; zero blank cells; ID set matches test split. |
| `artifacts/predictions/rf2_validation_predictions.csv` | Exists; 712 x 3; unique IDs; zero blank cells. |
| `artifacts/predictions/rf2_test_predictions.csv` | Exists; 712 x 3; unique IDs; zero blank cells. |
| `reports/model_metrics/rf1_metrics.csv` | Exists; 3 metric rows x 4 columns (`split`, `MAE`, `RMSE`, `R2`); includes train OOF, validation, test. |
| `reports/rf2_metrics.csv` | Exists; 2 rows x 5 columns; includes validation and test scores. This is evidence an RF2 metric output was created at some point, not proof it corresponds to the current sources. |

Stored metrics are numerically very high (RF1 test R2 about 0.997; RF2 test R2 about 0.994), but the target is derived from RF1's own environmental inputs. They must not be described as evidence of disease prediction.

**Execution evidence:** files demonstrate that RF1 prediction tables, ResNet feature tables, model binaries and RF2 validation/test outputs exist and have compatible metadata. They strongly suggest those stages were run at some point. They do not establish code revision, input hashes, exact target/split versions or reproducibility. The README's RF2 note conflicts with current RF2 artifacts/metrics. The image-pipeline split report proves a spatial split/check was recorded. No saved report proves `verify_alignment.py` itself was executed. No training, feature extraction, model loading, or alignment-gate run was performed in this audit.

### Missing experiment provenance

- Exact Python interpreter version and fully pinned package versions (requirements use minimum-version constraints).
- Hardware/OS/runtime environment for RF1, ResNet extraction, RF2 and alignment.
- Training/feature-extraction dates and run IDs linked to the artifacts.
- Source/data hashes and code revision associated with each model and output.
- Versioned target formula identifier and confirmation of which ERI/EMSS script version produced saved `ERI_target` values.
- Split manifest/version hash and exact sample-ID membership linked to model outputs.
- RF1 feature schema snapshot tied to the model; RF2 feature-column JSON exists but is not tied to a model checksum.
- Stored parameters and seed captured in a run manifest. Source code states RF1/RF2 use 300 trees, seed 42 and `n_jobs=-1`; ResNet code sets seed 42 and uses pretrained default weights. These source constants alone do not prove artifact provenance.

## 5. Alignment-Gate Readiness

`verify_alignment.py` was inspected but not executed. It expects, per split, the source split CSV, RF1 prediction CSV, and ResNet feature CSV. It checks rows, unique/equal `sample_id` sets, exact RF1 columns, 512 feature columns, nulls and a one-to-one ID merge.

| Required input | Expected columns | Expected rows | Exists? | Metadata suggests ready? | Potential blocker |
|---|---|---:|---|---|---|
| `data/processed/multimodal_splits/train.csv` | `sample_id`, `ERI_target`, `image_path`, `spatial_group` plus source/features/audit columns | 3,321 | Yes | Yes | Target is a constructed proxy; retain split hash. |
| `data/processed/multimodal_splits/validation.csv` | `sample_id`, `ERI_target`, `image_path`, plus source/features/audit columns | 712 | Yes | Yes | Target semantics/provenance, not structural alignment. |
| `data/processed/multimodal_splits/test.csv` | `sample_id`, `ERI_target`, `image_path`, plus source/features/audit columns | 712 | Yes | Yes | Keep test closed for any further model choices. |
| `artifacts/predictions/rf1_train_predictions.csv` | Exactly `sample_id`, `ERI_target`, `rf1_prediction` | 3,321 | Yes | Yes | Confirm artifact-to-code/target/split lineage. |
| `artifacts/predictions/rf1_validation_predictions.csv` | Exactly `sample_id`, `ERI_target`, `rf1_prediction` | 712 | Yes | Yes | Confirm artifact-to-code/target/split lineage. |
| `artifacts/predictions/rf1_test_predictions.csv` | Exactly `sample_id`, `ERI_target`, `rf1_prediction` | 712 | Yes | Yes | Confirm whether test predictions were made only after model decisions were final. |
| `artifacts/features/resnet_train_features.csv` | `sample_id` + exactly `f_001`..`f_512` | 3,321 | Yes | Yes | Record pretrained weights and transforms. |
| `artifacts/features/resnet_validation_features.csv` | `sample_id` + exactly `f_001`..`f_512` | 712 | Yes | Yes | Record pretrained weights and transforms. |
| `artifacts/features/resnet_test_features.csv` | `sample_id` + exactly `f_001`..`f_512` | 712 | Yes | Yes | Record extraction run provenance. |

Independent metadata checks found expected row/column counts, unique IDs, matching split/output ID sets, and no blank cells in these RF1/ResNet CSVs. This is not a substitute for the script's runtime check.

**Ready to run alignment gate.** The required files exist and inspected metadata satisfies the gate's expected structure. The gate was not run, per instruction; capture its result and input hashes when a human authorizes that separate action.

## 6. Dashboard and API Readiness

### Existing routes and behavior

- `GET /`: reads and returns the static HTML page.
- `WS /ws/progress`: publishes generator progress and accepts a persistent connection.
- `POST /api/start`: creates a `DatasetGenerator` and starts it in a background thread. The generator calls satellite/weather providers, downloads imagery and writes datasets/metadata.
- `POST /api/pause`, `/api/resume`, `/api/cancel`: control the long-running generator.
- `GET /api/image/{sample_id}`: reads and serves an image from a requested directory; it is not a prediction endpoint.

The current app is an operational collection tool: it can call providers and create/modify dataset outputs. No route loads RF1/RF2 or returns prediction records. Do not expose these collection controls in the exhibition app.

### Reuse/removal

- Safe to adapt after review: basic page layout, spacing, typography and selected CSS variables/components.
- There are no existing maps, risk charts or prediction tables to reuse. The current gallery shows newly generated images and the live log shows collection progress; neither represents model risk outputs.
- Remove generation form/provider settings, Generate/Pause/Resume/Cancel buttons, output-directory field, collection progress/telemetry, generation gallery actions and writable collection API calls. Do not connect WebSocket or collection endpoint behavior to the exhibition.

### Required read-only dashboard schema

The future versioned record should expose exactly the requested fields, with documented types/semantics:

| Field | Suggested meaning |
|---|---|
| `sample_id` | Stable identifier from the Bhopal aligned sample. |
| `latitude`, `longitude` | Recorded sample coordinate in EPSG:4326. |
| `actual_acquisition_date` | Satellite acquisition date, not requested weather date. |
| `final_eri_score` | RF2 output on the declared target scale; clearly label as a model estimate of the formula-derived proxy, not observed risk. |
| `risk_band` | Band derived from published, versioned thresholds. |
| `rf1_prediction` | Environmental branch score on the same declared target scale. |
| `top_feature_reasons` | Explainability output tied to a documented method; do not fabricate reasons from RF feature importance alone. |
| `model_version` | Immutable model ID/checksum or release version. |
| `target_version` | Formula/version identifier, including EMSS thresholds/units. |
| `data_limitations` | Explicit proxy-target, historical-data, spatial/temporal coverage and uncertainty statement. |

Recommended minimal view: sample map or sortable table, consistent risk-band colors, final suitability score, top environmental factors, model/target version, acquisition date and a prominent limitation notice. It must make no external provider calls and expose no collection, generation, write or delete controls. A human must approve score bands and wording before public display.

## Top Five Blocking Issues

1. **Dataset-generation provenance:** 5,000 km checked-in radius conflicts with 50 km application defaults and the stored 50 km manifest; other manifest settings also differ from current defaults. The historical run configuration and source records must be confirmed.
2. **Target validity:** EMSS is a rule-derived 0..100 score built from temperature, humidity and precipitation; RF1 consumes those same inputs. Current high metrics measure agreement with the constructed score, not independent dengue/mosquito outcomes.
3. **Artifact reproducibility:** RF2 binary/prediction/metric outputs conflict with README text saying RF2 was not run, and none are bound to code/data/target/split hashes or a run manifest.
4. **Exhibition service absent:** current FastAPI and UI can collect data and write outputs; they provide no read-only RF2 inference/prediction endpoint. A new safe API/dashboard is required.
5. **Source attribution and scientific approval:** dataset license/citation/attribution and the biological thresholds/precipitation units need human confirmation before Bhopal or breeding-risk claims.

## Exact Human Decisions Required Before Merge

1. Data Team: verify the historical center, radius, coordinates, date range, item IDs, weather source/model, and which saved data-generation parameters produced these 4,745 rows; resolve the 5,000/50 km discrepancy and location/coordinate defaults.
2. Project Lead: confirm Sentinel-2/Microsoft Planetary Computer and Open-Meteo/ECMWF licenses, citation format, attribution obligations, and permission to exhibit the saved imagery.
3. ML Team + domain reviewer: approve the target name, formula, biological thresholds and units; decide whether the project will make only suitability-proxy claims or obtain independent observed outcome labels.
4. ML Team: reconcile the README and RF2 artifact conflict; identify exact Python/packages, hardware, run date, model/data/code/split hashes, parameters, and whether held-out test outputs were produced after model selection.
5. ML Team: authorize a separate alignment-gate run and retain its output. The audit did not run the gate.
6. Dashboard Team: approve a read-only API contract, risk-band thresholds, model/target versioning, limitation text, and the map/table presentation. Do not reuse collection endpoints.

## Recommended Sequence After Verification

1. Resolve provenance, radius, attribution and target decisions; record them in a versioned data/experiment manifest.
2. Freeze Bhopal source hashes, split IDs, target version, feature schemas and package lock. Keep source folders read-only until a separately approved merge.
3. Reconcile RF1/RF2/ResNet artifact provenance. If artifacts cannot be tied to the frozen inputs, obtain approval for controlled reproduction before making claims; do not use the held-out test set for further model selection.
4. Run `verify_alignment.py` as a separately authorized, read-only check on the frozen inputs and save its output. Do not train models as part of this verification audit.
5. Implement a new read-only API and dashboard against reviewed Bhopal outputs; remove all data-generation/provider controls.
6. Review the exhibition wording, risk-band interpretation, limitations and source attribution with the Project Lead/domain reviewer.

## Audit Boundary

Only source, configuration, report and metadata inspection was performed. CSVs were read for headers/row counts/IDs/null checks and selected metadata; model binaries were not opened. No provider/API call, notebook, model, training code, target-generation script, image-feature extraction, split regeneration or alignment gate was executed. No files in `Coding_Part/` or `Vector_Borne_Risk_Project/` were changed.
