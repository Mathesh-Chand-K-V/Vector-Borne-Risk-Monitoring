# Offline training data and real-world inference

Project Exhibition can use offline data to develop RF1, then run the model on real-world records. Dataset usage roles should remain distinct from claims about how the data was produced or whether a live service exists.

## Current data and roles

| Data | Actual source status | Intended role | Current limitation |
|---|---|---|---|
| `experiments/auxiliary_research/data/interim/climate/final_climate_lag_dataset.csv` | Historical city-week observations from Iquitos and San Juan, 2000–2008 | Source for an offline synthetic RF1 training candidate | The source itself is not synthetic; its `risk_label`/`next_week_cases` target differs from `ERI_target` |
| `experiments/auxiliary_research/data/synthetic/climate_synthetic_eri_train.csv` | Generated with a city-stratified Gaussian copula and project EMSS formula | Candidate offline training data for an experimental common-feature RF1 | Formula-derived targets, only three features, and source climate remains a different geography/domain |
| `experiments/auxiliary_research/data/synthetic/simulated_multimodal_demo_pairs.csv` | Artificial seeded pairs of Bhopal images/ResNet features and synthetic climate rows | Demo the multimodal schema and data-flow only | No true location/date relationship; prohibited for model training, evaluation, or accuracy claims |
| `data/interim/environmental_risk/master_dataset_with_emss.csv` and aligned splits | Stored real-world Bhopal-area satellite/environment records | Main target-domain training, validation, and test data; schema for future inference | Static records, not a live feed; current RF1 uses 17 features |
| `experiments/auxiliary_research/data/raw/mosquito_yolo/` | Separately annotated image data | Separate object-detection experiment | No verified sample linkage to the climate or Bhopal fusion rows |

## Current status

The synthetic dataset has 1,178 rows, one per historical climate source row. This is not larger than the 4,745-row Bhopal master, and synthetic expansion does not create independent evidence. Its features are `temperature_c`, `humidity`, and daily-equivalent `precipitation`; `ERI_target` reuses the current candidate EMSS scoring function. Weekly precipitation is divided by seven to approximate a mean daily amount. The formula and time-scale conversion are documented in the synthetic dataset metadata.

The generated ERI distribution is shifted high (median 95.45) compared with the Bhopal EMSS distribution (median 45.49). A comparison experiment is required before using these rows for RF1. Do not merge them into the existing 17-feature production RF1 or regenerate its artifacts without a validation-based comparison.

The simulated multimodal demo contains 4,745 rows and 512 existing image embedding features per row. Climate rows are selected randomly with replacement (seed 2026); they are not matched by geography or date. `simulated_climate_ERI_target` is calculated from climate values only and is not a ground-truth label for the image. This table is suitable for verifying a prototype's data loading and fusion interface, but not for fitting or evaluating a predictive model.

## Next modeling steps

1. Review the EMSS suitability thresholds and the weekly-to-daily precipitation assumption.
2. Fit a three-feature Bhopal-only RF1 baseline and a synthetic-augmented variant using the same features.
3. Compare both on the same Bhopal validation samples. Keep the Bhopal test partition untouched.
4. If the augmented variant improves validation and has defensible error across conditions, regenerate RF1 predictions with the original `sample_id` values and rerun sample-alignment checks before retraining RF2.
5. For live inference, connect a real-time or regularly refreshed input feed to the frozen preprocessing and model. Until then, refer to the stored Bhopal dataset as the real-world target domain, not real-time data.
