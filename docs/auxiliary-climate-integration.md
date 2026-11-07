# Auxiliary climate data for RF1

## Decision

The `experiments/auxiliary_research/data/interim/climate/final_climate_lag_dataset.csv` table is a candidate auxiliary environmental source for RF1. It has about 1,178 weekly records from Iquitos (`iq`) and San Juan (`sj`), dated 2000-07-15 to 2008-04-22. The current image-aligned master has 4,745 Bhopal-area records. The auxiliary table is historically broader but is not larger by row count in this copy, and its geography and period differ from the main data. Its supplied splits contain 1,176 rows (823 train, 176 validation, 177 test), two fewer than the climate master. No duplicate city-week keys were found; the omitted keys are `iq|2010|25` and `sj|2008|17`, the last recorded week for each city. They likely lack a following-week outcome, but verify target creation before rebuilding splits. See `experiments/auxiliary_research/reports/climate/city_week_split_audit.csv`.

Use the auxiliary climate records only in an RF1 training experiment after target and feature harmonization. They have no image or `sample_id` linkage, so they cannot be joined directly into ResNet features or the RF2 fusion rows. Keep the Bhopal-aligned train/validation/test partitions intact.

## Target compatibility

Current RF1 is trained to predict `ERI_target` (the current target-generation script uses this column for the environmental suitability score). The auxiliary climate dataset instead provides `next_week_cases` and `risk_label`, where `risk_label` is a category derived from next-week case counts.

For an experiment that preserves the existing architecture, define auxiliary `ERI_target` with the exact same finalized, frozen ERI/EMSS formula used for the Bhopal training data. Do not substitute `risk_label` or train the current RF1 on `next_week_cases`; those are disease-outcome targets with different meaning. Exclude both `total_cases` and `next_week_cases` from environmental suitability inputs. If the intended project outcome changes to next-week dengue prediction, make that a separately named/versioned modeling objective and evaluate it consistently across every branch.

The ERI/EMSS formula is a project decision recorded as not yet finalized in `docs/context/project_context.md`. Finalize and document it before generating auxiliary targets.

## Feature compatibility

Current RF1 uses latitude, longitude, cloud, temperature, humidity, wind, precipitation, and weather-code features. The auxiliary table has no latitude/longitude, cloud, wind, or weather-code equivalents. Potential common variables are:

- Temperature: map a single selected climate field to Celsius. `station_avg_temp_c` is already Celsius; Kelvin reanalysis fields require conversion.
- Humidity: `reanalysis_relative_humidity_percent` to the current humidity definition.
- Precipitation: select one justified field (`precipitation_amt_mm` or `station_precip_mm`) that matches the main pipeline's measurement definition.

Do not fill unavailable location/cloud/wind fields with arbitrary constants. Either train a clearly labeled common-feature RF1 using only compatible predictors and regenerate its predictions/features throughout, or develop a separately tracked transfer model. Add lag features only if matching lag definitions can be built for the main data without crossing split boundaries.

## Experiment sequence

1. Audit city-week uniqueness, missingness, units, dates, lag creation, and the existing chronological split. Fit imputers, encoders, scaling, and any outcome-derived cutoffs on training data only.
2. Freeze the same ERI/EMSS scoring method and feature definitions for both sources. Keep source and geography identifiers for analysis; do not let an arbitrary integer `city_encoded` stand in for a measured environmental variable.
3. Train two RF1 variants: the existing Bhopal-only baseline and an auxiliary-augmented variant. Use the original Bhopal training partition plus auxiliary training records. Do not add auxiliary records to validation or test.
4. Compare the variants on the untouched Bhopal validation partition. Check performance by source/geography and inspect error and feature importance changes.
5. Choose the RF1 variant using validation only. Then regenerate its train/validation/test predictions for the original Bhopal samples, retain `sample_id`, run alignment checks, and continue to RF2. Open the held-out test only for final evaluation.

## Related but separate tracks

The YOLO mosquito images and the municipality-wide satellite-dengue tables in `experiments/auxiliary_research/` are separate data tracks. They do not provide direct image/sample alignment for the Bhopal fusion rows. Do not combine them with climate rows unless a documented location-time/sample key and compatible target are established.


## Synthetic candidate now prepared

A separate synthetic feature/target table has been generated at `experiments/auxiliary_research/data/synthetic/climate_synthetic_eri_train.csv`, using the source climate measurements, a city-stratified Gaussian copula, weekly-to-daily mean precipitation conversion, and the current EMSS scoring implementation. Generation code is `experiments/auxiliary_research/scripts/generate_synthetic_climate.py`; parameters and limitations are in the synthetic folder's metadata and README.

This synthetic candidate uses only temperature, humidity, and precipitation, while production RF1 currently uses 17 features. Its target median (95.45) is substantially above the Bhopal target median (45.49). Do not directly concatenate it into the current RF1 training table. The next safe experiment is a three-feature Bhopal-only baseline versus a three-feature synthetic-augmented RF1, selected on the existing Bhopal validation rows. Production RF1 artifacts remain unchanged until that comparison supports a change.
