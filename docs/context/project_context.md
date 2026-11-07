
# Project Exhibition — Complete Context Recovery Summary

## 1. Project Purpose and Overall Architecture

Project Exhibition is a **multimodal machine-learning system** that combines **image-based information** with **environmental/tabular information** for disease/environmental risk prediction.

The architecture has three major components:

1. **ResNet** — image branch and feature extractor.
2. **RF1** — environmental Random Forest branch that learns/predicts the Environmental Risk Index (ERI).
3. **RF2** — final fusion Random Forest combining ResNet image features with RF1 environmental information.

The fundamental architecture is:

```text
                    ORIGINAL MASTER DATASET
                             │
              ┌──────────────┴──────────────┐
              │                             │
           IMAGE DATA                ENVIRONMENTAL DATA
              │                             │
              ▼                             ▼
           ResNet                          RF1
              │                             │
              ▼                             ▼
      Image feature vector          RF1 ERI prediction
              │                             │
              └──────────────┬──────────────┘
                             ▼
                            RF2
                             │
                             ▼
                      Final prediction
```

The responsibilities are deliberately separated:

- **ResNet = visual information**
- **RF1 = environmental information / ERI**
- **RF2 = multimodal fusion**

The architecture should **not be replaced merely because a larger environmental dataset was discovered**. The larger dataset strengthens the environmental branch, particularly RF1/ERI development.

---

## 2. ResNet Branch

ResNet is used strictly as a **feature extractor**.

The image passes through ResNet and is converted into a numerical feature vector:

```text
Image
  ↓
ResNet
  ↓
Numerical feature vector
  ↓
ResNet feature CSV
```

ResNet **does not use `ERI_target`** during feature extraction.

This distinction is important:

```text
Image → ResNet → image features
```

NOT:

```text
Image + ERI_target → ResNet
```

The established ResNet artifacts are:

```text
D:\DSN_Project\models\resnet_train_features.csv
D:\DSN_Project\models\resnet_validation_features.csv
D:\DSN_Project\models\resnet_test_features.csv
```

These files contain numerical image features and eventually provide the visual input to RF2.

The exact feature dimensionality depends on the implemented ResNet configuration and must not be invented.

---

## 3. RF1 Environmental Branch

RF1 is a Random Forest model operating on environmental/tabular information.

Unlike ResNet, RF1 **does use the ERI target**.

Conceptually:

```text
Environmental variables
        ↓
Environmental preprocessing
        ↓
ERI_target
        ↓
RF1
        ↓
rf1_prediction
```

The established RF1 prediction files are:

```text
rf1_train_predictions.csv
rf1_validation_predictions.csv
rf1_test_predictions.csv
```

Their established structure is exactly:

```csv
sample_id,ERI_target,rf1_prediction
```

Meaning:

- `sample_id` = identity/linking key for the observation.
- `ERI_target` = ground-truth environmental risk value.
- `rf1_prediction` = RF1's predicted ERI.

These RF1 outputs are designed to be joined with ResNet features before RF2.

RF1 also has environmental train/validation/test datasets following the established pattern:

```text
rf1_train_*.csv
rf1_validation_*.csv
rf1_test_*.csv
```

The exact names of every RF1 preprocessing/input file have not all been explicitly established, so do **not invent filenames** for them.

---

## 4. ERI Target

Every relevant data point already has an **ERI target**.

ERI belongs to the environmental/RF1 branch.

It is not required by ResNet.

The intended relationship is:

```text
Environmental record
        +
   ERI_target
        ↓
       RF1
```

The ERI is intended to represent environmental risk/conditions through a compact environmental index.

An earlier idea was to create ERI through a weighted combination of variables such as:

- temperature
- cloud conditions
- humidity
- precipitation

However, raw variables have different units and scales, so a simple raw weighted average is not methodologically appropriate.

The preferred direction is:

```text
Environmental variables
        ↓
Normalization / standardization
        ↓
Environmental components
        ↓
Justified weighting methodology
        ↓
ERI
```

Possible conceptual components include:

```text
Temperature
Humidity
Precipitation
Vegetation / environmental conditions
```

Weights should not be arbitrary. They should be justified through a defensible methodology or derived from the data where appropriate.

The exact final ERI formulation is **not yet finalized**.

---

## 5. Sample ID Is the Critical Link

`sample_id` is effectively the primary linkage key across the multimodal system.

One observation must maintain the same identity across every branch:

```text
sample_id
   ├── image
   ├── ResNet features
   ├── environmental variables
   ├── ERI_target
   └── rf1_prediction
```

Matching row counts are not sufficient.

The actual samples must correspond.

For example, this is correct:

```text
sample_001 → ResNet TRAIN
sample_001 → RF1 TRAIN
```

This is incorrect:

```text
sample_001 → ResNet TRAIN
sample_001 → RF1 TEST
```

because RF2 would receive mismatched modalities.

Therefore, train/validation/test partitions must remain **sample-aligned**.

---

## 6. Train / Validation / Test Structure

The project uses three primary partitions:

```text
TRAIN
VALIDATION
TEST
```

For ResNet:

```text
resnet_train_features.csv
resnet_validation_features.csv
resnet_test_features.csv
```

For RF1, corresponding environmental and prediction artifacts exist.

The same `sample_id` must remain in the same partition across branches.

The held-out test set must remain isolated from model-development decisions.

---

## 7. RF1 Feature Importance

The project has an established RF1 feature-importance artifact:

```text
rf1_feature_importance.csv
```

Its purpose is to document the importance/contribution of environmental variables used by RF1.

Conceptually:

```text
feature
importance
rank
```

The exact columns depend on the implementation.

Whenever the RF1 feature set changes substantially, RF1 must be retrained and:

```text
rf1_feature_importance.csv
```

must be regenerated.

The RF1 prediction files must also be regenerated after such a change.

---

## 8. Original / Master Dataset

The original/master dataset is the **authoritative image-aligned dataset** for the complete multimodal system.

It maintains the relationship between environmental observations and corresponding image/sample information.

Conceptually:

```text
Environmental observation
          ↕
      sample_id
          ↕
         image
```

This original paired dataset should remain the basis for final end-to-end multimodal evaluation.

It should not be replaced by the newer environmental-only dataset.

---

## 9. New Larger Environmental Dataset

A substantially larger environmental dataset was discovered.

It contains environmental and epidemiological information but **does not have corresponding images**.

The columns are:

```text
city
year
weekofyear
week_start_date

ndvi_ne
ndvi_nw
ndvi_se
ndvi_sw

precipitation_amt_mm

reanalysis_air_temp_k
reanalysis_avg_temp_k
reanalysis_dew_point_temp_k
reanalysis_max_air_temp_k
reanalysis_min_air_temp_k
reanalysis_precip_amt_kg_per_m2
reanalysis_relative_humidity_percent
reanalysis_sat_precip_amt_mm
reanalysis_specific_humidity_g_per_kg
reanalysis_tdtr_k

station_avg_temp_c
station_diur_temp_rng_c
station_max_temp_c
station_min_temp_c
station_precip_mm

total_cases

rain_lag_1_week
rain_lag_2_weeks
humidity_lag_1_week
humidity_lag_2_weeks
temp_lag_1_week
temp_lag_2_weeks
```

Particularly highlighted variables include:

```text
precipitation_amt_mm
reanalysis_tdtr_k
humidity_lag_2_weeks
```

but these are not automatically the only important variables.

This dataset should currently be treated as an:

**auxiliary / additional tabular environmental dataset**

It should **not automatically be called synthetic**. It should only be called synthetic if it was actually synthetically generated.

---

## 10. How the New Dataset Should Be Used

The larger environmental dataset should strengthen the RF1/ERI branch.

It should not be forced into the image branch.

The intended workflow is:

```text
Large environmental-only dataset
             ↓
Environmental preprocessing
             ↓
Feature grouping / selection
             ↓
ERI development
             ↓
RF1 development
```

Then the finalized environmental model/pipeline can be applied to the original image-aligned dataset:

```text
Original paired dataset
       │
       ├── Images → ResNet → image features
       │
       └── Environment → RF1 → RF1 prediction
                                │
                                ▼
                              RF2
```

There is **no need to fabricate images** for the new environmental observations merely to make them usable.

The larger dataset can improve environmental modeling while the original dataset remains responsible for complete multimodal evaluation.

---

## 11. Environmental Feature Groups

The new dataset contains many variables measuring similar underlying phenomena.

They should therefore not automatically all be treated as independent, equally important features.

### Temperature

Examples:

```text
reanalysis_air_temp_k
reanalysis_avg_temp_k
reanalysis_dew_point_temp_k
reanalysis_max_air_temp_k
reanalysis_min_air_temp_k
station_avg_temp_c
station_diur_temp_rng_c
station_max_temp_c
station_min_temp_c
reanalysis_tdtr_k
```

### Humidity

```text
reanalysis_relative_humidity_percent
reanalysis_specific_humidity_g_per_kg
humidity_lag_1_week
humidity_lag_2_weeks
```

### Precipitation

```text
precipitation_amt_mm
reanalysis_precip_amt_kg_per_m2
reanalysis_sat_precip_amt_mm
station_precip_mm
```

### Vegetation

```text
ndvi_ne
ndvi_nw
ndvi_se
ndvi_sw
```

### Lagged weather

```text
rain_lag_1_week
rain_lag_2_weeks
humidity_lag_1_week
humidity_lag_2_weeks
temp_lag_1_week
temp_lag_2_weeks
```

### Temporal

```text
year
weekofyear
week_start_date
```

### Geographic

```text
city
```

### Outcome-related

```text
total_cases
```

The key methodological issue is redundancy. For example, including many highly correlated temperature measurements in an ERI can effectively cause temperature to be counted multiple times.

Therefore, feature selection and importance analysis are required.

---

## 12. Lagged Variables

The new dataset contains environmental history variables:

```text
rain_lag_1_week
rain_lag_2_weeks

humidity_lag_1_week
humidity_lag_2_weeks

temp_lag_1_week
temp_lag_2_weeks
```

These are potentially important because disease/environmental effects may not occur immediately after a weather event.

They allow RF1 to represent:

```text
Current environment
        +
Recent environmental history
```

rather than only the current week's conditions.

The exact decision of whether lagged variables belong directly in the ERI construction or only as RF1 predictors remains **unfinalized**.

---

## 13. `total_cases` Must Be Treated Carefully

`total_cases` requires explicit leakage analysis.

It may represent the downstream disease outcome.

Therefore, it must **not automatically be used as an RF1 input feature**.

If `total_cases` is the final prediction target, including it in RF1 could cause target leakage.

Its final role depends on the precise final prediction objective, which still needs to be explicitly defined.

Do not silently include it.

---

## 14. `city` Must Be Treated Carefully

`city` contains geographic information.

It should not automatically be one-hot encoded and added to RF1.

The project needs to decide whether the purpose is:

1. learning city/location-specific relationships, or
2. learning more general environmental relationships that should generalize across locations.

`city` may therefore be treated as a geographic/context/grouping variable rather than automatically as a normal predictive feature.

This remains an explicit methodological decision point.

---

## 15. Temporal Variables

The new dataset contains:

```text
year
weekofyear
week_start_date
```

`week_start_date` should not simply be treated as an arbitrary raw text column.

If temporal information is used, appropriate temporal representations should be derived in a methodologically justified way.

Temporal splitting may also matter if the desired evaluation is temporal generalization.

The exact temporal validation strategy is not yet finalized.

---

## 16. RF2 Fusion

RF2 is the final multimodal fusion model.

It combines the two learned representations:

```text
ResNet features
       +
RF1 prediction
       ↓
      RF2
       ↓
Final prediction
```

The intended RF2 dataset conceptually contains:

```text
sample_id
feature_1
feature_2
...
feature_N
rf1_prediction
final_target
```

The exact number of ResNet features depends on the actual ResNet implementation.

The essential requirement is:

```text
ResNet features.sample_id
              =
RF1 prediction.sample_id
```

for every RF2 observation.

RF2 should not be responsible for relearning all raw environmental variables. RF1 already provides the environmental representation.

---

## 17. Complete Artifact Dependency

The complete pipeline is:

```text
                         MASTER DATASET
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
             Image data              Environmental data
                 │                           │
                 ▼                           ▼
              ResNet                       RF1
                 │                           │
                 ▼                           ▼
resnet_train_features.csv      rf1_train_predictions.csv
resnet_validation_features.csv rf1_validation_predictions.csv
resnet_test_features.csv       rf1_test_predictions.csv
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               ▼
                         RF2 fusion dataset
                               │
                               ▼
                              RF2
                               │
                               ▼
                        Final prediction
```

The larger environmental dataset sits upstream of RF1:

```text
Large environmental dataset
           ↓
     ERI development
           ↓
      RF1 development
           ↓
RF1 applied to original paired data
           ↓
         RF2
```

---

## 18. Current Established Artifacts

Known exact artifacts:

```text
D:\DSN_Project\models\resnet_train_features.csv
D:\DSN_Project\models\resnet_validation_features.csv
D:\DSN_Project\models\resnet_test_features.csv

rf1_train_predictions.csv
rf1_validation_predictions.csv
rf1_test_predictions.csv

rf1_feature_importance.csv
```

The RF1 prediction structure is:

```csv
sample_id,ERI_target,rf1_prediction
```

The exact filenames of every RF1 preprocessing dataset, the original master dataset, the new environmental dataset, and the final RF2 dataset have **not all been explicitly established**. Do not invent them.

---

## 19. Current Project Status

### Established / implemented

- Multimodal architecture defined.
- ResNet image branch established.
- ResNet used for feature extraction only.
- ResNet train/validation/test feature CSVs established.
- RF1 environmental branch established.
- ERI target available for relevant observations.
- RF1 train/validation/test organization established.
- RF1 prediction format established.
- RF1 prediction CSVs established.
- RF1 feature-importance artifact established.
- `sample_id` alignment requirement established.
- Consistent train/validation/test partition requirement established.
- Original image-aligned dataset established as the core multimodal dataset.
- Larger environmental-only dataset identified.
- Decision made not to fabricate images for the environmental-only dataset.
- Environmental feature grouping established conceptually.
- Potential leakage from `total_cases` identified.
- `city` treatment identified as a methodological decision.
- Lagged environmental variables identified as potentially important.

### Still to finalize

1. Exact ERI construction methodology.
2. Exact ERI weighting methodology.
3. Final RF1 feature set.
4. Whether lagged variables enter ERI directly or only RF1.
5. Treatment of `city`.
6. Treatment of temporal variables.
7. Exact role of `total_cases`.
8. Leakage-safe splitting strategy.
9. Retraining RF1 using the finalized feature set.
10. Regenerating `rf1_feature_importance.csv`.
11. Regenerating all RF1 prediction artifacts after final RF1 changes.
12. Applying the finalized RF1 pipeline to the original image-aligned dataset.
13. Constructing the final RF2 fusion dataset.
14. Training RF2.
15. Evaluating the complete ResNet + RF1 + RF2 system on the held-out test set.

---

## 20. Non-Negotiable Implementation Rules

These rules must be preserved when continuing the project:

1. **ResNet does not use `ERI_target` for feature extraction.**
2. **RF1 uses the ERI target.**
3. **RF1 predictions retain `sample_id`.**
4. **RF1 prediction files use `sample_id,ERI_target,rf1_prediction`.**
5. **ResNet and RF1 train/validation/test partitions must remain sample-aligned.**
6. **RF2 receives ResNet-derived image features and RF1-derived environmental information.**
7. **The new environmental-only dataset is not an image dataset.**
8. **Do not call the new dataset synthetic unless it was actually synthetically generated.**
9. **`total_cases` must be checked for target leakage before being used as an RF1 feature.**
10. **`city` must have an explicit methodological role.**
11. **Highly correlated environmental variables must be evaluated before constructing ERI.**
12. **Changing the RF1 feature set requires RF1 retraining.**
13. **Changing RF1 features requires regeneration of `rf1_feature_importance.csv`.**
14. **Changing RF1 features also requires regeneration of RF1 prediction artifacts.**
15. **The held-out test set must remain isolated from model-development decisions.**
16. **The original image-aligned dataset remains the basis for evaluating the complete multimodal system.**
17. **Do not invent filenames, feature dimensions, targets, or preprocessing steps that have not been established.**
18. **Do not redesign the ResNet branch merely because the environmental dataset becomes larger.**
19. **The larger environmental dataset strengthens RF1/ERI development rather than replacing the paired multimodal dataset.**
20. **The same `sample_id` must represent the same observation across image, environmental, ERI, RF1, and RF2 artifacts.**

---

## 21. One-Sentence Recovery of the Entire Project

**Project Exhibition is a sample-aligned multimodal system in which ResNet extracts image features without using ERI, RF1 learns/predicts an environmental risk representation from structured environmental data using the ERI target, a larger image-free environmental dataset is used to strengthen ERI/RF1 development, and RF2 fuses ResNet features with RF1 predictions to produce the final prediction on the original image-aligned dataset.**

## 22. Recovery Instruction for Future Chat

If this summary is pasted into a new chat, treat it as the **current authoritative project context**.

Do not restart the architecture discussion from scratch.

First preserve the established decisions and artifacts. Only modify them when the user explicitly proposes a methodological change or provides new evidence/data.

When discussing implementation, always distinguish:

```text
ResNet = image feature extraction
RF1    = environmental → ERI prediction
RF2    = image + environmental fusion
```

and always preserve:

```text
sample_id
```

as the cross-branch alignment key.


## Later update: synthetic auxiliary candidate

The historical climate source discussed in the earlier sections remains observational and must not itself be labeled synthetic. A separate generated derivative now exists at `experiments/auxiliary_research/data/synthetic/climate_synthetic_eri_train.csv`. It was generated using a city-stratified Gaussian copula and the current candidate EMSS scoring implementation; it excludes disease case outcomes. It is an experimental three-feature RF1 training candidate only. It has not been added to production RF1, and the ERI/EMSS formula remains subject to scientific review. See `docs/data-roles.md` and `docs/auxiliary-climate-integration.md`.
