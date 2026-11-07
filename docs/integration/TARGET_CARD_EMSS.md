# Target Card: EMSS

| Attribute | Definition |
|---|---|
| Target name | Environmental Mosquito Suitability Score (EMSS) |
| Stored column name | `ERI_target` |
| Scale | 0 to 100 |
| Target type | Formula-derived environmental proxy |
| Formula implementation | `Vector_Borne_Risk_Project/scripts/data/generate_emss_target.py` |

## Formula

```text
EMSS = 100 * (
    0.40 * temperature_suitability +
    0.30 * humidity_suitability +
    0.30 * precipitation_suitability
)
```

The component suitability values are generated from temperature, humidity and precipitation thresholds and clipped to [0, 1]. Review the script's thresholds and measurement units before interpreting the score. The computed EMSS value is stored in the legacy-named `ERI_target` column for pipeline compatibility.

## What This Is Not

The target is not:

- confirmed dengue prediction
- disease diagnosis
- observed mosquito abundance
- confirmed breeding-ground label

It is not an independently observed mosquito or disease outcome. RF1 also receives environmental variables used to construct the target, so its predictions can reproduce the formula rather than forecast an independent outcome.

## Approved Exhibition Wording

> AI-based environmental mosquito suitability estimation using aligned satellite imagery and environmental variables.

## Required Limitation Statement

> The model estimates a formula-derived environmental suitability proxy and should support preventive inspection prioritization, not clinical diagnosis or confirmed outbreak prediction.

Use this statement adjacent to displayed scores, maps and metrics. Do not imply that a high EMSS value confirms mosquito presence, breeding sites, dengue cases, or an outbreak.
