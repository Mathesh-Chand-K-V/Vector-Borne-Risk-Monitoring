# VectorSafe AI – Environmental Mosquito Suitability Dashboard

## 1. Purpose

This Streamlit dashboard provides a read-only visualization of the supplied Bhopal environmental suitability records.

It focuses on the formula-derived Environmental Mosquito Suitability Score (EMSS) and is intended to support preventive environmental inspection prioritization.

The dashboard does not perform model training or target generation.

## 2. Installation

Install the dashboard dependencies from the repository root:

```bash
pip install -r dashboard/requirements.txt
```

Alternatively, change into the dashboard directory:

```bash
cd dashboard
pip install -r requirements.txt
```

## 3. Start the dashboard

From the repository root:

```bash
streamlit run dashboard/app.py
```

Alternatively, from the dashboard directory:

```bash
streamlit run app.py
```

On Windows, the batch helper can also be used from the dashboard directory:

```bat
run_dashboard.bat
```

## 4. Data requirements

The dashboard reads:

```text
dashboard/data/dashboard_records.csv
dashboard/data/dashboard_metadata.json
```

These files are retained in the source repository because the dashboard currently depends on the supplied dashboard records.

If these files are missing or replaced, the dashboard may fail to start or display different results.

The modelling pipeline and dashboard data-generation relationship must be verified separately before replacing these files.

Retraining RF1 or RF2 does not automatically regenerate the dashboard records.

## 5. Dashboard features

- Sidebar filters for split, risk band, and score threshold
- Optional date-range filtering
- Map view for Bhopal-area environmental suitability
- Risk-band count chart
- Final EMSS distribution histogram
- RF1 versus RF2 comparison scatter plot when RF1 output is available
- Split-wise EMSS chart
- Sortable table of filtered records

### Fast Exhibition Mode

Fast Exhibition Mode is enabled by default.

It limits the visible table and hides the RF1-versus-RF2 scatter plot unless the mode is turned off.

Only one interactive PyDeck map is intentionally displayed to avoid multiple map renderers and slow page startup.

## 6. EMSS risk-band thresholds

| Risk band | Score range |
|---|---:|
| Low | 0.00–33.33 |
| Medium | 33.34–66.66 |
| High | 66.67–100.00 |

These are visualization thresholds for environmental inspection prioritization.

## 7. Safe interpretation

Appropriate interpretations:

- EMSS is a formula-derived environmental suitability proxy.
- The dashboard supports preventive environmental inspection prioritization.
- Results apply to the supplied Bhopal-area records and their documented spatial and date ranges.

The dashboard must not be used to claim:

- Dengue diagnosis
- Confirmed disease outbreak prediction
- Confirmed mosquito presence
- Confirmed breeding-ground detection

## 8. Limitations

- EMSS is a formula-derived environmental proxy.
- The dashboard does not diagnose disease or predict confirmed outbreaks.
- The dashboard does not establish mosquito presence or breeding-ground detection.
- Risk bands are visualization thresholds, not independently confirmed biological classifications.
- Results are limited to the supplied Bhopal-area dataset and its documented date range.
- The dashboard is read-only and does not train models or generate targets.
- Final licensing, attribution wording, and provider terms must be reviewed before public deployment.

## 9. Data-source attribution

The project documentation identifies these data sources:

- Copernicus Sentinel-2 L2A through Microsoft Planetary Computer
- Open-Meteo Historical Weather API, using ECMWF ERA5 / ERA5-Land where applicable

Review applicable provider terms and attribution requirements before public deployment.

## 10. Notes

The dashboard is a visualization layer over supplied records.

It is separate from the upstream data-generation, preprocessing, and modelling stages.

The original source projects are not modified by dashboard execution.
