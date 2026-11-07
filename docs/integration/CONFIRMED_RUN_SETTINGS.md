# Confirmed Run Settings

## Confirmation

The ML team confirmed the intended official collection settings: Bhopal, India; 50 km radius; 2024-01-01 through 2024-06-30; maximum cloud coverage 20%; 512 x 512 imagery; seed 42; generation output directory `dataset`; and EMSS stored in `ERI_target` on a 0 to 100 scale.

**The official collection radius is 50 km.** The `Vector_Borne_Risk_Project/config/dataset_config.json` value of 5,000 km is obsolete/invalid and must not be treated as the generation record. The generator schema limits radius to 500 km, and the stored manifest records 50 km.

## Settings and Evidence

| setting | confirmed_value | evidence_source | status |
|---|---|---|---|
| Location | Bhopal, India | ML team confirmation; stored `data/raw/bhopal_satellite/metadata/dataset_manifest.json` | CONFIRMED |
| Center coordinates | Not specified by confirmation; manifest records 23.2584857, 77.401989; config/runtime defaults record 23.2599, 77.4126 | ML team confirmation; stored manifest; `config/dataset_config.json`; `app/config.py` | CLARIFICATION_REQUIRED |
| Radius | 50 km | ML team confirmation; stored manifest; `app/config.py` and web default | CONFIRMED; 5,000 km config is obsolete/invalid |
| Start date | 2024-01-01 | ML team confirmation; stored manifest and dataset metadata | CONFIRMED |
| End date | 2024-06-30 | ML team confirmation | CLARIFICATION_REQUIRED: stored manifest ends 2025-06-30 and stored observations extend to 2025-06-19 requested / 2025-06-16 acquired |
| Cloud maximum | 20% | ML team confirmation; `config/dataset_config.json`; current API default | CLARIFICATION_REQUIRED: stored manifest says 50%, and quality report maximum is 49.46% |
| Image size | 512 x 512 | ML team confirmation; stored manifest | CONFIRMED; current config/API defaults say 256 |
| Random seed | 42 | ML team confirmation; stored manifest and dataset config | CONFIRMED |
| Generation output directory | `dataset` | ML team confirmation; `app/config.py` and web request default | CLARIFICATION_REQUIRED: checked-in config names `data/raw/bhopal_satellite`; current files are stored there, while manifest does not record the original output path |
| Requested sample count | “samples 7” as relayed in the confirmation context; intended meaning is unclear | ML team confirmation (user-provided) | CLARIFICATION_REQUIRED: config says 1,500; stored manifest says 5,000; actual stored valid rows are 4,745 |
| Actual stored record count | 4,745 valid records | `metadata/dataset_manifest.json`; `metadata/quality_report.json`; split report | CONFIRMED as stored count; does not establish requested target count |
| Target | Environmental Mosquito Suitability Score (EMSS), stored in `ERI_target` | ML team confirmation; `scripts/data/generate_emss_target.py`; target-bearing split files | CONFIRMED |
| Target scale | 0 to 100 | ML team confirmation; `scripts/data/generate_emss_target.py` | CONFIRMED |
| Alignment gate | Authorized and run read-only; all train/validation/test checks passed | ML team authorization; `scripts/preprocessing/verify_alignment.py` execution in this audit | PASS |
| RF2 status | Trained after revised scripts; model, validation/test predictions and metrics exist | ML team confirmation; artifacts and metrics; `README.md` | CONFIRMED by ML team; README statement is stale |

## Required Clarification: “Samples 7”

The ML confirmation says “samples 7,” while the stored Bhopal data has 4,745 valid records. The stored manifest reports `target_samples: 5000`; the checked-in dataset config reports 1,500. Confirm whether “samples 7” means a requested count, a command-line argument, a shorthand for 7,000, or something else. Do not silently reinterpret it. The actual stored record count remains 4,745 until that question is resolved.

## Stored-Run Conflicts

The confirmed end date 2024-06-30 differs from the stored manifest end date 2025-06-30 and stored observation dates through 2025-06-19. The confirmed 20% cloud maximum differs from the manifest's 50% and quality report's maximum of 49.46%. The confirmed image size 512 agrees with the manifest, while current defaults/config say 256. These conflicts mean the confirmed settings should be treated as the team's intended run settings, not yet as a fully reconciled provenance record for every stored row.

No source config, metadata, split, image, or model artifact was changed to resolve these discrepancies. This note is a correction record only; it does not replace `config/dataset_config.json` or `dataset_manifest.json`.
