# Dataset Roles and Safety

## Controlling rule

- Bhopal environmental records and Bhopal satellite images linked by `sample_id` are the MAIN production/research workflow.
- Climate-lag data from Iquitos/San Juan and MosquitoFusion/YOLO images are AUXILIARY experiments unless a verified matching location/time/`sample_id` relationship exists.
- Do not directly fuse auxiliary climate rows with auxiliary mosquito images, and do not use simulated pairings as real-world accuracy evidence.
- If a dataset is synthetic or simulated, label it clearly in filenames, metadata, reports, plots and presentations; never describe it as observed data.

The climate-lag table's `risk_label`/`next_week_cases` target is not interchangeable with Bhopal's formula-derived `ERI_target`/EMSS. The inspected target scripts derive the Bhopal target from environmental variables that are also RF1 inputs, so current metrics can measure reconstruction of a constructed suitability score rather than prediction of observed disease incidence. Review thresholds and units, and obtain an independently observed outcome before making disease-forecast claims.

The available simulated climate-image table pairs Iquitos/San Juan climate with Bhopal image features by seeded random sampling. Its metadata marks it simulated and prohibits predictive training/evaluation or real-world accuracy claims.

Keep sample IDs, data provenance and fixed split assignments unchanged. Never join domains by row order, filename or assumed geography. The Bhopal files are historical, not a live feed.

Audit scope: metadata/source inspection only. No training, downloads, notebook execution, image decoding, data regeneration, source edits, copying or deletion occurred.
