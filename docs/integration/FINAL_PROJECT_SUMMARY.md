# Final Project Summary

## Contributions

- **Coding_Part** contributes the Iquitos/San Juan historical climate-lag dataset and splits, data dictionary/cleaning notebook, plots, and a separate MosquitoFusion/YOLO breeding-image experiment with binary labels. Its 1229 inventoried relevant files support auxiliary research, not a verified Bhopal fusion.
- **Vector_Borne_Risk_Project** contributes the Bhopal environmental/satellite records, sample_id-aligned splits, ERI/EMSS scripts, frozen ResNet-18 features, spatially grouped RF1, RF2 fusion, alignment checks, reports/artifacts, and a FastAPI dataset-generation app. It has 7270 inventoried relevant files.

## Official workflow

The official main candidate is the Bhopal environmental and satellite workflow linked by `sample_id`: prepare and freeze splits, run RF1 plus 512-feature ResNet-18, verify alignment, then use RF2. The present ERI/EMSS target is formula-derived, not an observed disease outcome; report that distinction clearly.

## Auxiliary work

Keep Iquitos/San Juan climate-lag and MosquitoFusion/YOLO breeding images separate. Mark synthetic/simulated data explicitly and never use random pairings as real-world accuracy evidence.

## Exhibition architecture

A reproducible Bhopal table feeds an environmental RF1 and frozen ResNet-18 branch; a sample_id alignment gate protects RF2 fusion; a read-only dashboard displays approved Bhopal output, model/data version and limitations. Do not connect the current dataset-generation API as the exhibition prediction service.

## Next actions in priority order

1. Confirm Bhopal provenance, geography, dates and the configured 5000 km versus 50 km radius discrepancy.
2. Scientifically review ERI/EMSS target definition/units and obtain an independent observed outcome before disease-forecast claims.
3. Freeze checksums, target code, split IDs, feature schema and tested dependencies; verify RF1 grouped OOF prediction provenance.
4. Resolve two omitted climate city-weeks and five orphan MosquitoFusion test labels before using those auxiliary splits.
5. Run the alignment gate in a controlled environment, then reproduce/validate artifacts while keeping test data closed until final evaluation.
6. Define a read-only Bhopal output API and adapt frontend only; keep dataset generation controls out of the dashboard.
7. Preserve auxiliary experiments separately with licenses, data dictionaries, split audits and domain limitations.

## Key risks before submission

- ERI/EMSS is derived from environmental variables also used by RF1; current metrics can reflect proxy reconstruction rather than observed disease prediction.
- Bhopal sampling radius is inconsistent between config and web defaults; verify what generated stored records.
- RF2 artifact provenance and training status require confirmation; project docs state RF2 training was not run in this copy.
- Alignment and split integrity were inspected in code but not executed here.
- Climate master has 1,178 rows while splits total 1,176; two city-weeks are omitted. MosquitoFusion test labels have five orphan files.
- Resolve licensing, package versions, paths, dashboard security and representativeness before submission.

## Five-line explanation for faculty judges

1. Our main workflow links Bhopal environmental records and satellite images using a shared sample ID.
2. A Random Forest models environmental inputs while ResNet-18 extracts image features.
3. We check that both branches refer to the same samples before a second Random Forest combines them.
4. The current target is a formula-based environmental suitability score, not confirmed disease prediction.
5. Climate-lag and mosquito-breeding datasets are separate auxiliary experiments, not matched Bhopal observations.

## Audit scope

Read-only audit: no training, data downloads, notebook execution, source modification, copying or deletion. Duplicate report rows: 1259.
