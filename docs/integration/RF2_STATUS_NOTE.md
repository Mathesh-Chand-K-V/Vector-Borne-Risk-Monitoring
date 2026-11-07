# RF2 Status Note

- The RF2 model, validation predictions, test predictions and metrics exist under `Vector_Borne_Risk_Project/artifacts/` and `Vector_Borne_Risk_Project/reports/`.
- The ML team confirmed RF2 was trained after the latest revised scripts were used to generate the updated metrics.
- The statement in `Vector_Borne_Risk_Project/README.md` that RF2 training has not yet been run is outdated. Correct it in the future final merged project; the source README is intentionally unchanged by this update.
- Artifact provenance remains incomplete. Before exhibition/release, create a run manifest that binds the RF2 model and outputs to checksums for model/data/code, Python and package versions, target/formula version, split version or split hashes, feature schema, and training/evaluation run date. Include the run configuration and hardware/environment details where available.
- The read-only alignment gate was authorized and passed for all three existing splits in this audit. No RF2 training or model loading was performed here.
