# Reproducibility boundary

This release is a source reconstruction and packaging update for the 960-clip JoVE workflow, not the later 1500-clip method experiments. Original exact library versions, original weight-file checksum, timing and peak RAM were not recovered. requirements.txt gives compatible constraints for a new environment; record the actual resolved versions with capture_environment.py and pip freeze.

At least two historical Test prediction sets existed. The archived statement test_evaluated_once=true is not evidence of an untouched Test. The public final-evaluation entry point corrects that metadata and identifies new output as retrospective reproduction. Its class-stratified clip-bootstrap intervals are legacy output, NOT the revised manuscript participant-cluster intervals. No claim of participant uncertainty is made by that script.

New packaging utilities: run.py, prepare_dataset.py, tests/test_smoke.py and GitHub documentation. daisee_mean_validation_only.py is a revision helper; it is not an original historical script. Other recovered script SHA-256 values before packaging edits are recorded in source_inventory.json. The final-evaluation metadata/docstring was corrected for retrospective status, explicit split and label alignment checks were added; fitting choices are unchanged. Do not use the old baseline train/all modes as the final method.

Reported historical macro-F1 (not a freshly reproduced result): Boredom 0.2792876098; Engagement 0.2564401761; Confusion 0.3129498472; Frustration 0.2013557006. No raw-video training or exact score reproduction was performed while building this release. Archived per-clip results and statistical inputs are not redistributed here. verify_archived_predictions.py and replay_validation_selection.py can be used only when the corresponding authorized artifacts are independently available.

The subset uses joint label strata, so it is a compact diagnostic subset rather than an unbiased substitute for the full DAiSEE benchmark. Participant IDs are inferred from the first six filename characters; this is not an independent visual identity audit.
