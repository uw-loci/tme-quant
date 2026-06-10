# API Changes

## 2026-06-10 (fire-only option)
- `curvealign_ctfire_mode_pipeline()`: added `use_ct_reconstruction: bool = True`
  parameter — when `False`, calls `fire_2d_angle()` directly on the normalised image,
  bypassing `ct_reconstruction`; **no curvelops or curvelet library required** for
  this mode, only `ctfire_py` needed

## 2026-06-10
- `tme_quant.tme_analysis.pipelines`: added `curvealign_ctfire_mode_pipeline()` — CT-FIRE individual-fiber pipeline returning `CTFirePipelineResult`
- `tme_quant.tme_analysis.pipelines`: added `CTFirePipelineResult` dataclass — typed return value with `fiber_structure`, `fiber_features_df`, `density_df`, `alignment_df`, `roi_measurements_df`, `roi_summary_df`, `in_curvs_flag`, `nearest_angles`, `boundary_measurement`, `roi_coordinates`, `params`
- `tme_quant` (public API): re-exported `curvealign_ctfire_mode_pipeline` and `CTFirePipelineResult`
