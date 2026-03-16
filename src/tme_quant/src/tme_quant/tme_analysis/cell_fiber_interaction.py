"""
cell_fiber_interaction.py — RETIRED.

This module has been superseded.  All functionality has been migrated to the
following canonical locations:

  Interaction detection
  ---------------------
  tme_analysis.core.interaction_detector.InteractionDetector
    detect_cell_fiber_interactions()
    detect_cell_cell_interactions()
    detect_fiber_tumor_interactions()

  Measurement & feature extraction
  ---------------------------------
  tme_analysis.core.measurement_engine.MeasurementEngine
    compute_tacs_features()             ← replaces _calculate_tacs_scores()
    compute_mechanical_features()       ← replaces _calculate_mechanical_features()
    compute_contact_pattern_features()  ← replaces _analyze_contact_patterns()
    compute_composite_prognostic_scores() ← replaces _calculate_prognostic_scores()
    compute_morphological_features()
    compute_spatial_features()
    compute_orientation_features()
    compute_density_features()
    compute_prognostic_scores()
    compute_distance_maps()

  TACS classification
  -------------------
  tme_analysis.core.tacs_classifier.classify_fiber_tacs()

This file is kept only so that any existing import of InteractionAnalyzer
raises an informative ImportError rather than a confusing AttributeError.
"""

import warnings


class InteractionAnalyzer:  # noqa: N801
    """Retired — use InteractionDetector + MeasurementEngine instead."""

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "InteractionAnalyzer has been retired.  "
            "Use tme_analysis.core.interaction_detector.InteractionDetector "
            "for interaction detection and "
            "tme_analysis.core.measurement_engine.MeasurementEngine "
            "for feature extraction."
        )