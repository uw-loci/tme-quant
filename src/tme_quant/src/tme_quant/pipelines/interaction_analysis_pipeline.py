"""
InteractionAnalysisPipeline — end-to-end interaction analysis workflow.

Orchestrates:
  1. Interaction detection  (InteractionDetector)
  2. Per-pair feature annotation  (annotate_interaction_pairs)
  3. Feature extraction  (MeasurementEngine)
  4. Result export  (TMEAnalysisExporter)

Supports both single-image and batch modes.

Example
-------
>>> from tme_quant.pipelines.interaction_analysis_pipeline import (
...     InteractionAnalysisPipeline, PipelineConfig,
... )
>>> cfg = PipelineConfig(boundary_distance=100.0, tacs_zone_width=100.0)
>>> pipeline = InteractionAnalysisPipeline(cfg)
>>> result = pipeline.run(cells=cells, fibers=fibers, tumors=tumors)
>>> result.to_csv('output/')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.tme_models.cell_model import CellObject
from ..core.tme_models.fiber_model import FiberObject
from ..core.tme_models.tumor_model import TumorRegion
from ..tme_analysis.core.interaction_detector import InteractionDetector
from ..tme_analysis.core.measurement_engine import MeasurementEngine
from ..tme_analysis.config.analysis_params import InteractionPair
from ..measurement.interaction_features import annotate_interaction_pairs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Runtime configuration for InteractionAnalysisPipeline.

    All distance thresholds are in micrometres (µm).
    """
    # Detection thresholds
    boundary_distance: float = 100.0    # max cell-fiber interaction distance
    tacs_zone_width: float = 100.0      # TACS classification zone around boundary
    contact_threshold: float = 5.0      # physical contact distance

    # Feature flags
    compute_tacs: bool = True
    compute_morphology: bool = True
    compute_spatial: bool = True
    compute_orientation: bool = True
    compute_density: bool = True
    compute_mechanical: bool = True
    compute_contact_patterns: bool = True
    compute_prognostic: bool = True

    # Verbose output
    verbose: bool = False

    # Export settings
    export_dir: Optional[Path] = None
    export_formats: List[str] = field(default_factory=lambda: ['csv', 'json'])


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Holds all outputs from a single pipeline run."""
    image_id: str
    interaction_pairs: List[InteractionPair]
    features: Dict[str, Any]
    prognostic_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Return a flat summary dict suitable for logging or CSV export."""
        out: Dict[str, Any] = {
            'image_id':           self.image_id,
            'n_interactions':     len(self.interaction_pairs),
        }
        out.update(self.prognostic_scores)
        # Flatten scalar features
        for k, v in self.features.items():
            if isinstance(v, (int, float, str, bool)):
                out[k] = v
        return out

    def to_csv(self, output_dir: Path) -> Path:
        """Write summary row as a CSV file. Returns the path written."""
        import csv
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f'{self.image_id}_interaction_summary.csv'
        row = self.summary()
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class InteractionAnalysisPipeline:
    """
    End-to-end pipeline: detection → annotation → measurement → export.

    Parameters
    ----------
    config:
        PipelineConfig instance.  If None, defaults are used.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.detector = InteractionDetector(verbose=self.config.verbose)
        self.engine   = MeasurementEngine(verbose=self.config.verbose)

    # ------------------------------------------------------------------
    # Single-image entry point
    # ------------------------------------------------------------------

    def run(
        self,
        cells: List[CellObject],
        fibers: List[FiberObject],
        tumors: Optional[List[TumorRegion]] = None,
        region_area: Optional[float] = None,
        image_id: str = 'image',
    ) -> PipelineResult:
        """
        Run the full pipeline for a single image / region.

        Parameters
        ----------
        cells:
            Segmented cell objects.
        fibers:
            Extracted fiber objects with TACS angles pre-computed.
        tumors:
            Tumor region objects (required for TACS / fiber-tumor interactions).
        region_area:
            Area of the analysis region in µm² (used for density features).
        image_id:
            Identifier stored in the result.

        Returns
        -------
        PipelineResult
        """
        cfg = self.config

        # ---- Step 1: detect interactions ----
        all_pairs: List[InteractionPair] = []

        if cfg.verbose:
            print(f'[{image_id}] Detecting cell-fiber interactions …')
        cf_pairs = self.detector.detect_cell_fiber_interactions(
            cells=cells,
            fibers=fibers,
            max_distance=cfg.boundary_distance,
        )
        all_pairs.extend(cf_pairs)

        if tumors:
            if cfg.verbose:
                print(f'[{image_id}] Detecting fiber-tumor interactions …')
            ft_pairs = self.detector.detect_fiber_tumor_interactions(
                fibers=fibers,
                tumor_regions=tumors,
                boundary_distance=cfg.tacs_zone_width,
            )
            all_pairs.extend(ft_pairs)

        if cfg.verbose:
            print(f'[{image_id}] {len(all_pairs)} interaction pairs detected.')

        # ---- Step 2: annotate pairs with per-interaction features ----
        annotate_interaction_pairs(
            pairs=all_pairs,
            fibers=fibers,
            cells=cells,
            contact_threshold=cfg.contact_threshold,
            tacs_zone_width=cfg.tacs_zone_width,
        )

        # ---- Step 3: extract aggregate features ----
        features: Dict[str, Any] = {}

        if cfg.compute_tacs and all_pairs:
            features.update(self.engine.compute_tacs_features(all_pairs))

        if cfg.compute_morphology and all_pairs:
            features.update(
                self.engine.compute_morphological_features(cells, fibers, all_pairs)
            )

        if cfg.compute_spatial and all_pairs:
            features.update(
                self.engine.compute_spatial_features(cells, fibers, all_pairs)
            )

        if cfg.compute_orientation and all_pairs:
            features.update(self.engine.compute_orientation_features(all_pairs))

        if cfg.compute_density:
            features.update(
                self.engine.compute_density_features(cells, fibers, region_area)
            )

        if cfg.compute_mechanical and all_pairs:
            features.update(
                self.engine.compute_mechanical_features(all_pairs, fibers)
            )

        if cfg.compute_contact_patterns and all_pairs:
            features.update(
                self.engine.compute_contact_pattern_features(
                    all_pairs, region_area=region_area
                )
            )

        # ---- Step 4: composite prognostic scores ----
        prognostic: Dict[str, float] = {}
        if cfg.compute_prognostic:
            prognostic.update(
                self.engine.compute_composite_prognostic_scores(features)
            )

        result = PipelineResult(
            image_id=image_id,
            interaction_pairs=all_pairs,
            features=features,
            prognostic_scores=prognostic,
            metadata={
                'n_cells':  len(cells),
                'n_fibers': len(fibers),
                'n_tumors': len(tumors) if tumors else 0,
                'config':   self.config.__dict__,
            },
        )

        # ---- Step 5: export if configured ----
        if cfg.export_dir:
            result.to_csv(cfg.export_dir)

        return result

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------

    def run_batch(
        self,
        samples: List[Dict[str, Any]],
        export_dir: Optional[Path] = None,
    ) -> List[PipelineResult]:
        """
        Run the pipeline on multiple images.

        Parameters
        ----------
        samples:
            List of dicts, each with keys:
              ``cells``, ``fibers``, optionally ``tumors``,
              ``region_area``, ``image_id``.
        export_dir:
            If provided, each result is exported here.

        Returns
        -------
        List of PipelineResult, one per sample.
        """
        results = []
        for i, sample in enumerate(samples):
            image_id = sample.get('image_id', f'sample_{i}')
            if self.config.verbose:
                print(f'Batch: processing {image_id} ({i+1}/{len(samples)}) …')

            result = self.run(
                cells=sample.get('cells', []),
                fibers=sample.get('fibers', []),
                tumors=sample.get('tumors'),
                region_area=sample.get('region_area'),
                image_id=image_id,
            )

            if export_dir:
                result.to_csv(export_dir)

            results.append(result)

        if self.config.verbose:
            print(f'Batch complete: {len(results)} samples processed.')

        return results

    def generate_batch_report(
        self,
        results: List[PipelineResult],
    ) -> Dict[str, Any]:
        """
        Aggregate statistics across all samples in a batch run.

        Returns a dict with per-feature mean ± std across samples and
        an overall cohort-level prognostic score summary.
        """
        if not results:
            return {}

        all_summaries = [r.summary() for r in results]
        numeric_keys = {
            k for s in all_summaries for k, v in s.items()
            if isinstance(v, (int, float)) and k != 'n_interactions'
        }

        report: Dict[str, Any] = {
            'n_samples': len(results),
            'total_interactions': sum(len(r.interaction_pairs) for r in results),
        }
        for key in sorted(numeric_keys):
            vals = [s[key] for s in all_summaries if key in s]
            if vals:
                report[f'{key}_mean'] = float(np.mean(vals))
                report[f'{key}_std']  = float(np.std(vals))

        return report


__all__ = [
    'PipelineConfig',
    'PipelineResult',
    'InteractionAnalysisPipeline',
]