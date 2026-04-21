# -*- coding: utf-8 -*-
"""Tests for core/exceptions.py — tme_quant exception hierarchy."""

import pytest

from tme_quant.core.exceptions import (
    FiberAnalysisError,
    ROIProcessingError,
    BoundaryAnalysisError,
    FeatureExtractionError,
    ImageProcessingError,
)


class TestExceptionHierarchy:

    def test_base_is_exception(self):
        assert issubclass(FiberAnalysisError, Exception)

    def test_all_subclass_base(self):
        for cls in (
            ROIProcessingError,
            BoundaryAnalysisError,
            FeatureExtractionError,
            ImageProcessingError,
        ):
            assert issubclass(cls, FiberAnalysisError)

    def test_each_is_raiseable(self):
        for cls in (
            FiberAnalysisError,
            ROIProcessingError,
            BoundaryAnalysisError,
            FeatureExtractionError,
            ImageProcessingError,
        ):
            with pytest.raises(cls):
                raise cls("test message")

    def test_subclass_caught_by_base(self):
        with pytest.raises(FiberAnalysisError):
            raise ROIProcessingError("roi failed")

    def test_public_api_exports(self):
        import tme_quant
        for name in (
            "FiberAnalysisError",
            "ROIProcessingError",
            "BoundaryAnalysisError",
            "FeatureExtractionError",
            "ImageProcessingError",
        ):
            assert hasattr(tme_quant, name)
