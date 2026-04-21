# -*- coding: utf-8 -*-
"""Tests for fiber_analysis/io.py — export_dataframe_to_excel."""

import numpy as np
import pandas as pd
import pytest

openpyxl = pytest.importorskip("openpyxl", reason="openpyxl not installed")

from tme_quant.fiber_analysis.io import export_dataframe_to_excel


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_df(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "fiber_id":   [f"f{i}" for i in range(5)],
        "center_row": rng.uniform(0, 100, 5),
        "angle":      rng.uniform(0, 180, 5),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExportDataframeToExcel:

    def test_creates_file(self, tmp_path):
        out = tmp_path / "out.xlsx"
        export_dataframe_to_excel(_make_df(), str(out))
        assert out.exists()

    def test_data_round_trips(self, tmp_path):
        df = _make_df()
        out = tmp_path / "out.xlsx"
        export_dataframe_to_excel(df, str(out))
        result = pd.read_excel(out, sheet_name="Sheet1")
        assert list(result.columns) == list(df.columns)
        assert len(result) == len(df)

    def test_header_frozen(self, tmp_path):
        out = tmp_path / "out.xlsx"
        export_dataframe_to_excel(_make_df(), str(out))
        wb = openpyxl.load_workbook(str(out))
        ws = wb["Sheet1"]
        assert ws.freeze_panes == "A2"

    def test_column_widths_positive(self, tmp_path):
        out = tmp_path / "out.xlsx"
        export_dataframe_to_excel(_make_df(), str(out))
        wb = openpyxl.load_workbook(str(out))
        ws = wb["Sheet1"]
        for col_dim in ws.column_dimensions.values():
            assert col_dim.width > 0

    def test_custom_sheet_name(self, tmp_path):
        out = tmp_path / "out.xlsx"
        export_dataframe_to_excel(_make_df(), str(out), sheet_name="Fibers")
        wb = openpyxl.load_workbook(str(out))
        assert "Fibers" in wb.sheetnames

    def test_append_mode_adds_sheet(self, tmp_path):
        out = tmp_path / "out.xlsx"
        export_dataframe_to_excel(_make_df(seed=0), str(out), sheet_name="Sheet1")
        export_dataframe_to_excel(_make_df(seed=1), str(out), sheet_name="Sheet2", mode="a")
        wb = openpyxl.load_workbook(str(out))
        assert "Sheet1" in wb.sheetnames
        assert "Sheet2" in wb.sheetnames

    def test_append_mode_replaces_sheet(self, tmp_path):
        out = tmp_path / "out.xlsx"
        df1 = _make_df(seed=0)
        df2 = _make_df(seed=1)
        export_dataframe_to_excel(df1, str(out), sheet_name="Sheet1")
        export_dataframe_to_excel(df2, str(out), sheet_name="Sheet1", mode="a")
        result = pd.read_excel(out, sheet_name="Sheet1")
        # The second write replaces the first; compare float columns
        np.testing.assert_allclose(
            result["center_row"].values, df2["center_row"].values, rtol=1e-5
        )

    def test_empty_dataframe(self, tmp_path):
        out = tmp_path / "out.xlsx"
        empty = pd.DataFrame(columns=["center_row", "angle"])
        export_dataframe_to_excel(empty, str(out))
        assert out.exists()
        wb = openpyxl.load_workbook(str(out))
        assert "Sheet1" in wb.sheetnames
