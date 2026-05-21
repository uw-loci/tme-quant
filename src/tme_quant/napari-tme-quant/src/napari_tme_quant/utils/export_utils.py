"""Figure QDialog viewer and tabular export wrappers.

Usage:
    from napari_tme_quant.utils.export_utils import open_figure_dialog, export_df

All Qt calls must happen on the Qt main thread; callers are responsible for
ensuring that when invoking from a thread_worker returned callback.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QScrollArea,
    QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget


class FigureDialog(QDialog):
    """Lightweight QDialog that renders a matplotlib Figure as a PNG label.

    Parameters
    ----------
    fig :
        A ``matplotlib.figure.Figure`` to display.
    title :
        Window title shown in the dialog title bar.
    parent :
        Optional Qt parent widget.
    dpi :
        DPI used when rasterising the figure (default 100).
    """

    def __init__(self, fig, title: str = "Figure", parent: Optional["QWidget"] = None, dpi: int = 100) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self._build_ui(fig, dpi)

    def _build_ui(self, fig, dpi: int) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read(), "PNG")

        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

        # Size the dialog to roughly fit the figure, capped at screen size
        w = min(pixmap.width() + 40, 1400)
        h = min(pixmap.height() + 60, 900)
        self.resize(w, h)


def open_figure_dialog(fig, title: str = "Figure", parent: Optional["QWidget"] = None) -> FigureDialog:
    """Create and show a non-modal FigureDialog for *fig*.

    Returns the dialog so the caller can keep a reference and prevent garbage
    collection while the window is open.

    Parameters
    ----------
    fig :
        A ``matplotlib.figure.Figure``.
    title :
        Window title.
    parent :
        Optional Qt parent widget.
    """
    dlg = FigureDialog(fig, title=title, parent=parent)
    dlg.show()
    return dlg


class DataFrameDialog(QDialog):
    """Non-modal dialog showing a pandas DataFrame in a scrollable QTableWidget."""

    def __init__(self, df: pd.DataFrame, title: str = "Data", parent: Optional["QWidget"] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(900, 500)

        table = QTableWidget(len(df), len(df.columns))
        table.setHorizontalHeaderLabels(df.columns.tolist())
        table.verticalHeader().setVisible(False)
        for r, row in enumerate(df.itertuples(index=False)):
            for c, val in enumerate(row):
                if isinstance(val, float):
                    text = f"{val:.4g}"
                else:
                    text = str(val)
                table.setItem(r, c, QTableWidgetItem(text))
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)

        layout = QVBoxLayout(self)
        layout.addWidget(table)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.close)
        layout.addWidget(btns)


def open_dataframe_dialog(
    df: pd.DataFrame,
    title: str = "Data",
    parent: Optional["QWidget"] = None,
) -> DataFrameDialog:
    """Create and show a non-modal DataFrameDialog for *df*."""
    dlg = DataFrameDialog(df, title=title, parent=parent)
    dlg.show()
    dlg.raise_()
    return dlg


def export_df_to_csv(df: pd.DataFrame, path: "str | Path") -> None:
    """Write *df* to *path* as UTF-8 CSV (no index).

    Parameters
    ----------
    df :
        DataFrame to export.
    path :
        Destination file path.  Parent directory must already exist.
    """
    df.to_csv(path, index=False, encoding="utf-8")


def export_df_to_excel(df: pd.DataFrame, path: "str | Path") -> None:
    """Write *df* to *path* as .xlsx with auto-column-widths and a frozen header.

    Delegates to ``tme_quant.fiber_analysis.io.export_dataframe_to_excel`` so
    that formatting logic stays in the library (single implementation).

    Parameters
    ----------
    df :
        DataFrame to export.
    path :
        Destination .xlsx file path.
    """
    from tme_quant.fiber_analysis.io import export_dataframe_to_excel
    export_dataframe_to_excel(df, path)
