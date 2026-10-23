"""Modal dialogs and small Qt models for CurveAlign (tables, advanced params).

Kept separate from the main dock widget, similar to how `napari-imagej`_ splits UI pieces
under ``napari_imagej/widgets/``.

.. _napari-imagej: https://github.com/imagej/napari-imagej/tree/main/src/napari_imagej/widgets
"""

from __future__ import annotations

from enum import Enum
import pandas as pd
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QTableView,
    QAbstractScrollArea,
    QHeaderView,
)
from qtpy.QtCore import Qt, QAbstractTableModel
from qtpy.QtGui import QColor

__all__ = [
    "AdvancedParametersDialog",
    "BoundaryType",
    "MetricsDialog",
    "ResultsDialog",
    "ResultsTableModel",
]


class BoundaryType(Enum):
    """How to treat image boundaries when running CurveAlign-style analysis.

    Mirrors MATLAB-style boundary handling options exposed in the main tab.
    """

    NO_BOUNDARY = "No boundary"
    TIFF_BOUNDARY = "Tiff boundary"


class AdvancedParametersDialog(QDialog):
    """Modal dialog for extra numeric parameters passed through to the analysis backend."""

    def __init__(self, parent=None):
        """Build spin boxes for advanced parameters and OK/Cancel actions.

        Parameters
        ----------
        parent : QWidget or None
            Optional parent for window modality and lifetime.
        """
        super().__init__(parent)
        self.setWindowTitle("Advanced Parameters")

        layout = QVBoxLayout()

        self.param1 = QDoubleSpinBox()
        self.param1.setRange(0.0, 1.0)
        self.param1.setSingleStep(0.01)
        self.param1.setValue(0.1)
        param1_layout = QHBoxLayout()
        param1_layout.addWidget(QLabel("Advanced Param 1:"))
        param1_layout.addWidget(self.param1)
        layout.addLayout(param1_layout)

        self.param2 = QDoubleSpinBox()
        self.param2.setRange(0.0, 100.0)
        self.param2.setSingleStep(1.0)
        self.param2.setValue(5.0)
        param2_layout = QHBoxLayout()
        param2_layout.addWidget(QLabel("Advanced Param 2:"))
        param2_layout.addWidget(self.param2)
        layout.addLayout(param2_layout)

        self.param3 = QSpinBox()
        self.param3.setRange(1, 100)
        self.param3.setValue(10)
        param3_layout = QHBoxLayout()
        param3_layout.addWidget(QLabel("Iterations:"))
        param3_layout.addWidget(self.param3)
        layout.addLayout(param3_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_parameters(self):
        """Return the current advanced-parameter values as a plain dict.

        Returns
        -------
        dict
            Keys ``advanced_param1``, ``advanced_param2``, and ``iterations`` with
            numeric values suitable for the main widget's ``advanced_params`` dict.
        """
        return {
            "advanced_param1": self.param1.value(),
            "advanced_param2": self.param2.value(),
            "iterations": self.param3.value(),
        }


class ResultsTableModel(QAbstractTableModel):
    """Qt model backing :class:`QTableView` for a rectangular :class:`pandas.DataFrame`."""

    def __init__(self, data, parent=None):
        """Store the dataframe and derive column headers from :attr:`data.columns`.

        Parameters
        ----------
        data : pandas.DataFrame
            Table contents; must not be mutated externally without calling layoutChanged.
        parent : QObject or None
            Parent for Qt object ownership.
        """
        super().__init__(parent)
        self._data = data
        self._headers = list(data.columns) if data is not None else []

    def rowCount(self, parent=None):
        """Return the number of data rows (Qt table model API)."""
        return len(self._data)

    def columnCount(self, parent=None):
        """Return the number of columns from the dataframe header list."""
        return len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        """Return cell text for ``DisplayRole`` or a light striping color for ``BackgroundRole``."""

        if not index.isValid():
            return None

        row = index.row()
        col = index.column()

        if role == Qt.DisplayRole:
            return str(self._data.iloc[row, col])

        if role == Qt.BackgroundRole and row % 2 == 0:
            return QColor(240, 240, 240)

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Return column names for the horizontal header or 1-based row indices for the vertical header."""

        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return self._headers[section]
        return str(section + 1)


class ResultsDialog(QDialog):
    """Read-only dialog showing a dataframe in a :class:`QTableView` with an OK button."""

    def __init__(self, data, parent=None):
        """Attach a :class:`ResultsTableModel` to a stretched table view.

        Parameters
        ----------
        data : pandas.DataFrame
            Rows and columns to display.
        parent : QWidget or None
            Optional parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Analysis Results")
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout()

        self.table_view = QTableView()
        self.table_view.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.model = ResultsTableModel(data)
        self.table_view.setModel(self.model)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)

        layout.addWidget(self.table_view)
        layout.addWidget(button_box)

        self.setLayout(layout)


class MetricsDialog(QDialog):
    """Tabular ROI measurements with an extra action to export the underlying dataframe to CSV."""

    def __init__(self, data: pd.DataFrame, parent=None):
        """Same layout as :class:`ResultsDialog` but titled for measurements and with CSV export.

        Parameters
        ----------
        data : pandas.DataFrame
            Per-ROI or per-metric table to show and optionally export.
        parent : QWidget or None
            Optional parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("ROI Measurements")
        self.df = data

        layout = QVBoxLayout()
        self.table_view = QTableView()
        self.table_view.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.model = ResultsTableModel(data)
        self.table_view.setModel(self.model)
        layout.addWidget(self.table_view)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        export_btn = button_box.addButton("Export CSV", QDialogButtonBox.ActionCommandRole)
        button_box.rejected.connect(self.reject)
        export_btn.clicked.connect(self._export_csv)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def _export_csv(self):
        """Save :attr:`df` to a user-chosen path using :meth:`pandas.DataFrame.to_csv`."""
        path, _ = QFileDialog.getSaveFileName(self, "Export Measurements", "", "CSV Files (*.csv)")
        if path:
            self.df.to_csv(path, index=False)
