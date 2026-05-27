"""Launch napari with the TMEQuant dock widget open.

Run via VS Code: select "Launch napari (TMEQuant plugin, open dock widget)"
Run via terminal: python src/tme_quant/napari-tme-quant/scripts/launch_plugin.py
"""

import napari
from napari_tme_quant._main_widget import TMEQuantDockWidget

viewer = napari.Viewer()
widget = TMEQuantDockWidget(viewer)
viewer.window.add_dock_widget(widget, name="TMEQuant", area="right")

napari.run()
