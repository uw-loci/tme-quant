import napari
viewer = napari.Viewer()
viewer.window.console
# In console:
from napari_plugin_engine import plugin_manager
plugin_manager.discoverable_plugins
# Should list 'napari-curvealign'