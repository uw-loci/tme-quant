# Moved to tme_quant.integrations.fiji_bridge.
# This shim re-exports the public API for backward import compatibility.
from tme_quant.integrations.fiji_bridge import (  # noqa: F401
    FijiBackendMixin,
    FijiBridge,
    OrientationJBridge,
    RidgeDetectionBridge,
    normalise_image,
    contrast_value,
    contrast_to_fraction,
    make_color_survey,
    order_by_nearest_neighbour,
)
