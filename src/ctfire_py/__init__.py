from .ct_fire import load_ctfire_params

__all__ = ["load_ctfire_params"]

# Check for curvelops backend
try:
    from curvelops import fdct2d_wrapper
    HAS_CURVELOPS = True
except ImportError:
    HAS_CURVELOPS = False

# Check for C++ fiber backend
try:
    from .fire_2d_angle import fiber_backend
    HAS_FIBER_BACKEND = fiber_backend is not None
except ImportError:
    HAS_FIBER_BACKEND = False

if HAS_FIBER_BACKEND:
    from .fire_2d_angle import fire_2d_angle
    __all__.append("fire_2d_angle")

if HAS_CURVELOPS:
    from .ct_reconstruction import ct_reconstruction
    __all__.append("ct_reconstruction")

if HAS_CURVELOPS and HAS_FIBER_BACKEND:
    from .ct_fire import ct_fire
    __all__.append("ct_fire")
