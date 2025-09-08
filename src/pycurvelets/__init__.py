# Optional import: allow the package to be imported without native curvelet deps.
try:
    from .new_curv import new_curv  # type: ignore
    HAS_CURVELETS = True
except Exception:
    HAS_CURVELETS = False

__all__ = ["new_curv", "HAS_CURVELETS"]


