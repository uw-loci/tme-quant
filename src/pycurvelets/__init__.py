try:
    from .new_curv import new_curv  # type: ignore

    HAS_CURVELETS = True
except Exception:
    HAS_CURVELETS = False

__all__ = [
    "HAS_CURVELETS",
    "new_curv",
]
