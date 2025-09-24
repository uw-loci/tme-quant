"""
CT-FIRE configuration options.

This module defines the primary configuration class for controlling
CT-FIRE analysis behavior and parameters.
"""

from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class CTFireOptions:
    """
    Configuration options for CT-FIRE analysis.
    
    Based on the parameters in ctFIRE_1.m and ctFIRE.m control panels.
    
    Parameters
    ----------
    run_mode : {"ctfire", "fire", "both"}, default "ctfire"
        Processing mode: ctfire only, FIRE only, or both
    sigma_im : float, default 1.0
        Gaussian smoothing sigma for image preprocessing
    thresh_im : float, optional
        Image threshold as fraction of max intensity
    thresh_im2 : float, optional
        Absolute threshold value
    dtype : {"euclidean", "cityblock"}, default "euclidean"
        Distance transform type
    thresh_linka : float, default 0.5
        Angle threshold for linking fibers
    thresh_linkd : float, default 10.0
        Distance threshold for linking fibers across gaps
    thresh_flen : float, default 30.0
        Minimum fiber length threshold
    thresh_numv : int, default 5
        Minimum number of vertices per fiber
    s_fiberdir : float, default 0.5
        Fiber direction smoothing parameter
    line_width : float, default 0.5
        Line width for fiber visualization
    length_limit : float, default 30.0
        Length limit for display (only show fibers > limit)
    fiber_number_limit : int, default 9999
        Maximum number of fibers to display
    """
    run_mode: Literal["ctfire", "fire", "both"] = "ctfire"
    sigma_im: float = 1.0
    thresh_im: Optional[float] = None
    thresh_im2: Optional[float] = None
    dtype: Literal["euclidean", "cityblock"] = "euclidean"
    thresh_linka: float = 0.5
    thresh_linkd: float = 10.0
    thresh_flen: float = 30.0
    thresh_numv: int = 5
    s_fiberdir: float = 0.5
    line_width: float = 0.5
    length_limit: float = 30.0
    fiber_number_limit: int = 9999
    
    # FDCT parameters for curvelet enhancement
    keep: float = 0.001
    scale: Optional[int] = None
    
    # Output control
    plot_fibers: bool = True
    plot_nonfibers: bool = False
    save_outputs: bool = True
    output_format: Literal["csv", "xlsx"] = "csv"
