"""
Fiber Angle Calculation

Calculates angles of fibers at individual points along each fiber.
"""

import numpy as np
from typing import List, Dict


def calc_fiberang2(
    X: np.ndarray,
    F: List[Dict],
    k: int
) -> List[Dict]:
    """
    Calculate the angles of fibers at each point.
    
    Computes tangent angles at regular intervals along each fiber.
    The angle is calculated using k consecutive vertices.
    
    Parameters
    ----------
    X : np.ndarray
        Vertex coordinates, shape (N, 2) or (N, 3)
    F : list of dict
        Fiber structures, each with 'v' field containing vertex indices
    k : int
        Number of lag points - the number of consecutive vertices
        used for tangent angle calculation. Typical value is 3-5.
        
    Returns
    -------
    Fang : list of dict
        Fiber angle structures, each containing:
        - angle_xz : np.ndarray
            Angles in xz plane at each point
        - angle_xy : np.ndarray
            Angles in xy plane at each point
            
    Notes
    -----
    For fibers shorter than k vertices, the angle is calculated
    using the start and end points and replicated for all points.
    
    For longer fibers, angles are calculated at intervals of k vertices,
    and the last k points use the same angle as the previous calculation.
    
    Angles are computed as atan(dy/dx) — single-argument arctangent,
    range [-π/2, π/2] — matching MATLAB's `atan(...)` in calc_fiberang2.m.
    This differs from atan2 which returns [-π, π]: fibers are undirected
    so the sign of direction should not affect the angle.
    """
    Fang = []
    eps = np.finfo(float).eps

    def _atan_ratio(dy: float, dx: float) -> float:
        """atan(dy / (dx + eps)), matching MATLAB atan((x2-x1)/(x1-x2+eps))."""
        return float(np.arctan(dy / (dx + eps)))

    for fi in range(len(F)):
        fv = F[fi]['v']
        Lf = len(fv)
        
        fang = {
            'angle_xz': [],
            'angle_xy': []
        }
        
        if Lf <= k:
            # Fiber is too short, use start and end points
            v1 = fv[0]
            v2 = fv[-1]
            
            x1 = X[v1, :]
            x2 = X[v2, :]
            
            # MATLAB: atan((x2(:,3)-x1(:,3)) ./ (x2(:,1)-x1(:,1)+eps))
            angxz = _atan_ratio(x2[2] - x1[2], x2[0] - x1[0]) if X.shape[1] >= 3 else 0.0
            angxy = _atan_ratio(x2[1] - x1[1], x2[0] - x1[0])
            
            # Replicate angle for all points
            fang['angle_xz'] = np.full(k, angxz)
            fang['angle_xy'] = np.full(k, angxy)
            
        else:
            # Fiber is long enough, calculate angles at intervals
            angle_xz = []
            angle_xy = []
            
            for j in range(Lf - k):
                # Calculate angle orientation at point j
                v1 = fv[j]
                v2 = fv[j + k]
                
                x1 = X[v1, :]
                x2 = X[v2, :]
                
                # MATLAB: atan((x2(:,3)-x1(:,3)) ./ (x2(:,1)-x1(:,1)+eps))
                angxz = _atan_ratio(x2[2] - x1[2], x2[0] - x1[0]) if X.shape[1] >= 3 else 0.0
                angxy = _atan_ratio(x2[1] - x1[1], x2[0] - x1[0])
                
                angle_xz.append(angxz)
                angle_xy.append(angxy)
            
            # The last k points have the same angle as the last calculation
            # MATLAB: Fang(fi).angle_xz(j+1:Lf) = repmat(Fang(fi).angle_xz(j), 1, k)
            if len(angle_xz) > 0:
                angle_xz.extend([angle_xz[-1]] * k)
                angle_xy.extend([angle_xy[-1]] * k)
            
            fang['angle_xz'] = np.array(angle_xz)
            fang['angle_xy'] = np.array(angle_xy)
        
        Fang.append(fang)
    
    return Fang
