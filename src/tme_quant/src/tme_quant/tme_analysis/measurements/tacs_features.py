"""
TACS-specific feature extraction for prognostic analysis.
"""

import numpy as np
from typing import Dict, Any, List
from ..config.analysis_params import InteractionPair


class TACSFeatureExtractor:
    """
    Extract TACS-specific prognostic features.
    
    Based on literature:
        - Provenzano et al. (2006): TACS-3 correlates with poor prognosis
        - Conklin et al. (2011): TACS progression predicts invasion
        - Bredfeldt et al. (2014): Quantitative TACS analysis
    """
    
    def extract_features(
        self,
        interaction_pairs: List[InteractionPair],
        fibers: List = None
    ) -> Dict[str, float]:
        """
        Extract comprehensive TACS features.
        
        Returns:
            Dictionary with TACS prognostic features
        """
        features = {}
        
        # Basic TACS counts and ratios (from Part 4)
        # ... (already implemented in MeasurementEngine)
        
        # Advanced TACS features
        
        # 1. TACS Transition Score
        # Measures presence of TACS-1 → TACS-2 → TACS-3 progression
        features['tacs_progression_score'] = self._compute_tacs_progression(
            interaction_pairs
        )
        
        # 2. Invasive Front Density
        # High density of TACS-3 at invasive front
        features['invasive_front_density'] = self._compute_invasive_front_density(
            interaction_pairs
        )
        
        # 3. Collagen Organization Index
        # Measures fiber alignment at boundary
        features['collagen_organization_index'] = self._compute_organization_index(
            interaction_pairs
        )
        
        # 4. Mechanical Stiffness Proxy
        # TACS-3 fibers contribute to tissue stiffness
        features['mechanical_stiffness_proxy'] = self._compute_stiffness_proxy(
            interaction_pairs, fibers
        )
        
        return features
    
    def _compute_tacs_progression(
        self,
        pairs: List[InteractionPair]
    ) -> float:
        """
        Compute TACS progression score.
        
        Score increases with TACS-3 presence and decreases with TACS-1.
        """
        tacs_types = [p.interaction_type for p in pairs]
        
        if not tacs_types:
            return 0.0
        
        tacs1_count = tacs_types.count('TACS-1')
        tacs2_count = tacs_types.count('TACS-2')
        tacs3_count = tacs_types.count('TACS-3')
        
        total = len(tacs_types)
        
        # Progression score: weighted by TACS type
        # TACS-1 (early) = low, TACS-3 (late) = high
        score = (tacs3_count * 1.0 - tacs1_count * 0.5 + tacs2_count * 0.5) / total
        
        return float(score)
    
    def _compute_invasive_front_density(
        self,
        pairs: List[InteractionPair]
    ) -> float:
        """Compute density of TACS-3 fibers at invasive front."""
        # Filter TACS-3 fibers close to boundary (< 20 microns)
        tacs3_close = [
            p for p in pairs
            if p.interaction_type == 'TACS-3' and p.distance < 20.0
        ]
        
        # Density relative to all boundary fibers
        total_boundary = len(pairs)
        
        if total_boundary == 0:
            return 0.0
        
        density = len(tacs3_close) / total_boundary
        
        return float(density)
    
    def _compute_organization_index(
        self,
        pairs: List[InteractionPair]
    ) -> float:
        """
        Compute collagen organization index.
        
        Higher values indicate more organized (aligned) fibers.
        """
        angles = [
            p.angle_to_boundary_normal for p in pairs
            if p.angle_to_boundary_normal is not None
        ]
        
        if not angles:
            return 0.0
        
        # Compute circular variance
        angles_rad = np.radians(angles)
        mean_cos = np.mean(np.cos(2 * angles_rad))
        mean_sin = np.mean(np.sin(2 * angles_rad))
        
        organization_index = np.sqrt(mean_cos**2 + mean_sin**2)
        
        return float(organization_index)
    
    def _compute_stiffness_proxy(
        self,
        pairs: List[InteractionPair],
        fibers: List = None
    ) -> float:
        """
        Compute proxy for mechanical stiffness.
        
        TACS-3 fibers (perpendicular) contribute to higher stiffness.
        """
        tacs3_pairs = [p for p in pairs if p.interaction_type == 'TACS-3']
        
        if not tacs3_pairs:
            return 0.0
        
        # Stiffness increases with:
        # 1. Number of TACS-3 fibers
        # 2. Fiber straightness (if available)
        # 3. Fiber density
        
        tacs3_ratio = len(tacs3_pairs) / len(pairs) if pairs else 0.0
        
        # If fiber data available, incorporate straightness
        if fibers:
            straightnesses = [
                f.straightness for f in fibers
                if hasattr(f, 'straightness') and f.straightness is not None
            ]
            
            if straightnesses:
                mean_straightness = np.mean(straightnesses)
                stiffness_proxy = tacs3_ratio * mean_straightness
                return float(stiffness_proxy)
        
        return float(tacs3_ratio)