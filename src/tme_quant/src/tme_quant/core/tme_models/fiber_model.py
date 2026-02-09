# fiber_model.py
"""
Fiber-related data models with comprehensive fiber types and networks
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from enum import Enum, auto
from scipy.interpolate import splprep, splev
from ..base_models import TMEObject, Geometry, Measurement

class FiberType(Enum):
    """Fiber type classification"""
    COLLAGEN = "collagen"
    RETICULAR = "reticular"
    ELASTIC = "elastic"
    MUSCLE = "muscle"
    NERVE = "nerve"
    UNKNOWN = "unknown"

class CollagenType(Enum):
    """Collagen fiber subtypes"""
    TYPE_I = "type_i"
    TYPE_II = "type_ii"
    TYPE_III = "type_iii"
    TYPE_IV = "type_iv"
    TYPE_V = "type_v"

@dataclass
class Fiber(TMEObject):
    """Base fiber model with comprehensive properties"""
    points: np.ndarray  # Nx3 array of points along the fiber
    fiber_type: FiberType = FiberType.UNKNOWN
    thickness: float = 1.0
    intensity_profile: Optional[np.ndarray] = None
    
    # Geometric properties
    curvature: Optional[np.ndarray] = None
    torsion: Optional[np.ndarray] = None
    orientation: Optional[float] = None  # Average orientation in degrees
    
    # Structural properties
    waviness: float = 0.0
    straightness: float = 1.0
    branching_points: List[np.ndarray] = field(default_factory=list)
    
    # Measurements
    measurements: List[Measurement] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize fiber object"""
        super().__post_init__()
        self.type = TMEType.FIBER
        self.properties['object_type'] = 'fiber'
        
        # Calculate derived properties
        self._calculate_geometry()
    
    @property
    def length(self) -> float:
        """Calculate fiber length"""
        if len(self.points) < 2:
            return 0.0
        
        # Calculate cumulative distance along fiber
        diffs = np.diff(self.points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return float(np.sum(distances))
    
    @property
    def centroid(self) -> np.ndarray:
        """Calculate fiber centroid"""
        return np.mean(self.points, axis=0)
    
    def _calculate_geometry(self) -> None:
        """Calculate geometric properties of the fiber"""
        if len(self.points) < 3:
            return
        
        # Calculate tangent vectors
        tangents = np.diff(self.points, axis=0)
        tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
        
        # Calculate curvature
        if len(tangents) > 1:
            tangent_changes = np.diff(tangents, axis=0)
            self.curvature = np.linalg.norm(tangent_changes, axis=1)
        
        # Calculate average orientation
        if len(tangents) > 0:
            # Use first principal component
            cov_matrix = np.cov(tangents.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            principal_component = eigenvectors[:, np.argmax(eigenvalues)]
            self.orientation = np.arctan2(principal_component[1], principal_component[0])
            self.orientation = np.degrees(self.orientation) % 180
        
        # Calculate waviness (ratio of actual length to straight line distance)
        straight_distance = np.linalg.norm(self.points[-1] - self.points[0])
        if straight_distance > 0:
            self.waviness = self.length / straight_distance
            self.straightness = 1.0 / self.waviness
    
    def resample(self, num_points: int = 100) -> np.ndarray:
        """Resample fiber with uniform spacing"""
        if len(self.points) < 2:
            return self.points
        
        # Calculate cumulative distance
        diffs = np.diff(self.points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        cumulative_distances = np.cumsum(distances)
        cumulative_distances = np.insert(cumulative_distances, 0, 0)
        
        # Normalize to [0, 1]
        normalized_distances = cumulative_distances / cumulative_distances[-1]
        
        # Interpolate
        resampled_points = []
        for dim in range(self.points.shape[1]):
            interp_func = np.interp(
                np.linspace(0, 1, num_points),
                normalized_distances,
                self.points[:, dim]
            )
            resampled_points.append(interp_func)
        
        return np.column_stack(resampled_points)
    
    def get_orientation_at_point(self, point_index: int) -> Optional[float]:
        """Get fiber orientation at specific point"""
        if len(self.points) < 2:
            return None
        
        if point_index >= len(self.points) - 1:
            point_index = len(self.points) - 2
        
        # Calculate tangent at point
        tangent = self.points[point_index + 1] - self.points[point_index]
        if np.linalg.norm(tangent) > 0:
            tangent = tangent / np.linalg.norm(tangent)
            orientation = np.arctan2(tangent[1], tangent[0])
            return np.degrees(orientation) % 180
        
        return None
    
    def to_geometry(self) -> Geometry:
        """Convert fiber to geometry object"""
        return Geometry(
            type=GeometryType.PATH,
            coordinates=self.points,
            is_3d=self.points.shape[1] >= 3
        )

@dataclass
class CollagenFiber(Fiber):
    """Collagen fiber specialization"""
    collagen_type: CollagenType = CollagenType.TYPE_I
    crosslinking_density: float = 0.0
    birefringence: float = 0.0
    maturation_state: str = ""  # immature, mature, degraded
    
    def __post_init__(self):
        """Initialize as collagen fiber"""
        super().__post_init__()
        self.fiber_type = FiberType.COLLAGEN
        self.properties['collagen_type'] = self.collagen_type.value

@dataclass
class FiberNetwork(TMEObject):
    """Network of interconnected fibers"""
    fibers: List[Fiber] = field(default_factory=list)
    adjacency_matrix: Optional[np.ndarray] = None
    network_density: float = 0.0
    connectivity: Dict[str, List[str]] = field(default_factory=dict)  # fiber_id: [connected_fiber_ids]
    
    def __post_init__(self):
        """Initialize fiber network"""
        super().__post_init__()
        self.type = TMEType.HIERARCHY
        
        # Add fibers as children
        for fiber in self.fibers:
            self.add_child(fiber)
        
        # Build connectivity if not provided
        if not self.connectivity:
            self._build_connectivity()
    
    def _build_connectivity(self) -> None:
        """Build connectivity graph based on spatial proximity"""
        self.connectivity = {}
        
        # Simple proximity-based connectivity
        for i, fiber1 in enumerate(self.fibers):
            connected = []
            for j, fiber2 in enumerate(self.fibers):
                if i != j:
                    # Check distance between fiber endpoints
                    distance = np.min([
                        np.linalg.norm(fiber1.points[0] - fiber2.points[0]),
                        np.linalg.norm(fiber1.points[0] - fiber2.points[-1]),
                        np.linalg.norm(fiber1.points[-1] - fiber2.points[0]),
                        np.linalg.norm(fiber1.points[-1] - fiber2.points[-1])
                    ])
                    
                    if distance < 5.0:  # 5 micron threshold
                        connected.append(fiber2.id)
            
            self.connectivity[fiber1.id] = connected
        
        # Calculate network density
        total_possible = len(self.fibers) * (len(self.fibers) - 1) / 2
        total_connections = sum(len(conn) for conn in self.connectivity.values()) / 2
        if total_possible > 0:
            self.network_density = total_connections / total_possible
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive network metrics"""
        metrics = {
            'num_fibers': len(self.fibers),
            'total_length': sum(fiber.length for fiber in self.fibers),
            'average_length': np.mean([fiber.length for fiber in self.fibers]),
            'network_density': self.network_density,
            'average_degree': np.mean([len(conn) for conn in self.connectivity.values()]),
            'degree_distribution': self._calculate_degree_distribution(),
            'clustering_coefficient': self._calculate_clustering_coefficient()
        }
        return metrics
    
    def _calculate_degree_distribution(self) -> Dict[int, int]:
        """Calculate degree distribution"""
        degrees = [len(self.connectivity.get(fiber.id, [])) for fiber in self.fibers]
        distribution = {}
        for degree in degrees:
            distribution[degree] = distribution.get(degree, 0) + 1
        return distribution
    
    def _calculate_clustering_coefficient(self) -> float:
        """Calculate average clustering coefficient"""
        coefficients = []
        for fiber_id, neighbors in self.connectivity.items():
            k = len(neighbors)
            if k < 2:
                coefficients.append(0.0)
                continue
            
            # Count edges between neighbors
            edges_between = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in self.connectivity.get(neighbors[i], []):
                        edges_between += 1
            
            # Clustering coefficient for this node
            coefficient = (2 * edges_between) / (k * (k - 1))
            coefficients.append(coefficient)
        
        return np.mean(coefficients) if coefficients else 0.0