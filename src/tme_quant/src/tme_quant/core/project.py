"""
TME Project management module for hierarchical analysis.
"""
from typing import Dict, List, Optional, Tuple, Any
from .base_models import TMEObject, TMEType
from .tme_models import (
    Interaction,
    InteractionNetwork,
    InteractionType,
    Cell,
    Fiber
)
from ..tme_analysis.cell_fiber_interaction import InteractionAnalyzer


class ImageObject(TMEObject):
    """Image object in the TME hierarchy."""
    def __init__(self, name: str = "", **kwargs):
        super().__init__()
        self.name = name
        self.type = TMEType.IMAGE


class InteractionNetworkWrapper(TMEObject):
    """
    Wrapper to integrate dataclass-based InteractionNetwork with TMEObject hierarchy.
    This allows interaction networks to be part of the TME hierarchy tree.
    """
    def __init__(self, network: InteractionNetwork, name: str = "", **kwargs):
        super().__init__()
        self.name = name or "interaction_network"
        self.type = TMEType.HIERARCHY
        self.network = network  # The actual InteractionNetwork dataclass
    
    @property
    def interactions(self) -> List[Interaction]:
        """Access interactions from wrapped network."""
        return self.network.interactions
    
    @property
    def network_metrics(self) -> Dict[str, Any]:
        """Calculate and return network metrics."""
        return self.network.calculate_network_metrics()
    
    def find_critical_interactions(self, centrality_threshold: float = 0.8) -> List[Interaction]:
        """Find critical interactions based on centrality threshold."""
        # Filter interactions by interaction_strength as proxy for centrality
        return [i for i in self.network.interactions if i.interaction_strength >= centrality_threshold]
    
    def get_cell_interaction_profile(self, cell_id: str) -> Optional[Dict[str, Any]]:
        """Get interaction profile for a specific cell."""
        cell_interactions = self.network.get_interactions_by_cell(cell_id)
        if not cell_interactions:
            return None
        
        # Count interaction types
        interaction_types = {}
        for interaction in cell_interactions:
            int_type = interaction.interaction_type.value
            interaction_types[int_type] = interaction_types.get(int_type, 0) + 1
        
        return {
            'degree': len(cell_interactions),
            'interaction_types': interaction_types,
            'interactions': cell_interactions
        }


class TMEProject:
    """Main container managing the entire hierarchy - UPDATED WITH INTERACTIONS."""
    
    def __init__(self, name: str):
        self.name = name
        self.images: Dict[str, ImageObject] = {}
        self.objects: Dict[str, TMEObject] = {}
        self.hierarchy_roots: List[TMEObject] = []
        
        # Interaction management
        self.interaction_networks: Dict[str, InteractionNetworkWrapper] = {}
        self.interaction_analyzer: Optional[InteractionAnalyzer] = None
    
    def find_objects(self, object_type: TMEType) -> List[TMEObject]:
        """Find all objects of a specific type in the project."""
        found_objects = []
        for obj in self.objects.values():
            if obj.type == object_type:
                found_objects.append(obj)
            # Also search descendants
            found_objects.extend([d for d in obj.get_descendants() if d.type == object_type])
        return found_objects
    
    def analyze_cell_fiber_interactions(
        self,
        region_id: Optional[str] = None,
        pixel_size: Tuple[float, float] = (1.0, 1.0),
        max_distance: float = 50.0
    ) -> InteractionNetworkWrapper:
        """
        Analyze cell-fiber interactions in a region.
        
        Args:
            region_id: ID of region to analyze (None for entire project)
            pixel_size: Pixel size in microns
            max_distance: Maximum interaction distance in microns
            
        Returns:
            InteractionNetworkWrapper containing all interactions
        """
        # Initialize analyzer if needed
        if self.interaction_analyzer is None:
            self.interaction_analyzer = InteractionAnalyzer(
                max_interaction_distance=max_distance
            )
        
        # Get cells and fibers based on region
        if region_id:
            # Analyze specific region
            region = self.objects.get(region_id)
            if not region:
                raise ValueError(f"Region {region_id} not found")
            
            cells = [obj for obj in region.get_descendants() if obj.type == TMEType.CELL]
            fibers = [obj for obj in region.get_descendants() if obj.type == TMEType.FIBER]
            region_context = region if region.type == TMEType.REGION else None
        else:
            # Analyze entire project
            cells = self.find_objects(TMEType.CELL)
            fibers = self.find_objects(TMEType.FIBER)
            region_context = None
        
        # Run analysis
        network_data = self.interaction_analyzer.analyze_region(
            cells=cells,
            fibers=fibers,
            region_context=region_context,
            pixel_size=pixel_size
        )
        
        # Wrap the network for hierarchy integration
        network = InteractionNetworkWrapper(
            network=network_data,
            name=f"network_{region_context.name if region_context else 'global'}"
        )
        
        # Store network
        network_id = f"network_{len(self.interaction_networks)}"
        self.interaction_networks[network_id] = network
        
        # Add to hierarchy
        if region_id:
            region.add_child(network)
        else:
            self.hierarchy_roots.append(network)
        
        # Calculate tumor-associated features if applicable
        if region_context and region_context.type == TMEType.REGION:
            tumor_features = self.interaction_analyzer.calculate_tumor_associated_collagen_features(
                network.network,  # Pass the unwrapped network dataclass
                region_context
            )
            region_context.update_properties(tumor_collagen_features=tumor_features)
        
        return network
    
    def get_interaction_statistics(self) -> Dict[str, Any]:
        """Get statistics for all interaction networks."""
        stats = {
            'total_networks': len(self.interaction_networks),
            'total_interactions': 0,
            'networks': {}
        }
        
        for net_id, network in self.interaction_networks.items():
            net_stats = {
                'num_interactions': len(network.interactions),
                'network_metrics': network.network_metrics,
                'region_statistics': network.properties.get('region_statistics', {})
            }
            stats['networks'][net_id] = net_stats
            stats['total_interactions'] += len(network.interactions)
        
        return stats
    
    def find_critical_interactions(
        self,
        network_id: str,
        centrality_threshold: float = 0.8
    ) -> List[Interaction]:
        """Find critical interactions in a network."""
        network = self.interaction_networks.get(network_id)
        if not network:
            return []
        
        return network.find_critical_interactions(centrality_threshold)
    
    def get_cell_interaction_profile(self, cell_id: str) -> Dict[str, Any]:
        """Get interaction profile for a specific cell across all networks."""
        profiles = {}
        
        for net_id, network in self.interaction_networks.items():
            profile = network.get_cell_interaction_profile(cell_id)
            if profile:
                profiles[net_id] = profile
        
        # Combine profiles if cell appears in multiple networks
        combined = {
            'cell_id': cell_id,
            'total_interactions': 0,
            'networks': list(profiles.keys()),
            'interaction_summary': {}
        }
        
        for net_profile in profiles.values():
            combined['total_interactions'] += net_profile.get('degree', 0)
            
            # Aggregate interaction types
            for int_type, count in net_profile.get('interaction_types', {}).items():
                combined['interaction_summary'][int_type] = (
                    combined['interaction_summary'].get(int_type, 0) + count
                )
        
        return combined