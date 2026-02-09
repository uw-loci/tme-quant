class CombinedCellFiberAnalysis:
    """UPDATED WITH INTERACTION ANALYSIS"""
    
    def _analyze_cell_fiber_interactions(
        self,
        cells: List[Cell],
        fibers: List[CollagenFiber],
        tumor_region: Optional[TumorRegion] = None
    ) -> Dict[str, Any]:
        """
        Analyze interactions between cells and fibers.
        
        Args:
            cells: List of cells
            fibers: List of fibers
            tumor_region: Optional tumor region context
            
        Returns:
            Interaction analysis results
        """
        from tme_quant.tme_analysis.cell_fiber_interaction import InteractionAnalyzer
        
        # Initialize analyzer
        analyzer = InteractionAnalyzer(
            max_interaction_distance=50.0,  # microns
            parallel_processing=True,
            verbose=True
        )
        
        # Analyze interactions
        interaction_network = analyzer.analyze_region(
            cells=cells,
            fibers=fibers,
            region_context=tumor_region,
            pixel_size=self._get_pixel_size()  # Get from image metadata
        )
        
        # Calculate tumor-associated features if in tumor region
        tumor_features = {}
        if tumor_region:
            tumor_features = analyzer.calculate_tumor_associated_collagen_features(
                interaction_network, tumor_region
            )
            
            # Store in tumor region
            tumor_region.add_measurement('tumor_collagen_features', tumor_features)
        
        # Get network statistics
        network_stats = interaction_network.network_metrics
        region_stats = interaction_network.get_measurement('region_statistics', {})
        
        # Find critical interactions
        critical_interactions = interaction_network.find_critical_interactions(
            threshold=0.8
        )
        
        # Prepare results
        results = {
            'interaction_network': interaction_network,
            'num_interactions': len(interaction_network.interactions),
            'network_statistics': network_stats,
            'region_statistics': region_stats,
            'tumor_collagen_features': tumor_features,
            'critical_interactions': [
                {
                    'cell': i.cell.name,
                    'fiber': i.fiber.name,
                    'type': i.interaction_type.value,
                    'strength': i.strength.value,
                    'distance': i.distance,
                    'invasive_potential': i.invasive_potential_score
                }
                for i in critical_interactions
            ]
        }
        
        # Visualize in Napari
        self._visualize_interactions(interaction_network)
        
        return results
    
    def _visualize_interactions(self, network: InteractionNetwork):
        """Visualize interactions in Napari."""
        from tme_quant.visualization.interaction_visualization import InteractionVisualizer
        
        # Create visualizer
        visualizer = InteractionVisualizer()
        
        # Add interaction lines to Napari
        for interaction in network.interactions:
            if interaction.cell.rois and interaction.fiber.rois:
                # Get centroids
                cell_center = interaction.cell.rois[0].centroid()
                fiber_center = interaction.fiber.rois[0].centroid()
                
                # Create line segment
                line_segment = np.array([
                    [cell_center[0], cell_center[1]],
                    [fiber_center[0], fiber_center[1]]
                ])
                
                # Add to Napari as shapes layer
                self.viewer.add_shapes(
                    line_segment.reshape(1, 2, 2),
                    shape_type='line',
                    edge_color=self._get_interaction_color(interaction),
                    edge_width=2,
                    opacity=0.6,
                    name=f"Interaction_{interaction.cell.name}_{interaction.fiber.name}"
                )
        
        # Add interaction strength indicators
        for interaction in network.interactions:
            if interaction.distance and interaction.distance < 20:
                # Add strength indicator at midpoint
                cell_center = interaction.cell.rois[0].centroid()
                fiber_center = interaction.fiber.rois[0].centroid()
                midpoint = (
                    (cell_center[0] + fiber_center[0]) / 2,
                    (cell_center[1] + fiber_center[1]) / 2
                )
                
                # Size based on strength
                size = {
                    'weak': 3,
                    'moderate': 5,
                    'strong': 7,
                    'very_strong': 9
                }.get(interaction.strength.value, 5)
                
                self.viewer.add_points(
                    [midpoint],
                    size=size,
                    face_color=self._get_interaction_color(interaction),
                    name=f"Strength_{interaction.cell.name}"
                )
    
    def _get_interaction_color(self, interaction: CellFiberInteraction) -> str:
        """Get color for interaction based on type."""
        color_map = {
            'physical_contact': '#ff0000',
            'spatial_proximity': '#ff9900',
            'parallel_alignment': '#00ff00',
            'perpendicular_crossing': '#0000ff',
            'tumor_associated': '#9900ff',
        }
        return color_map.get(interaction.interaction_type.value, '#666666')
    
    """Updated with full interaction analysis."""
    
    def run_interaction_analysis(self, roi_index: Optional[int] = None):
        """Run comprehensive interaction analysis."""
        # 1. Get cells and fibers
        # 2. Run interaction analysis
        # 3. Calculate tumor-associated features
        # 4. Visualize in Napari
        # 5. Export results
        pass
    
    def visualize_interactions(self, interaction_network: InteractionNetwork):
        """Visualize interactions in Napari viewer."""
        # Add interaction lines
        # Add strength indicators
        # Color by interaction type
        # Add network overlay
        pass