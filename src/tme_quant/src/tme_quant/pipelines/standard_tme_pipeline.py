class StandardTMEPipeline:
    """UPDATED WITH INTERACTION ANALYSIS"""
    
    def run_with_interactions(
        self,
        image_data: np.ndarray,
        channels: Dict[str, int],
        analyze_interactions: bool = True,
        interaction_distance: float = 50.0
    ) -> TMEProject:
        """
        Run complete TME analysis including interactions.
        
        Args:
            image_data: Multi-channel image
            channels: Channel mapping
            analyze_interactions: Whether to analyze interactions
            interaction_distance: Maximum interaction distance (microns)
            
        Returns:
            TMEProject with complete analysis including interactions
        """
        # Run standard analysis (cells, fibers, tumor regions)
        project = self.run_standard_analysis(image_data, channels)
        
        if analyze_interactions:
            # Analyze interactions in each tumor region
            tumor_regions = project.find_objects(ObjectType.TUMOR_REGION)
            
            for tumor in tumor_regions:
                print(f"Analyzing interactions in {tumor.name}")
                
                # Get cells and fibers in this tumor region
                cells = tumor.get_descendants(ObjectType.CELL)
                fibers = tumor.get_descendants(ObjectType.COLLAGEN_FIBER)
                
                if cells and fibers:
                    # Run interaction analysis
                    interaction_results = project.analyze_cell_fiber_interactions(
                        region_id=tumor.id,
                        pixel_size=self._get_pixel_size(image_data),
                        max_distance=interaction_distance
                    )
                    
                    # Get tumor-associated collagen features
                    tumor_features = tumor.get_measurement('tumor_collagen_features', {})
                    
                    print(f"  Found {len(interaction_results.interactions)} interactions")
                    print(f"  TACS scores: {tumor_features.get('tacs1_score', 0):.2f}, "
                          f"{tumor_features.get('tacs2_score', 0):.2f}, "
                          f"{tumor_features.get('tacs3_score', 0):.2f}")
            
            # Also analyze global interactions (across entire image)
            print("Analyzing global interactions...")
            global_network = project.analyze_cell_fiber_interactions(
                region_id=None,
                pixel_size=self._get_pixel_size(image_data),
                max_distance=interaction_distance
            )
            
            # Store global statistics
            global_stats = project.get_interaction_statistics()
            project.metadata['interaction_statistics'] = global_stats
        
        return project
    
    def generate_interaction_report(
        self,
        project: TMEProject,
        output_path: str
    ) -> None:
        """
        Generate comprehensive interaction analysis report.
        
        Args:
            project: TMEProject with interaction analysis
            output_path: Path to save report
        """
        from tme_quant.visualization.interaction_visualization import InteractionVisualizer
        import matplotlib.pyplot as plt
        
        visualizer = InteractionVisualizer()
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Get interaction networks
        networks = list(project.interaction_networks.values())
        
        if networks:
            # Plot 1: Network visualization
            ax1 = plt.subplot(2, 3, 1)
            if networks[0].graph is not None:
                visualizer._plot_network_layout(networks[0], ax1, show_labels=False)
            
            # Plot 2: Interaction type distribution
            ax2 = plt.subplot(2, 3, 2)
            visualizer._plot_interaction_types(networks[0], ax2)
            
            # Plot 3: Distance histogram
            ax3 = plt.subplot(2, 3, 3)
            distances = [i.distance for i in networks[0].interactions if i.distance]
            ax3.hist(distances, bins=20, alpha=0.7, color='skyblue')
            ax3.set_title("Interaction Distance Distribution")
            ax3.set_xlabel("Distance (microns)")
            ax3.set_ylabel("Count")
            
            # Plot 4: Angle distribution
            ax4 = plt.subplot(2, 3, 4)
            angles = [i.angle for i in networks[0].interactions if i.angle]
            ax4.hist(angles, bins=18, range=(0, 90), alpha=0.7, color='lightgreen')
            ax4.set_title("Cell-Fiber Angle Distribution")
            ax4.set_xlabel("Angle (degrees)")
            ax4.set_ylabel("Count")
            
            # Plot 5: Network metrics
            ax5 = plt.subplot(2, 3, 5)
            if networks[0].network_metrics:
                metrics = networks[0].network_metrics
                metric_names = ['num_nodes', 'num_edges', 'network_density', 'avg_degree']
                metric_values = [metrics.get(m, 0) for m in metric_names]
                bars = ax5.bar(range(len(metric_names)), metric_values, alpha=0.7)
                ax5.set_title("Network Metrics")
                ax5.set_xticks(range(len(metric_names)))
                ax5.set_xticklabels(metric_names, rotation=45)
            
            # Plot 6: Tumor features (if available)
            ax6 = plt.subplot(2, 3, 6)
            tumor_regions = project.find_objects(ObjectType.TUMOR_REGION)
            for tumor in tumor_regions:
                features = tumor.get_measurement('tumor_collagen_features', {})
                if features:
                    tacs_scores = [
                        features.get('tacs1_score', 0),
                        features.get('tacs2_score', 0),
                        features.get('tacs3_score', 0)
                    ]
                    ax6.bar(['TACS-1', 'TACS-2', 'TACS-3'], tacs_scores, alpha=0.7)
                    ax6.set_title("Tumor Associated Collagen Signatures")
                    ax6.set_ylabel("Score")
                    break
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()