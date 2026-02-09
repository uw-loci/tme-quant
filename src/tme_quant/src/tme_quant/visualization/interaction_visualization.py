# Visualization tools:
# - Network graph plotting
# - Spatial interaction maps
# - Heatmaps of interaction density
# - Interactive Plotly visualizations
# - 3D interaction visualization

"""
Visualization tools for cell-fiber interactions.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core.tme_models.interaction_models import (
    CellFiberInteraction, InteractionNetwork, InteractionCategory
)


class InteractionVisualizer:
    """Visualizer for cell-fiber interactions."""
    
    def __init__(self, colormap: str = 'viridis'):
        self.colormap = colormap
        self.color_dict = {
            InteractionCategory.PHYSICAL_CONTACT: '#ff0000',  # Red
            InteractionCategory.SPATIAL_PROXIMITY: '#ff9900',  # Orange
            InteractionCategory.PARALLEL_ALIGNMENT: '#00ff00',  # Green
            InteractionCategory.PERPENDICULAR_CROSSING: '#0000ff',  # Blue
            InteractionCategory.TUMOR_ASSOCIATED: '#9900ff',  # Purple
            InteractionCategory.UNKNOWN: '#666666',  # Gray
        }
    
    def plot_interaction_network(
        self,
        network: InteractionNetwork,
        figsize: Tuple[int, int] = (12, 10),
        show_labels: bool = False
    ) -> plt.Figure:
        """
        Plot interaction network as a graph.
        
        Args:
            network: InteractionNetwork to visualize
            figsize: Figure size
            show_labels: Whether to show node labels
            
        Returns:
            Matplotlib figure
        """
        if network.graph is None:
            network.build_network_graph()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Network layout
        self._plot_network_layout(network, ax1, show_labels)
        
        # Plot 2: Interaction types distribution
        self._plot_interaction_types(network, ax2)
        
        plt.tight_layout()
        return fig
    
    def _plot_network_layout(
        self,
        network: InteractionNetwork,
        ax: plt.Axes,
        show_labels: bool
    ):
        """Plot network graph layout."""
        import networkx as nx
        
        # Create layout
        pos = nx.spring_layout(network.graph, seed=42)
        
        # Draw nodes by type
        cell_nodes = [n for n, d in network.graph.nodes(data=True) 
                     if d['type'] == 'cell']
        fiber_nodes = [n for n, d in network.graph.nodes(data=True) 
                      if d['type'] == 'fiber']
        
        # Draw cell nodes
        nx.draw_networkx_nodes(
            network.graph, pos,
            nodelist=cell_nodes,
            node_color='red',
            node_size=300,
            alpha=0.8,
            ax=ax,
            label='Cells'
        )
        
        # Draw fiber nodes
        nx.draw_networkx_nodes(
            network.graph, pos,
            nodelist=fiber_nodes,
            node_color='blue',
            node_size=200,
            alpha=0.8,
            ax=ax,
            label='Fibers'
        )
        
        # Draw edges colored by interaction type
        for interaction in network.interactions:
            edge = (interaction.cell.id, interaction.fiber.id)
            color = self.color_dict[interaction.interaction_type]
            nx.draw_networkx_edges(
                network.graph, pos,
                edgelist=[edge],
                edge_color=color,
                width=2,
                alpha=0.6,
                ax=ax
            )
        
        if show_labels:
            nx.draw_networkx_labels(
                network.graph, pos,
                font_size=8,
                ax=ax
            )
        
        ax.set_title(f"Interaction Network: {len(network.interactions)} interactions")
        ax.legend()
        ax.axis('off')
    
    def _plot_interaction_types(
        self,
        network: InteractionNetwork,
        ax: plt.Axes
    ):
        """Plot distribution of interaction types."""
        # Count interaction types
        type_counts = {}
        for interaction in network.interactions:
            int_type = interaction.interaction_type.value
            type_counts[int_type] = type_counts.get(int_type, 0) + 1
        
        # Prepare data
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        
        # Get colors
        colors = [self.color_dict[InteractionCategory(t)] for t in types]
        
        # Create bar plot
        bars = ax.bar(types, counts, color=colors, alpha=0.8)
        ax.set_title("Interaction Type Distribution")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}',
                ha='center', va='bottom'
            )
    
    def plot_interaction_spatial_map(
        self,
        interactions: List[CellFiberInteraction],
        background_image: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 10)
    ) -> plt.Figure:
        """
        Plot interactions on spatial map.
        
        Args:
            interactions: List of interactions to plot
            background_image: Optional background image
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot background image if provided
        if background_image is not None:
            ax.imshow(background_image, cmap='gray', alpha=0.3)
        
        # Plot each interaction
        for interaction in interactions:
            self._plot_single_interaction(interaction, ax)
        
        ax.set_title(f"Spatial Map of {len(interactions)} Interactions")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.axis('equal')
        
        # Create legend
        self._create_interaction_legend(ax)
        
        plt.tight_layout()
        return fig
    
    def _plot_single_interaction(
        self,
        interaction: CellFiberInteraction,
        ax: plt.Axes
    ):
        """Plot a single interaction."""
        if not interaction.cell.rois or not interaction.fiber.rois:
            return
        
        # Get centroids
        cell_center = interaction.cell.rois[0].centroid()
        fiber_center = interaction.fiber.rois[0].centroid()
        
        # Get color based on interaction type
        color = self.color_dict[interaction.interaction_type]
        
        # Plot cell as circle
        cell_radius = np.sqrt(interaction.cell.get_measurement('area', default=100) / np.pi)
        circle = plt.Circle(
            cell_center, 
            cell_radius,
            color='red',
            alpha=0.5,
            fill=True
        )
        ax.add_patch(circle)
        
        # Plot fiber as line (simplified)
        if interaction.fiber.rois[0].type == 'line':
            fiber_line = interaction.fiber.rois[0].coordinates
            ax.plot(
                [p[0] for p in fiber_line],
                [p[1] for p in fiber_line],
                color='blue',
                linewidth=2,
                alpha=0.7
            )
        
        # Plot interaction line
        ax.plot(
            [cell_center[0], fiber_center[0]],
            [cell_center[1], fiber_center[1]],
            color=color,
            linewidth=2,
            alpha=0.6,
            linestyle='--'
        )
        
        # Add strength indicator
        strength_radius = {
            'weak': 2,
            'moderate': 4,
            'strong': 6,
            'very_strong': 8
        }.get(interaction.strength.value, 3)
        
        midpoint = (
            (cell_center[0] + fiber_center[0]) / 2,
            (cell_center[1] + fiber_center[1]) / 2
        )
        
        strength_circle = plt.Circle(
            midpoint,
            strength_radius,
            color=color,
            fill=True,
            alpha=0.8
        )
        ax.add_patch(strength_circle)
    
    def _create_interaction_legend(self, ax: plt.Axes):
        """Create custom legend for interaction types."""
        from matplotlib.patches import Patch
        
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Cell'),
            Patch(facecolor='blue', alpha=0.7, label='Fiber'),
        ]
        
        for int_type, color in self.color_dict.items():
            legend_elements.append(
                Patch(facecolor=color, alpha=0.6, label=int_type.value.replace('_', ' ').title())
            )
        
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize='small'
        )
    
    def create_interactive_plot(
        self,
        network: InteractionNetwork
    ) -> go.Figure:
        """
        Create interactive Plotly visualization of interactions.
        
        Args:
            network: InteractionNetwork to visualize
            
        Returns:
            Plotly figure
        """
        if network.graph is None:
            network.build_network_graph()
        
        import plotly.graph_objects as go
        import networkx as nx
        
        # Create layout
        pos = nx.spring_layout(network.graph, seed=42)
        
        # Prepare node traces
        cell_nodes = []
        fiber_nodes = []
        cell_x, cell_y = [], []
        fiber_x, fiber_y = [], []
        cell_labels, fiber_labels = [], []
        
        for node, data in network.graph.nodes(data=True):
            x, y = pos[node]
            if data['type'] == 'cell':
                cell_x.append(x)
                cell_y.append(y)
                cell_labels.append(
                    f"Cell: {data['object'].name}<br>"
                    f"Type: {data['classification']}"
                )
                cell_nodes.append(node)
            else:
                fiber_x.append(x)
                fiber_y.append(y)
                fiber_labels.append(
                    f"Fiber: {data['object'].name}<br>"
                    f"Type: {data['fiber_type']}"
                )
                fiber_nodes.append(node)
        
        # Create edge traces
        edge_x, edge_y = [], []
        edge_colors = []
        edge_widths = []
        edge_labels = []
        
        for interaction in network.interactions:
            cell_pos = pos[interaction.cell.id]
            fiber_pos = pos[interaction.fiber.id]
            
            edge_x.extend([cell_pos[0], fiber_pos[0], None])
            edge_y.extend([cell_pos[1], fiber_pos[1], None])
            
            color = self.color_dict[interaction.interaction_type]
            edge_colors.append(color)
            
            # Width based on strength
            width = {
                'weak': 1,
                'moderate': 2,
                'strong': 3,
                'very_strong': 4
            }.get(interaction.strength.value, 2)
            edge_widths.append(width)
            
            edge_labels.append(
                f"Interaction: {interaction.interaction_type.value}<br>"
                f"Distance: {interaction.distance:.1f} µm<br>"
                f"Strength: {interaction.strength.value}"
            )
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        for i in range(0, len(edge_x), 3):
            fig.add_trace(go.Scatter(
                x=edge_x[i:i+3],
                y=edge_y[i:i+3],
                mode='lines',
                line=dict(
                    color=edge_colors[i//3],
                    width=edge_widths[i//3]
                ),
                hoverinfo='text',
                text=edge_labels[i//3],
                showlegend=False
            ))
        
        # Add cell nodes
        fig.add_trace(go.Scatter(
            x=cell_x,
            y=cell_y,
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='circle'
            ),
            text=cell_labels,
            hoverinfo='text',
            name='Cells',
            showlegend=True
        ))
        
        # Add fiber nodes
        fig.add_trace(go.Scatter(
            x=fiber_x,
            y=fiber_y,
            mode='markers',
            marker=dict(
                size=10,
                color='blue',
                symbol='diamond'
            ),
            text=fiber_labels,
            hoverinfo='text',
            name='Fibers',
            showlegend=True
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Cell-Fiber Interaction Network ({len(network.interactions)} interactions)",
            showlegend=True,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600
        )
        
        return fig