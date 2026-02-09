"""
Facade patterns providing unified interfaces to complex subsystems.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

# Import bridges and adapters
from .bridges import BioImageBridge, VisualizationBridge
from .adapters import ProjectAdapter
from .events import EventPublisher, EventSubscriber

# Import hierarchy
from qupath_hierarchy import QuPathLikeProject


@dataclass
class UnifiedAnalysisFacade:
    """
    Facade providing unified interface for analysis workflows.
    """
    
    bioimage_bridge: BioImageBridge = field(default_factory=BioImageBridge)
    visualization_bridge: VisualizationBridge = field(default_factory=VisualizationBridge)
    project_adapter: ProjectAdapter = field(default_factory=ProjectAdapter)
    
    # Event system
    event_publisher: EventPublisher = field(default_factory=EventPublisher)
    
    def analyze_images(
        self,
        image_paths: List[Union[str, Path]],
        analysis_pipeline: str = "cell_analysis",
        output_dir: Optional[Union[str, Path]] = None
    ) -> QuPathLikeProject:
        """
        Unified image analysis workflow.
        
        Args:
            image_paths: List of image paths to analyze
            analysis_pipeline: Analysis pipeline to use
            output_dir: Optional output directory
            
        Returns:
            QuPathLikeProject with analysis results
        """
        from bioimage_core.utils.file_io import load_image
        from qupath_hierarchy import QuPathLikeProject
        
        # Create project
        project = QuPathLikeProject(
            name=f"Analysis_{analysis_pipeline}"
        )
        
        # Publish start event
        self.event_publisher.publish(
            "analysis_started",
            {
                "project_name": project.name,
                "num_images": len(image_paths),
                "pipeline": analysis_pipeline
            }
        )
        
        for img_idx, img_path in enumerate(image_paths):
            img_path = Path(img_path)
            
            # Publish image start event
            self.event_publisher.publish(
                "image_analysis_started",
                {
                    "image_path": str(img_path),
                    "image_index": img_idx,
                    "total_images": len(image_paths)
                }
            )
            
            # Load image using existing tools
            image_data, metadata = load_image(img_path)
            
            # Analyze with hierarchy integration
            img_project = self.bioimage_bridge.process_image_with_hierarchy(
                image_data,
                metadata,
                analysis_pipeline
            )
            
            # Merge into main project
            project.merge(img_project)
            
            # Publish completion event
            self.event_publisher.publish(
                "image_analysis_completed",
                {
                    "image_path": str(img_path),
                    "objects_created": len(img_project.objects),
                    "project_name": project.name
                }
            )
        
        # Save results if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            self._save_analysis_results(project, output_dir)
        
        # Publish completion event
        self.event_publisher.publish(
            "analysis_completed",
            {
                "project_name": project.name,
                "total_objects": len(project.objects),
                "total_images": len(project.images)
            }
        )
        
        return project
    
    def _save_analysis_results(
        self,
        project: QuPathLikeProject,
        output_dir: Path
    ) -> None:
        """Save analysis results to disk."""
        # Save hierarchy project
        project_path = output_dir / f"{project.name}.qproj"
        project.save(project_path, format='pickle')
        
        # Export measurements
        measurements_path = output_dir / "measurements.csv"
        df = project.export_measurements(format='dataframe')
        df.to_csv(measurements_path)
        
        # Export visualizations
        viz_path = output_dir / "visualizations"
        self.visualization_bridge.export_visualizations(
            project, viz_path, formats=['png', 'html']
        )
        
        # Export in existing format for backward compatibility
        existing_format = output_dir / "legacy_format.json"
        import json
        legacy_data = self.project_adapter.export_project_to_existing_format(
            project, format='legacy'
        )
        with open(existing_format, 'w') as f:
            json.dump(legacy_data, f, indent=2)
    
    def convert_existing_analysis(
        self,
        existing_results: Dict[str, Any],
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> QuPathLikeProject:
        """
        Convert existing analysis results to hierarchy format.
        
        Args:
            existing_results: Existing analysis results
            image_paths: Corresponding image paths
            output_dir: Optional output directory
            
        Returns:
            QuPathLikeProject
        """
        # Convert using adapter
        project = self.project_adapter.create_project_from_existing_analysis(
            existing_results, image_paths
        )
        
        # Save if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            project.save(output_dir / "converted_project.qproj")
            
            # Also export measurements
            df = project.export_measurements(format='dataframe')
            df.to_csv(output_dir / "converted_measurements.csv")
        
        return project
    
    def visualize_project(
        self,
        project: QuPathLikeProject,
        viewer_type: str = "napari",
        **kwargs
    ) -> Any:
        """
        Visualize hierarchy project.
        
        Args:
            project: Hierarchy project
            viewer_type: Type of viewer ('napari', 'matplotlib', 'plotly')
            **kwargs: Viewer-specific options
            
        Returns:
            Viewer instance
        """
        if viewer_type == "napari":
            return self.visualization_bridge.display_project_in_napari(
                project, **kwargs
            )
        elif viewer_type == "matplotlib":
            from qupath_hierarchy import MatplotlibVisualizer
            viz = MatplotlibVisualizer()
            return viz.display_project(project, **kwargs)
        else:
            raise ValueError(f"Unsupported viewer type: {viewer_type}")


@dataclass
class ProjectManagementFacade:
    """
    Facade for project management operations.
    """
    
    def create_project(
        self,
        name: str,
        description: str = "",
        tags: List[str] = None
    ) -> QuPathLikeProject:
        """
        Create a new hierarchy project.
        
        Args:
            name: Project name
            description: Project description
            tags: List of tags
            
        Returns:
            New QuPathLikeProject
        """
        from qupath_hierarchy import QuPathLikeProject
        
        project = QuPathLikeProject(name=name)
        project.metadata['description'] = description
        project.metadata['tags'] = tags or []
        
        return project
    
    def open_project(
        self,
        filepath: Union[str, Path],
        format: str = 'auto'
    ) -> QuPathLikeProject:
        """
        Open existing project file.
        
        Args:
            filepath: Path to project file
            format: File format ('auto', 'pickle', 'json', 'zarr')
            
        Returns:
            Loaded QuPathLikeProject
        """
        from qupath_hierarchy import QuPathLikeProject
        
        project = QuPathLikeProject(name="Temp")
        
        if format == 'auto':
            # Auto-detect format
            filepath = Path(filepath)
            if filepath.suffix == '.qproj':
                format = 'pickle'
            elif filepath.suffix == '.json':
                format = 'json'
            elif filepath.suffix == '.zarr':
                format = 'zarr'
            else:
                format = 'pickle'  # Default
        
        project.load(filepath, format=format)
        return project
    
    def merge_projects(
        self,
        projects: List[QuPathLikeProject],
        merged_name: Optional[str] = None
    ) -> QuPathLikeProject:
        """
        Merge multiple projects into one.
        
        Args:
            projects: List of projects to merge
            merged_name: Name for merged project
            
        Returns:
            Merged QuPathLikeProject
        """
        if not projects:
            raise ValueError("No projects to merge")
        
        merged_name = merged_name or f"Merged_{len(projects)}_projects"
        merged = QuPathLikeProject(name=merged_name)
        
        for project in projects:
            merged.merge(project)
        
        # Update metadata
        merged.metadata['merged_from'] = [
            project.name for project in projects
        ]
        merged.metadata['merge_date'] = datetime.now().isoformat()
        
        return merged
    
    def validate_project(
        self,
        project: QuPathLikeProject
    ) -> Dict[str, Any]:
        """
        Validate project integrity.
        
        Args:
            project: Project to validate
            
        Returns:
            Validation results
        """
        errors = project.validate()
        
        # Additional validations
        warnings = []
        
        # Check for duplicate names
        names = {}
        for obj in project.objects.values():
            if obj.name in names:
                warnings.append(f"Duplicate name: {obj.name}")
            names[obj.name] = names.get(obj.name, 0) + 1
        
        # Check measurement consistency
        for obj in project.objects.values():
            if not obj.measurements:
                warnings.append(f"Object {obj.name} has no measurements")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'statistics': project.get_statistics()
        }
    
    def export_project(
        self,
        project: QuPathLikeProject,
        output_path: Union[str, Path],
        formats: List[str] = None
    ) -> Dict[str, Path]:
        """
        Export project in multiple formats.
        
        Args:
            project: Project to export
            output_path: Base output path
            formats: List of formats to export
            
        Returns:
            Dictionary of exported file paths by format
        """
        if formats is None:
            formats = ['pickle', 'json', 'csv', 'html']
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        exported = {}
        
        # Export in each format
        for fmt in formats:
            if fmt == 'pickle':
                path = output_path.with_suffix('.qproj')
                project.save(path, format='pickle')
                exported['pickle'] = path
            
            elif fmt == 'json':
                path = output_path.with_suffix('.json')
                project.save(path, format='json')
                exported['json'] = path
            
            elif fmt == 'csv':
                path = output_path.with_suffix('.csv')
                df = project.export_measurements(format='dataframe')
                df.to_csv(path)
                exported['csv'] = path
            
            elif fmt == 'html':
                path = output_path.with_suffix('.html')
                # Create HTML report
                self._create_html_report(project, path)
                exported['html'] = path
        
        return exported
    
    def _create_html_report(
        self,
        project: QuPathLikeProject,
        output_path: Path
    ) -> None:
        """Create HTML report for project."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Project Report: {project_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Project Report: {project_name}</h1>
            
            <div class="section">
                <h2>Project Information</h2>
                <p><strong>Created:</strong> {created}</p>
                <p><strong>Modified:</strong> {modified}</p>
                <p><strong>Description:</strong> {description}</p>
            </div>
            
            <div class="section">
                <h2>Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {statistics_rows}
                </table>
            </div>
            
            <div class="section">
                <h2>Object Types</h2>
                <table>
                    <tr><th>Type</th><th>Count</th></tr>
                    {object_type_rows}
                </table>
            </div>
        </body>
        </html>
        """
        
        stats = project.get_statistics()
        
        # Format statistics rows
        stats_rows = ""
        for key, value in stats.items():
            if not isinstance(value, dict):
                stats_rows += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        # Format object type rows
        obj_type_rows = ""
        for obj_type, count in stats.get('object_types', {}).items():
            obj_type_rows += f"<tr><td>{obj_type}</td><td>{count}</td></tr>"
        
        # Fill template
        html_content = html_template.format(
            project_name=project.name,
            created=project.metadata.get('created', 'Unknown'),
            modified=project.metadata.get('modified', 'Unknown'),
            description=project.metadata.get('description', ''),
            statistics_rows=stats_rows,
            object_type_rows=obj_type_rows
        )
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)