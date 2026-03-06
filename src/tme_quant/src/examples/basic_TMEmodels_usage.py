# Example 1: get the fiber center coordinates from fiber objects
# Get center coordinates from FiberProperties
fiber_props = extraction_result.fibers[0]
center = fiber_props.center_coordinates  # np.array([x, y]) or [x, y, z]

# Get center from FiberObject
fiber_obj = FiberObject(...)
center = fiber_obj.get_center_coordinates()  # np.array([x, y]) or [x, y, z]

# Export to CSV will now include center_x, center_y, center_z columns
exporter = FiberAnalysisExporter()
exporter.export(result, output_dir, formats=["csv", "json", "geojson"])