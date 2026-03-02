# TACS Analysis Workflow Guide

This guide explains how to perform Tumor-Associated Collagen Signatures (TACS) analysis using the `napari-curvealign` plugin. The analysis calculates the relative angle of collagen fibers with respect to a tumor boundary (0° = tangential, 90° = perpendicular).

## Prerequisites

1.  **Image**: A microscopy image containing collagen fibers and tumor regions.
2.  **Fiber Data**: Detected fiber orientations (via Curvelets or CT-FIRE).
3.  **Tumor Boundary**: A defined Region of Interest (ROI) representing the tumor.

## Step-by-Step Workflow

### 1. Load Image and Detect Fibers
1.  Open the **CurveAlign** widget in napari.
2.  **Load Image**: Click "Open" in the Main tab and select your image.
3.  **Run Analysis**: Click "Run" (or "Analyze All Images") to detect fibers.
    *   This populates the internal database with "fiber" objects containing orientation data.
    *   *Note: Ensure "Curvelets" or "CT-FIRE" mode is selected.*

### 2. Define Tumor Boundary
You need to tell the plugin which region is the tumor.

1.  Switch to the **ROI Manager** tab.
2.  **Draw ROI**:
    *   Use the "Polygon" or "Freehand" tools to outline the tumor boundary.
    *   The ROI will appear in the "ROI List" at the bottom left.
3.  **Convert to Annotation**:
    *   Select the drawn ROI in the "ROI List".
    *   In the **Region Analysis (Advanced)** panel (right side), select **Type: Tumor**.
    *   Click **"Use Selected ROI"**.
    *   The ROI will now appear in the "Defined Regions" list as a "Tumor" region.

### 3. Run TACS Analysis
1.  In the **TACS Analysis** section (bottom right of ROI Manager tab):
2.  Select the **Tumor** region from the "Defined Regions" list.
3.  Click **"Run TACS"**.

### 4. View Results
The plugin will automatically switch to the **Post-Processing** tab.

*   **Histogram**: Displays the distribution of **Relative Angles** (0-90°).
    *   **Near 0°**: Fibers are parallel to the tumor boundary (TACS-1 / TACS-2).
    *   **Near 90°**: Fibers are perpendicular to the tumor boundary (TACS-3, indicative of invasion).
*   **Vertical Lines**: Markers at 45° help visualize the separation between tangential and perpendicular alignment.

## Troubleshooting

*   **"No fibers detected"**: Make sure you ran the main analysis (Step 1) before defining the tumor boundary.
*   **"No fibers found near the boundary"**: Check your scale or distance settings. The fibers might be too far from the drawn ROI.
*   **Fiji Integration**: You can also draw the tumor boundary in Fiji, push it to the ROI Manager, import it here using "Pull from Fiji", and then convert it to a Tumor region.
