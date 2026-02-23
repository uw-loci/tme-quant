"""
Environment Bridge for Cross-Python-Version Segmentation.

This module enables running segmentation models (like StarDist) in different
Python environments, similar to Appose framework approach.

Inspired by: https://github.com/apposed/appose-python
"""

import subprocess
import json
import tempfile
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class EnvironmentBridge:
    """
    Bridge to run code in a different Python environment.
    
    This allows StarDist (Python 3.9-3.12) to run alongside the main
    application (Python 3.13+) by using subprocess isolation.
    
    Similar to Appose framework but simplified for our use case.
    """
    
    def __init__(self, python_path: Optional[str] = None, env_name: Optional[str] = None):
        """
        Initialize environment bridge.
        
        Parameters
        ----------
        python_path : str, optional
            Full path to Python executable in target environment.
            E.g., "/path/to/conda/envs/stardist/bin/python"
        env_name : str, optional
            Conda/venv environment name. Will try to auto-detect path.
        """
        self.python_path = python_path
        self.env_name = env_name
        
        if python_path is None and env_name is not None:
            self.python_path = self._find_env_python(env_name)
    
    def _find_env_python(self, env_name: str) -> Optional[str]:
        """Try to find Python executable in conda/venv environment."""
        # Try conda first
        conda_base = os.environ.get('CONDA_PREFIX', os.path.expanduser('~/miniconda3'))
        conda_python = Path(conda_base) / 'envs' / env_name / 'bin' / 'python'
        if conda_python.exists():
            return str(conda_python)
        
        # Try venv in common locations
        for base in [Path.cwd(), Path.home()]:
            venv_python = base / env_name / 'bin' / 'python'
            if venv_python.exists():
                return str(venv_python)
        
        return None
    
    def run_script(self, script: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run Python script in target environment.
        
        Parameters
        ----------
        script : str
            Python code to execute
        inputs : dict, optional
            Input data to pass to script (serialized as JSON)
            
        Returns
        -------
        dict
            Output data from script
        """
        if self.python_path is None:
            raise ValueError(
                "No Python path configured. Provide python_path or env_name "
                "when creating EnvironmentBridge."
            )
        
        # Create temp files for communication
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_file = tmpdir / 'input.json'
            output_file = tmpdir / 'output.json'
            script_file = tmpdir / 'script.py'
            
            # Write inputs
            if inputs:
                with open(input_file, 'w') as f:
                    json.dump(inputs, f)
            
            # Wrap script to handle I/O
            wrapped_script = f"""
import json
import sys
from pathlib import Path

# Load inputs
input_file = Path('{input_file}')
if input_file.exists():
    with open(input_file) as f:
        inputs = json.load(f)
else:
    inputs = {{}}

# User script
{script}

# Save outputs (script should define 'outputs' dict)
with open('{output_file}', 'w') as f:
    json.dump(outputs, f)
"""
            
            # Write script
            with open(script_file, 'w') as f:
                f.write(wrapped_script)
            
            # Run in subprocess
            try:
                result = subprocess.run(
                    [self.python_path, str(script_file)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Script failed in environment:\n"
                        f"STDOUT: {result.stdout}\n"
                        f"STDERR: {result.stderr}"
                    )
                
                # Read outputs
                if output_file.exists():
                    with open(output_file) as f:
                        return json.load(f)
                else:
                    return {}
                    
            except subprocess.TimeoutExpired:
                raise TimeoutError("Script execution timed out after 5 minutes")


def segment_stardist_remote(
    image: np.ndarray,
    python_path: str,
    model_name: str = '2D_versatile_fluo',
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.4
) -> np.ndarray:
    """
    Run StarDist segmentation in a different Python environment.
    
    This allows using StarDist with Python 3.9-3.12 while main app uses 3.13+.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    python_path : str
        Path to Python executable with StarDist installed
    model_name : str
        StarDist model name
    prob_thresh : float
        Probability threshold
    nms_thresh : float
        NMS threshold
        
    Returns
    -------
    np.ndarray
        Labeled segmentation mask
        
    Examples
    --------
    >>> # First create an environment with Python 3.12 and StarDist:
    >>> # conda create -n stardist312 python=3.12
    >>> # conda activate stardist312  
    >>> # pip install stardist
    >>> 
    >>> # Then use it from Python 3.13:
    >>> python_path = "/path/to/conda/envs/stardist312/bin/python"
    >>> labels = segment_stardist_remote(image, python_path)
    """
    if not HAS_PIL:
        raise ImportError("PIL is required for remote segmentation")
    
    bridge = EnvironmentBridge(python_path=python_path)
    
    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        temp_image_path = f.name
        Image.fromarray(image).save(temp_image_path)
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        temp_output_path = f.name
    
    try:
        # Script to run in target environment
        script = f"""
from stardist.models import StarDist2D
import numpy as np
from PIL import Image

# Load image
image = np.array(Image.open(inputs['image_path']))

# Convert to grayscale if needed
if image.ndim == 3:
    image = 0.2125 * image[:,:,0] + 0.7154 * image[:,:,1] + 0.0721 * image[:,:,2]

# Normalize
image = (image - image.min()) / (image.max() - image.min() + 1e-10)

# Load model and predict
model = StarDist2D.from_pretrained(inputs['model_name'])
labels, details = model.predict_instances(
    image,
    prob_thresh=inputs['prob_thresh'],
    nms_thresh=inputs['nms_thresh']
)

# Save output
Image.fromarray(labels.astype(np.uint16)).save(inputs['output_path'])

outputs = {{
    'n_objects': int(labels.max()),
    'success': True
}}
"""
        
        # Run segmentation
        results = bridge.run_script(script, inputs={
            'image_path': temp_image_path,
            'output_path': temp_output_path,
            'model_name': model_name,
            'prob_thresh': prob_thresh,
            'nms_thresh': nms_thresh
        })
        
        # Load result
        labels = np.array(Image.open(temp_output_path))
        
        return labels
        
    finally:
        # Cleanup temp files
        for path in [temp_image_path, temp_output_path]:
            if os.path.exists(path):
                os.unlink(path)


def check_remote_environment(python_path: str, package: str) -> bool:
    """
    Check if a package is available in a remote environment.
    
    Parameters
    ----------
    python_path : str
        Path to Python executable
    package : str
        Package name to check
        
    Returns
    -------
    bool
        True if package is available
    """
    try:
        result = subprocess.run(
            [python_path, '-c', f'import {package}; print("OK")'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0 and 'OK' in result.stdout
    except:
        return False


def create_stardist_environment_guide() -> str:
    """
    Return instructions for creating a StarDist-compatible environment.
    
    Returns
    -------
    str
        Setup instructions
    """
    return """
╔════════════════════════════════════════════════════════════════╗
║  Creating StarDist Environment (Python 3.9-3.12)               
╚════════════════════════════════════════════════════════════════╝

Option 1: Using Conda (Recommended)
────────────────────────────────────

# Create new environment with Python 3.12
conda create -n stardist312 python=3.12 -y

# Activate it
conda activate stardist312

# Install StarDist
pip install stardist tensorflow

# Find Python path (save this!)
which python
# Example output: /Users/you/miniconda3/envs/stardist312/bin/python


Option 2: Using venv
────────────────────

# Install Python 3.12 (if not already installed)
# On macOS with Homebrew:
brew install python@3.12

# Create venv
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv ~/stardist312

# Activate
source ~/stardist312/bin/activate

# Install StarDist
pip install stardist tensorflow

# Python path:
~/stardist312/bin/python


Usage in CurveAlign:
────────────────────

from napari_curvealign.env_bridge import segment_stardist_remote

# Use the Python path from above
python_path = "/Users/you/miniconda3/envs/stardist312/bin/python"

# Run segmentation
labels = segment_stardist_remote(
    image,
    python_path=python_path,
    model_name='2D_versatile_fluo'
)


Or in GUI:
──────────

1. Create environment as above
2. In Segmentation tab, click "Configure StarDist Environment"
3. Paste Python path
4. Test connection
5. Now StarDist will work!

"""


# Auto-detect common StarDist environments
def find_stardist_environments() -> list:
    """
    Auto-detect Python environments that have StarDist installed.
    
    Returns
    -------
    list
        List of dicts with 'name', 'path', and 'version' keys
    """
    found = []
    
    # Check conda environments
    try:
        conda_base = os.environ.get('CONDA_PREFIX', os.path.expanduser('~/miniconda3'))
        envs_dir = Path(conda_base) / 'envs'
        if envs_dir.exists():
            for env_dir in envs_dir.iterdir():
                if not env_dir.is_dir():
                    continue
                python_path = env_dir / 'bin' / 'python'
                if python_path.exists():
                    if check_remote_environment(str(python_path), 'stardist'):
                        # Get Python version
                        result = subprocess.run(
                            [str(python_path), '--version'],
                            capture_output=True,
                            text=True
                        )
                        version = result.stdout.strip() if result.returncode == 0 else 'unknown'
                        
                        found.append({
                            'name': env_dir.name,
                            'path': str(python_path),
                            'version': version,
                            'type': 'conda'
                        })
    except:
        pass
    
    return found


if __name__ == '__main__':
    # Demo/test
    print("Environment Bridge for StarDist")
    print("="*60)
    print()
    print("Searching for StarDist environments...")
    envs = find_stardist_environments()
    
    if envs:
        print(f"\n✅ Found {len(envs)} environment(s) with StarDist:")
        for env in envs:
            print(f"  - {env['name']}: {env['path']}")
            print(f"    Python: {env['version']}")
    else:
        print("\n⚠️  No StarDist environments found.")
        print(create_stardist_environment_guide())

