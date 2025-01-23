import launch
import importlib
from packaging.version import Version
from packaging.requirements import Requirement

def is_installed(pip_package):
    """
    Check if a package is installed and meets version requirements specified in pip-style format.

    Args:
        pip_package (str): Package name in pip-style format (e.g., "numpy>=1.22.0").
    
    Returns:
        bool: True if the package is installed and meets the version requirement, False otherwise.
    """
    try:
        # Parse the pip-style package name and version constraints
        requirement = Requirement(pip_package)
        package_name = requirement.name
        specifier = requirement.specifier  # e.g., >=1.22.0
        
        # Check if the package is installed
        dist = importlib.metadata.distribution(package_name)
        installed_version = Version(dist.version)
        
        # Check version constraints
        if specifier.contains(installed_version):
            return True
        else:
            print(f"Installed version of {package_name} ({installed_version}) does not satisfy the requirement ({specifier}).")
            return False
    except importlib.metadata.PackageNotFoundError:
        print(f"Package {pip_package} is not installed.")
        return False
    
requirements = [
"diffusers==0.31.0",
"scikit-learn",
]

for module in requirements:
    if not is_installed(module):
        launch.run_pip(f"install {module}", module)