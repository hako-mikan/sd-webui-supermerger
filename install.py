import launch

if not launch.is_installed("sklearn"):
    launch.run_pip("install scikit-learn", "scikit-learn")

if not launch.is_installed("diffusers"):
    launch.run_pip("install diffusers", "diffusers")
