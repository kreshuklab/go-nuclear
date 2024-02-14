from pathlib import Path

# Find the global path of run-stardist
repo_global_path = Path(__file__).parent.absolute()

# Create configs directory at startup with pathlib
path_home = Path.home()
path_dir_models = path_home / ".runstardist"
path_dir_models.mkdir(exist_ok=True)
