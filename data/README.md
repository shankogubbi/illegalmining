# Data

Active labels: `raw/train_v2.csv`. Active training images: `raw/train-jpg/<image_name>.jpg`. Labels are space-separated tags.

Other folders under `raw/` are retained downloads/extracted sources and test imagery. The application paths are set in `app/utils/config.py`; these source folders are not silently interchanged. Download via the GUI Setup tab. Dataset audit: `python -m app.workflows.audit`. Large raw files and credentials are excluded from Git.
