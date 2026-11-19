"""Compatibility entry point; implementation lives in app.workflows.experiment."""
from app.workflows.experiment import RUN_DIR, SEED, SIZE, main, make_split, report_metrics, select_thresholds, write_json

if __name__ == '__main__':
    main()
