from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

PHASE2_PIPELINE = Path(__file__).resolve().parents[1] / "phase2" / "pipeline.py"
_spec = spec_from_file_location("phase2_pipeline", PHASE2_PIPELINE)
_module = module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_module)

for _name in dir(_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_module, _name)
