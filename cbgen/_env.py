import os
from pathlib import Path

from appdirs import user_cache_dir

__all__ = ["BGEN_CACHE_HOME"] #!!!cmk check that this is documented

BGEN_CACHE_HOME = Path(
    os.environ.get("BGEN_CACHE_HOME", default=Path(user_cache_dir("bgen", "limix"))) 
)

(BGEN_CACHE_HOME / "test_data").mkdir(parents=True, exist_ok=True)
