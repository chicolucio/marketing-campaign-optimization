import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

CURRENT_FOLDER = Path(__file__).resolve()
STYLE_FILE = CURRENT_FOLDER.parent / "flsbustamante.mplstyle"
ENV_FILE = CURRENT_FOLDER.parent / ".env"

plt.style.use(STYLE_FILE)

load_dotenv(ENV_FILE)

MLFLOW_ON = os.getenv("MLFLOW_ON") == "True"


if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
