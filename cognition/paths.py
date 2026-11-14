"""Anchor default persisted-state paths to the package root, not the
process's current working directory.

Every cognition module's default filepath was a bare relative string
("results/uil_X.json"), resolved against cwd at runtime. That's silently
wrong as soon as something runs this package from a different directory
-- which the webui server must do (uvicorn needs cwd=Downloads/, the
package's parent, for "figpso_uil.X" imports to resolve). The result:
every run started through the browser UI was writing into a second,
disconnected Downloads/results/ directory instead of figpso_uil/results/,
fragmenting research data with no warning (confirmed empirically).
"""

import os

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PACKAGE_ROOT, "results")


def results_path(filename: str) -> str:
    return os.path.join(RESULTS_DIR, filename)
