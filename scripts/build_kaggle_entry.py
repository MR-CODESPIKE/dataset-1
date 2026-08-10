"""Build the single-file Kaggle entrypoint from the scripts package."""
from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "scripts"
OUTPUT = ROOT / "waxal-asr-fine-tune.py"

modules: dict[str, str] = {}
for path in sorted(PACKAGE.glob("*.py")):
    modules[path.name] = base64.b64encode(path.read_bytes()).decode("ascii")

lines = [
    "#!/usr/bin/env python3",
    '"""Self-contained Kaggle entrypoint generated from the Waxal scripts package."""',
    "from __future__ import annotations",
    "import base64",
    "import os",
    "import runpy",
    "import subprocess",
    "import sys",
    "from pathlib import Path",
    "",
    f"MODULES = {modules!r}",
    "runtime_root = Path('/kaggle/working/waxal_runtime')",
    "package_root = runtime_root / 'scripts'",
    "package_root.mkdir(parents=True, exist_ok=True)",
    "for name, encoded in MODULES.items():",
    "    (package_root / name).write_bytes(base64.b64decode(encoded))",
    "runtime_root_str = str(runtime_root)",
    "sys.path.insert(0, runtime_root_str)",
    "os.environ['PYTHONPATH'] = runtime_root_str + os.pathsep + os.environ.get('PYTHONPATH', '')",
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'evaluate', 'jiwer'])",
    "runpy.run_module('scripts.run_pipeline', run_name='__main__')",
    "",
]
OUTPUT.write_text("\n".join(lines))
print(f"Wrote {OUTPUT} with {len(modules)} embedded modules")
