"""Verify smoke_test.py exits 0."""
import subprocess
import sys
from pathlib import Path


def test_smoke_pipeline():
    smoke_script = Path(__file__).parent.parent / "scripts" / "smoke_test.py"
    result = subprocess.run([sys.executable, str(smoke_script)], capture_output=False)
    assert result.returncode == 0, "smoke_test.py did not exit 0"
