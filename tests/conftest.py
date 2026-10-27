import os
import sys
import shutil
import warnings
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "data" / "tests" / uuid4().hex
os.environ["BOB_DATA"] = str(TEST_DATA)
os.environ["BOB_DEVICE"] = "cpu"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(ROOT / "backend"))


def pytest_sessionfinish(session, exitstatus):
    """Generated model weights must not silently accumulate with each test run."""
    if os.environ.get("BOB_KEEP_TEST_DATA") == "1":
        return
    api = sys.modules.get("app.api")
    if api is not None and hasattr(api, "jobs"):
        api.jobs.executor.shutdown(wait=True)
    target = TEST_DATA.resolve()
    expected_parent = (ROOT / "data" / "tests").resolve()
    if (
        not expected_parent.is_relative_to(ROOT.resolve())
        or target.parent != expected_parent
        or target.name != TEST_DATA.name
        or TEST_DATA.is_symlink()
    ):
        warnings.warn("Test data cleanup refused an unexpected path")
        return
    try:
        if target.exists():
            shutil.rmtree(target)
    except OSError as error:
        warnings.warn(f"Test data remains at {target}: {error}")
