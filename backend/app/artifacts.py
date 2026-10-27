from contextlib import contextmanager
from uuid import uuid4
import shutil
from .config import DATA


@contextmanager
def artifact_directory(kind):
    """Remove only the newly allocated directory if its operation fails or is cancelled."""
    identifier = uuid4().hex
    directory = DATA / kind / identifier
    directory.mkdir()
    try:
        yield identifier, directory
    except BaseException:
        shutil.rmtree(directory)
        raise
