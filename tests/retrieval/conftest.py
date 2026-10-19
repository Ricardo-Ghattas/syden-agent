import shutil
from pathlib import Path

import pytest

from app.retrieval import DocumentIndex

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def processed(tmp_path):
    folder = tmp_path / "processed"
    folder.mkdir()
    for path in FIXTURES.glob("*.jsonl"):
        shutil.copyfile(path, folder / path.name)
    return folder


@pytest.fixture
def index(processed):
    return DocumentIndex(processed)
