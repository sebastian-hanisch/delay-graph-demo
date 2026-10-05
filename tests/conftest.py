import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: langsame Tests (Wochenlauf); die Push-CI lässt sie mit -m \"not slow\" aus")
