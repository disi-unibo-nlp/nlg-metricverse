import os.path
from pathlib import Path

TESTS_NLGMETRICVERSE_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
TESTS_DIR = TESTS_NLGMETRICVERSE_DIR.parent
TEST_DATA_DIR = TESTS_DIR / "test_data"
EXPECTED_OUTPUTS = TEST_DATA_DIR / "expected_outputs"
