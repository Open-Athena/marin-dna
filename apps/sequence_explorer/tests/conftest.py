import sys
from pathlib import Path


APP_DIRECTORY = Path(__file__).parents[1]
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))
