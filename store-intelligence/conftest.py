"""
Root conftest.py — ensures store-intelligence/ is on PYTHONPATH for all tests.
"""
import sys
from pathlib import Path

# Add the project root to sys.path so `from app.xxx import ...` works
sys.path.insert(0, str(Path(__file__).parent))
