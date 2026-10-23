"""Run SeedPack from a source checkout: python varc.py ... (the installed command is `seedpack`)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seedpack.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
