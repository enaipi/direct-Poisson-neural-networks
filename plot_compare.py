#!/usr/bin/env python3
"""
Pure Python script for comparing and plotting Poisson structure learning results.

Usage:
    python plot_compare.py --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST

Replaces: plot-compare --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST
"""

import argparse
from pathlib import Path
import sys

# Add src to path so we can import dpnn
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dpnn.postprocessing.plot_compare import main

if __name__ == "__main__":
    main()
