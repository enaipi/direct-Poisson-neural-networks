#!/usr/bin/env python3
"""
Pure Python wrapper for comparison script.

Usage:
    python comparison.py --model=RB --steps=100 --implicit --soft --without --folder_name=TEST
    
or as a library:
    from src.dpnn.comparison import ComparisonConfig, ComparisonRunner
    config = ComparisonConfig(model="RB", methods=["implicit", "soft"])
    runner = ComparisonRunner(config)
    runner.run()
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dpnn.comparison import main

if __name__ == "__main__":
    main()
