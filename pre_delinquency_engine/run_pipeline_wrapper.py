"""Wrapper to suppress NumPy warnings and run pipeline"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress specific NumPy warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Now import and run the pipeline
from run_pipeline import main

if __name__ == "__main__":
    main()
