"""Alternative wrapper with deep NumPy warning suppression"""
import os
import sys

# Set environment variables before any imports
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

# Suppress all warnings
import warnings
warnings.filterwarnings('ignore')

# Try to import numpy and patch the issue
try:
    import numpy as np
    print(f"NumPy {np.__version__} imported successfully")
    
    # Now try to run the pipeline
    print("Starting pipeline...")
    from run_pipeline import main
    main()
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
