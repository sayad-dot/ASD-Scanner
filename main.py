"""
ASD-Scanner: TabNet-based Cross-Dataset Detection of Autism Spectrum Disorder
Phase 0: Initial Setup and Environment Verification
"""

import os
import sys
import pandas as pd
import numpy as np

# Verify core dependencies are installed
def verify_dependencies():
    """Verify that all required packages are installed"""
    required_packages = [
        'torch', 'pytorch_tabnet', 'pandas', 'numpy', 
        'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'pytorch_tabnet':
                import pytorch_tabnet
            elif package == 'torch':
                import torch
            elif package == 'pandas':
                import pandas
            elif package == 'numpy':
                import numpy
            elif package == 'scikit-learn':
                import sklearn
            elif package == 'matplotlib':
                import matplotlib
            elif package == 'seaborn':
                import seaborn
            print(f"✓ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")
    
    return missing_packages

def check_project_structure():
    """Verify project directory structure exists"""
    directories = [
        'data/raw', 'data/processed', 'data/external',
        'src/models', 'src/preprocessing', 'src/evaluation', 'src/utils',
        'notebooks/exploratory', 'notebooks/modeling', 'notebooks/results',
        'models/saved_models', 'models/checkpoints',
        'results/figures', 'results/metrics', 'results/reports',
        'docs', 'tests', 'configs'
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ {directory}/ - EXISTS")
        else:
            print(f"✗ {directory}/ - MISSING")
            os.makedirs(directory, exist_ok=True)
            print(f"  Created: {directory}/")

def main():
    """Main execution function for Phase 0"""
    print("=" * 60)
    print("ASD-Scanner: TabNet Cross-Dataset ASD Detection")
    print("Phase 0: Project Setup and Environment Verification")
    print("=" * 60)
    
    print("\n1. Checking Dependencies...")
    missing = verify_dependencies()
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Please install them using:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        return False
    
    print("\n2. Verifying Project Structure...")
    check_project_structure()
    
    print("\n3. Environment Information:")
    print(f"   Python Version: {sys.version.split()[0]}")
    print(f"   Working Directory: {os.getcwd()}")
    try:
        import torch
        print(f"   PyTorch Available: {torch.cuda.is_available()}")
    except ImportError:
        print("   PyTorch Available: Not installed")
    
    print("\n✅ Phase 0 Setup Complete!")
    print("\nNext Steps:")
    print("  - Move your ASD dataset CSV files to data/raw/")
    print("  - Ready to proceed to Phase 1: Data Analysis & Preprocessing")
    
    return True

if __name__ == "__main__":
    main()
