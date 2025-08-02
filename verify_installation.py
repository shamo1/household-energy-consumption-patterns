#!/usr/bin/env python3
"""
Household Energy Consumption Analysis - Installation Verification Script
This script verifies that all required packages are properly installed.
"""

import sys
import importlib
from packaging import version

# Required packages with minimum versions
REQUIRED_PACKAGES = {
    'pandas': '1.5.0',
    'numpy': '1.21.0',
    'matplotlib': '3.5.0',
    'seaborn': '0.11.0',
    'plotly': '5.0.0',
    'sklearn': '1.1.0',
    'tensorflow': '2.10.0',
    'keras': '2.10.0',
    'statsmodels': '0.13.0',
    'kerastuner': '1.0.0',
    'prophet': '1.1.0',
    'jupyter': '1.0.0',
    'IPython': '6.0.0'
}

def check_package(package_name, min_version):
    """Check if a package is installed and meets minimum version requirement."""
    try:
        # Handle special cases
        if package_name == 'sklearn':
            import sklearn
            installed_version = sklearn.__version__
        elif package_name == 'jupyter':
            # For jupyter, just check if it's installed
            import jupyter
            print(f"✅ {package_name}: Installed (package manager)")
            return True
        else:
            module = importlib.import_module(package_name)
            installed_version = module.__version__
        
        # Compare versions (skip for jupyter)
        if package_name != 'jupyter':
            if version.parse(installed_version) >= version.parse(min_version):
                print(f"✅ {package_name}: {installed_version} (>= {min_version})")
                return True
            else:
                print(f"⚠️  {package_name}: {installed_version} (< {min_version}) - UPDATE NEEDED")
                return False
        
    except ImportError:
        print(f"❌ {package_name}: NOT INSTALLED")
        return False
    except AttributeError:
        print(f"⚠️  {package_name}: Installed but version unknown")
        return False

def verify_tensorflow_gpu():
    """Check if TensorFlow can detect GPU."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 TensorFlow GPU Support: {len(gpus)} GPU(s) detected")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("💻 TensorFlow: CPU-only mode (no GPU detected)")
    except Exception as e:
        print(f"⚠️  TensorFlow GPU check failed: {e}")

def main():
    """Main verification function."""
    print("🔍 Household Energy Analysis - Package Verification")
    print("=" * 55)
    print()
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("⚠️  Warning: Python 3.8+ recommended")
    else:
        print("✅ Python version OK")
    
    print()
    print("📦 Package Verification:")
    print("-" * 25)
    
    # Check all required packages
    all_good = True
    for package, min_ver in REQUIRED_PACKAGES.items():
        if not check_package(package, min_ver):
            all_good = False
    
    print()
    
    # Special checks
    print("🔧 Additional Checks:")
    print("-" * 20)
    verify_tensorflow_gpu()
    
    # Test data science workflow
    try:
        import pandas as pd
        import numpy as np
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='15min'),
            'consumption': np.random.randn(100)
        })
        
        # Basic operations
        test_data['consumption'].mean()
        test_data.set_index('timestamp').resample('h').mean()
        
        print("✅ Basic data operations: Working")
        
    except Exception as e:
        print(f"❌ Basic data operations: Failed - {e}")
        all_good = False
    
    # Test ML workflow
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        
        # Simple ML test
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mean_squared_error(y_test, y_pred)
        
        print("✅ Machine Learning workflow: Working")
        
    except Exception as e:
        print(f"❌ Machine Learning workflow: Failed - {e}")
        all_good = False
    
    print()
    print("📊 Summary:")
    print("-" * 10)
    
    if all_good:
        print("🎉 All packages verified successfully!")
        print("📓 You can now run the Jupyter notebook: 2212172.ipynb")
        print()
        print("💡 Quick start:")
        print("   1. jupyter notebook")
        print("   2. Open 2212172.ipynb")
        print("   3. Run all cells")
    else:
        print("⚠️  Some packages need attention. Please:")
        print("   1. Check the installation log above")
        print("   2. Reinstall missing/outdated packages")
        print("   3. Run this verification script again")
        print()
        print("💡 To fix issues:")
        print("   pip install --upgrade <package-name>")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
