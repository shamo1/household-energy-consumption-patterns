#!/usr/bin/env python3
"""
Project Initialization Script

This script helps set up the project environment and validates the setup.
"""

import os
import sys
import subprocess
from pathlib import Path


def create_directories():
    """Create necessary project directories."""
    directories = [
        'models/saved_models',
        'results/plots',
        'visualizations/charts',
        'data/processed',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def check_data_file():
    """Check if the data file exists."""
    data_file = Path('data/household_data_15min_singleindex.csv')
    if data_file.exists():
        print(f"✅ Data file found: {data_file}")
        return True
    else:
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure the dataset is placed in the data/ directory")
        return False


def run_verification():
    """Run the installation verification script."""
    try:
        result = subprocess.run([sys.executable, 'verify_installation.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Installation verification passed")
            return True
        else:
            print("❌ Installation verification failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running verification: {e}")
        return False


def main():
    """Main initialization function."""
    print("🏠⚡ Household Energy Analysis - Project Initialization")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path('README.md').exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    # Check data file
    print("\n📊 Checking data availability...")
    data_ok = check_data_file()
    
    # Run verification
    print("\n🔍 Running installation verification...")
    install_ok = run_verification()
    
    # Summary
    print("\n📋 Initialization Summary:")
    print("-" * 30)
    
    if data_ok and install_ok:
        print("🎉 Project setup complete!")
        print("\n💡 Next steps:")
        print("1. jupyter notebook")
        print("2. Open: notebooks/household_energy_analysis.ipynb")
        print("3. Run all cells to start analysis")
    else:
        print("⚠️  Setup incomplete. Please address the issues above.")
        if not data_ok:
            print("   - Add dataset to data/ directory")
        if not install_ok:
            print("   - Fix installation issues")
    
    print("\n📚 Documentation available:")
    print("   - README.md: Project overview")
    print("   - docs/methodology.md: Technical details")
    print("   - docs/data_dictionary.md: Dataset information")


if __name__ == "__main__":
    main()
