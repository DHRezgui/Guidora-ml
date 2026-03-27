# ML Environment Setup Script for LightGBM Abandonment Prediction Model

"""
Setup script to initialize the ML environment and verify LightGBM installation.
Usage: python setup_ml_env.py
"""

import sys
import subprocess
import os
from pathlib import Path


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def check_python_version() -> bool:
    """Check if Python version meets requirements (3.9+)."""
    print(" Checking Python version...")
    version = sys.version_info
    min_version = (3, 9)
    
    if version >= min_version:
        print(f" Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f" Python {version.major}.{version.minor}.{version.micro} - FAILED")
        print(f"   Required: Python {min_version[0]}.{min_version[1]}+")
        return False


def install_dependencies(requirements_file: str = "requirements.txt") -> bool:
    """Install dependencies from requirements.txt."""
    print_section("Installing Dependencies")
    
    if not Path(requirements_file).exists():
        print(f" {requirements_file} not found!")
        return False
    
    print(f" Installing packages from {requirements_file}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--upgrade", "pip", "setuptools", "wheel"
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", requirements_file
        ])
        print(" Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Installation failed: {e}")
        return False


def verify_lightgbm() -> bool:
    """Verify LightGBM installation and check version."""
    print_section("Verifying LightGBM Installation")
    
    try:
        import lightgbm
        version = lightgbm.__version__
        print(f" LightGBM {version} is installed")
        
        # Print LightGBM build info
        print("\n LightGBM Build Information:")
        print(f"   Binary: {lightgbm.LGBMClassifier}")
        return True
    except ImportError:
        print(" LightGBM is not installed")
        return False


def verify_sentence_transformers() -> bool:
    """Verify sentence-transformers installation and check version."""
    print_section("Verifying sentence-transformers Installation")

    try:
        import sentence_transformers

        version = sentence_transformers.__version__
        print(f" sentence-transformers {version} is installed")
        print("   Embedding generation for semantic FAQ search is available")
        return True
    except ImportError:
        print(" sentence-transformers is not installed")
        return False


def verify_dependencies() -> bool:
    """Verify all key dependencies are installed."""
    print_section("Verifying Key Dependencies")
    
    dependencies = {
        "numpy": "NumPy",
        "pandas": "Pandas",
        "sklearn": "Scikit-learn",
        "scipy": "SciPy",
        "matplotlib": "Matplotlib",
        "psycopg2": "PostgreSQL adapter",
        "sqlalchemy": "SQLAlchemy",
        "sentence_transformers": "Sentence Transformers",
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            version = __import__(module).__version__
            print(f" {name:20} {version}")
        except ImportError:
            print(f" {name:20} NOT INSTALLED")
            all_ok = False
        except AttributeError:
            print(f" {name:20} (version not available)")
    
    return all_ok


def create_directories() -> bool:
    """Create necessary output directories."""
    print_section("Creating Project Directories")
    
    dirs = ["models", "data", "results", "notebooks/exploratory", "training", "rag"]
    
    try:
        for dir_name in dirs:
            path = Path(dir_name)
            path.mkdir(parents=True, exist_ok=True)
            print(f" {dir_name}")
        return True
    except Exception as e:
        print(f" Error creating directories: {e}")
        return False


def verify_database_connection() -> bool:
    """Test database connection configuration."""
    print_section("Testing Database Configuration")
    
    try:
        import psycopg2
        print(" PostgreSQL adapter available")
        print("   Note: Database connection will be tested during model training")
        return True
    except ImportError:
        print(" PostgreSQL adapter not found")
        return False


def generate_summary(checks: dict) -> None:
    """Generate setup summary."""
    print_section("Setup Summary")
    
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    for check_name, result in checks.items():
        symbol = "" if result else ""
        print(f"{symbol} {check_name}")
    
    print(f"\n{'=' * 60}")
    if passed == total:
        print(f" All checks passed! ({passed}/{total})")
        print("Your ML environment is ready for LightGBM model training.")
    else:
        print(f"️  {total - passed}/{total} checks failed. Please review the issues above.")
    print(f"{'=' * 60}\n")


def main():
    """Run the full setup process."""
    print("\n")
    print(" LightGBM ML Environment Setup")
    print("=" * 60)
    
    checks = {}
    
    # Pre-installation checks
    checks["Python Version"] = check_python_version()
    if not checks["Python Version"]:
        print("\n Setup aborted: Python version is too old")
        sys.exit(1)
    
    # Install dependencies
    checks["Dependencies Installation"] = install_dependencies()
    
    # Post-installation verification
    checks["LightGBM Verification"] = verify_lightgbm()
    checks["Sentence Transformers Verification"] = verify_sentence_transformers()
    checks["Core Dependencies"] = verify_dependencies()
    checks["Database Adapter"] = verify_database_connection()
    checks["Directory Structure"] = create_directories()
    
    # Generate summary
    generate_summary(checks)
    
    # Exit status
    if all(checks.values()):
        print("Next steps:")
        print("  1. Review config.py for model hyperparameters")
        print("  2. Run 'jupyter lab' to start the notebook server")
        print("  3. Review notebooks/exploratory/ for EDA")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

