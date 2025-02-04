import os
import subprocess
import sys

def build_with_make():
    # Ensure we're in the correct directory (root of the project with Makefile)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    try:
        subprocess.run(["make"])
    except subprocess.CalledProcessError as e:
        print("Error: make failed with exit code", e.returncode)
        sys.exit(1)

    try:
        subprocess.run(["make", "test"])
    except subprocess.CalledProcessError as e:
        print("Error: make test failed with exit code", e.returncode)
        sys.exit(1)
