"""Test runner script for running every test file individually."""
import subprocess
import sys
import os

def run_tests():
    tests_dir = "tests"
    test_files = [f for f in os.listdir(tests_dir) if f.startswith("test_") and f.endswith(".py")]
    
    all_passed = True
    for tf in sorted(test_files):
        print(f"\n{'='*60}")
        print(f"  Testing: {tf}")
        print(f"{'='*60}")
        
        result = subprocess.run(
            [sys.executable, "-m", "pytest", os.path.join(tests_dir, tf), "-v"],
            cwd=os.getcwd(),
            capture_output=False,
        )
        if result.returncode != 0:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("  🎉 All tests passed!")
    else:
        print("  ❌ Some tests failed")
    print(f"{'='*60}\n")
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    run_tests()
