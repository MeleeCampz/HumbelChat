"""
run_tests.sh  - Run every test file individually with --no-cov style isolation.
Exit immediately if any test returns non-zero.
"""
set -euo pipefail
cd "$(dirname "$0")"

echo "Running all bot tests …"
failed=()

for f in tests/test_*.py; do
    echo ""
    echo "═══════════════════════════════════════"
    echo "  Testing $(basename "$f")"
    echo "═══════════════════════════════════════"
    if python -m pytest "$f" -v --tb=short; then
        echo "✅ PASS"
    else
        echo "❌ FAIL"
        failed+=("$f")
    fi
done

echo ""
if ${failed[@]:+true}; then
    echo "FAILED tests:"
    for f in "${failed[@]}"; do echo "  - $f"; done
    exit 1
else
    echo "🎉 All tests passed!"
fi
