#!/bin/bash
# V1 Complete Test Suite Runner
# Runs all validation and stress tests

echo "================================================================================"
echo "                    V1 COMPLETE TEST SUITE"
echo "================================================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Track results
TOTAL_PASS=0
TOTAL_FAIL=0

echo "Running Core Validation Suite (22 tests)..."
echo "--------------------------------------------------------------------------------"
python3 v1/tests/test_validation_suite.py
VALIDATION_EXIT=$?

if [ $VALIDATION_EXIT -eq 0 ]; then
    echo ""
    echo "✅ Core Validation: ALL TESTS PASSED"
    TOTAL_PASS=$((TOTAL_PASS + 22))
else
    echo ""
    echo "❌ Core Validation: SOME TESTS FAILED"
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
fi

echo ""
echo ""
echo "Running Stress Test Suite (8 tests)..."
echo "--------------------------------------------------------------------------------"
python3 v1/tests/test_stress_suite.py
STRESS_EXIT=$?

if [ $STRESS_EXIT -eq 0 ]; then
    echo ""
    echo "✅ Stress Tests: ALL TESTS PASSED"
    TOTAL_PASS=$((TOTAL_PASS + 8))
else
    echo ""
    echo "❌ Stress Tests: SOME TESTS FAILED"
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
fi

# Final summary
echo ""
echo "================================================================================"
echo "                         FINAL TEST SUMMARY"
echo "================================================================================"
echo ""
echo "Core Validation:  22 tests"
echo "Stress Tests:      8 tests"
echo "================================================================================"
echo "Total Tests:      30"
echo ""

if [ $TOTAL_FAIL -eq 0 ]; then
    echo "✅ ALL TESTS PASSED (100%)"
    echo ""
    echo "V1 is fully validated and ready for production!"
    exit 0
else
    echo "❌ SOME TESTS FAILED"
    echo ""
    echo "Please review the test output above for details."
    exit 1
fi
