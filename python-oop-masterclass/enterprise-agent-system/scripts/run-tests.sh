#!/bin/bash

# ============================================================================
# Test Execution Script
# ============================================================================

set -e

echo "Running Test Suite"
echo "=================="

# Default options
TEST_TYPE="all"
COVERAGE=true
PARALLEL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --e2e)
            TEST_TYPE="e2e"
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run-tests.sh [--unit|--integration|--e2e] [--no-coverage] [--parallel]"
            exit 1
            ;;
    esac
done

# Build test command
CMD="pytest"

# Add test markers
case $TEST_TYPE in
    unit)
        CMD="$CMD -m unit"
        echo "Running unit tests only"
        ;;
    integration)
        CMD="$CMD -m integration"
        echo "Running integration tests only"
        ;;
    e2e)
        CMD="$CMD -m e2e"
        echo "Running end-to-end tests only"
        ;;
    *)
        echo "Running all tests"
        ;;
esac

# Add coverage
if [ "$COVERAGE" = true ]; then
    CMD="$CMD --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml"
fi

# Add parallel execution
if [ "$PARALLEL" = true ]; then
    CMD="$CMD -n auto"
    echo "Running tests in parallel"
fi

# Add verbose output
CMD="$CMD -v"

echo ""
echo "Command: $CMD"
echo ""

# Run tests
$CMD

# Display results
echo ""
echo "=================="
echo "Tests completed!"

if [ "$COVERAGE" = true ]; then
    echo ""
    echo "Coverage report generated at: htmlcov/index.html"
    echo "To view: open htmlcov/index.html"
fi

echo ""
