#!/bin/bash
# Easy local way to run the linting + format check pre-commit :) 
echo "Running flake8..."
flake8 src utils ara_loop.py

echo "Checking formatting with Black..."
black --check .