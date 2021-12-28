#!/bin/bash

bash generate_data.sh

python run_tests.py

echo "Deleting test data."
rm -r data

