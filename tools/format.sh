#!/bin/bash
# This script will apply black and isort to all scripts.

# apply black
black .

# apply isort
isort .