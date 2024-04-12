#!/bin/bash
echo "Running subtyping pipeline..."

# Usage: ./run_subtyping.sh <path_to_fasta_file>

# Check if the correct number of arguments was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_fasta_file>"
    exit 1
fi

# Assigning the command line argument to a variable
FASTA_FILE=$1

# Define the data directory and output directory
DATA_DIR="./data"
OUTPUT_DIR="./output"

# Ensure directories exist
mkdir -p $DATA_DIR
mkdir -p $OUTPUT_DIR

cp map_label_to_subtype.pkl $DATA_DIR
# Preprocessing step
echo "Running preprocessing on $FASTA_FILE..."
python preproc_inference.py "$FASTA_FILE" "$DATA_DIR"

# Inference step
echo "Running DIVES inference..."
python DIVES_inference.py "$DATA_DIR" "$OUTPUT_DIR"

echo "Inference complete. Check the output directory for results."
