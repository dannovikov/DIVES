@echo off

:: Usage: run_subtyping_windows.bat <path_to_fasta_file>

:: verify that the correct number of arguments were passed
if "%~1"=="" (
    echo Error: No FASTA file provided.
    echo Usage: run_subtyping_windows.bat ^<path_to_fasta_file^>
    exit /b 1
)


:: Assigning the command line argument to a variable
set "FASTA_FILE=%~1"
echo Running DIVES inference on %FASTA_FILE%...

:: Extract the base name of the FASTA file without the extension
for %%I in ("%FASTA_FILE%") do set "FASTA_BASE=%%~nI"

:: Define the data directory and output directory
set "DATA_DIR=.\preproc_%FASTA_BASE%"
set "OUTPUT_DIR=.\output"

:: Ensure directories exist
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Copy the map_label_to_subtype.pkl to the data directory
echo Copying map_label_to_subtype.pkl to %DATA_DIR%...
copy map_label_to_subtype.pkl "%DATA_DIR%"

:: Preprocessing step
echo Running preprocessing on %FASTA_FILE%...
python preproc_inference.py "%FASTA_FILE%" "%DATA_DIR%"

:: Inference step
echo Running DIVES inference...
python DIVES_inference.py "%DATA_DIR%" "%OUTPUT_DIR%"

echo Inference complete. Check the output directory for results.
