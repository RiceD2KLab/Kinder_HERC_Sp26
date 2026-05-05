"""Batch chunking helper — runs create_chunks.py on every transcript in a directory.

Edit ``input_dir`` and ``output_dir`` at the top of this file before running.

Usage:
    python mult_chunk.py
"""

import os
import subprocess
from pathlib import Path

#setup paths
input_dir = Path("Kinder_HERC_Sp26/transcripts_to_label/hisd_transcripts") #change path here as needed
output_dir = Path("Kinder_HERC_Sp26/transcripts_to_label/chunked_transcripts") #change path here as needed
chunk_script = "Kinder_HERC_Sp26/transcript_chunking/create_chunks.py"

def run_batch():
    """Chunk every ``.txt`` transcript in ``input_dir`` and write CSVs to ``output_dir``.

    Calls ``create_chunks.py`` via subprocess for each file so the chunking
    logic stays in one place and both tools remain usable independently.

    Inputs
    ------
    None — reads the module-level ``input_dir``, ``output_dir``, and
           ``chunk_script`` path constants.

    Outputs
    -------
    None — one CSV per transcript is written to ``output_dir``.
           Prints a progress line for each file processed.
    """
    #loop through every .txt file in the input folder
    for txt_file in input_dir.glob("*.txt"):
        #create a matching CSV filename
        csv_filename = txt_file.stem + ".csv"
        output_path = output_dir / csv_filename
        
        print(f"Processing: {txt_file.name}...") #sanity check
        
        #call chunking script
        subprocess.run(["python", chunk_script, "--input", str(txt_file), "--output", str(output_path)], check=True)

if __name__ == "__main__":
    run_batch()
    print("\n All files processed.")