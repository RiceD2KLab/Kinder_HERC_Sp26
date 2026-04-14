import os
import subprocess
from pathlib import Path

#setup paths
input_dir = Path("Kinder_HERC_Sp26/transcripts_to_label/hisd_transcripts") #change path here as needed
output_dir = Path("Kinder_HERC_Sp26/transcripts_to_label/chunked_transcripts") #change path here as needed
chunk_script = "Kinder_HERC_Sp26/transcript_chunking/create_chunks.py"

def run_batch():
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