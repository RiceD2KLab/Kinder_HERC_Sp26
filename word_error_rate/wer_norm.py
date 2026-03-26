"""Calculate Word Error Rate (WER) between a reference and hypothesis transcript.

Applies normalization (lowercasing, contraction expansion, number-to-word
conversion, punctuation removal, etc.) before comparison to minimize false
errors from formatting differences.

Usage:
    python wer_norm.py reference.txt hypothesis.txt
"""

import re
import sys
from pathlib import Path
from collections import Counter
import jiwer
import contractions
from num2words import num2words

def load_text(path):
    """Read the full content of a text file.

    Inputs
    ------
    path : str
        File path to read.

    Outputs
    -------
    str
        The file's content as a single string.
    """
    return Path(path).read_text(encoding="utf-8")

def replace_numbers(text):
    """Convert all digit sequences in text to their word equivalents.

    Inputs
    ------
    text : str
        Input text potentially containing numbers.

    Outputs
    -------
    str
        Text with digits replaced by words (e.g. "50" -> "fifty").
    """
    return re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)

def normalize_text(text):
    """Standardize text to minimize false errors during WER comparison.

    Applies lowercasing, contraction expansion, number-to-word conversion,
    timestamp/tag removal, speaker indicator removal, time format normalization,
    filler word removal, percent expansion, punctuation removal, and title
    normalization.

    Inputs
    ------
    text : str
        Raw transcript text (reference or hypothesis).

    Outputs
    -------
    str
        Normalized text ready for WER comparison.
    """
    # lowercase
    text = text.lower()
    
    # Expand contractions (e.g., "I'm" -> "I am", "won't" -> "will not")
    text = contractions.fix(text)

    # Replace numbers to be written out ("50 percent" -> "fifty percent")
    text = replace_numbers(text)

    # Remove timestamps and bracketed tags
    text = re.sub(r"\[\d{2}:\d{2}(?::\d{2})?(?:[-–]\d{2}:\d{2}(?::\d{2})?)?\]", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)

    # Remove speaker indicators
    text = re.sub(r">>?", " ", text)

    # Normalize a.m. / p.m. and time formats
    text = re.sub(r"\ba\.?\s*m\.?\b", " am ", text)
    text = re.sub(r"\bp\.?\s*m\.?\b", " pm ", text)
    text = re.sub(r"(\d+):(\d+)", r"\1 \2", text)

    # Remove filler words
    text = re.sub(r"\b(uh|um|erm|ah)\b", " ", text)

    # Normalize percent (% --> percent)
    text = re.sub(r"\%", "percent", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize titles (Ms --> Miss)
    text = re.sub(r"ms", "miss", text)

    return text


def main():
    """Run the WER calculation pipeline from the command line.

    Reads two file paths from sys.argv, normalizes both texts, computes
    WER using jiwer, and prints summary metrics with top-10 error breakdowns.

    Inputs
    ------
    sys.argv[1] : str
        Path to the reference (ground truth) transcript file.
    sys.argv[2] : str
        Path to the hypothesis (ASR-generated) transcript file.

    Outputs
    -------
    None
        Results are printed to stdout: WER score, substitution/deletion/insertion
        counts, and top-10 most frequent errors per category.
    """
    if len(sys.argv) != 3:
        print("Usage: python wer_norm.py reference.txt hypothesis.txt")
        return

    # Load and Normalize
    reference = normalize_text(load_text(sys.argv[1]))
    hypothesis = normalize_text(load_text(sys.argv[2]))

    # Process words
    out = jiwer.process_words(reference, hypothesis)
    
    # Extract Error Counts from Alignment Chunks
    subs, dels, ins = Counter(), Counter(), Counter()

    # We iterate through the alignment segments
    for chunk in out.alignments[0]:
        # Reference tokens and Hypothesis tokens are lists of words
        ref_tokens = out.references[0]
        hyp_tokens = out.hypotheses[0]

        if chunk.type == 'substitute':
            r = " ".join(ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx])
            h = " ".join(hyp_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx])
            subs[f"{r} -> {h}"] += 1
        elif chunk.type == 'delete':
            r = " ".join(ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx])
            dels[r] += 1
        elif chunk.type == 'insert':
            h = " ".join(hyp_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx])
            ins[h] += 1

    # Output Formatting
    print("===== SUMMARY METRICS =====")
    print(f"Word Error Rate: {out.wer:.4f}")
    print(f"Substitutions: {out.substitutions} | Deletions: {out.deletions} | Insertions: {out.insertions}\n")

    for label, counter in [
        ("Substitutions (Ref -> Hyp)", subs), 
        ("Deletions (Missing)", dels), 
        ("Insertions (Extra)", ins)
    ]:
        print(f"--- Top 10 {label} ---")
        for item, count in counter.most_common(10):
            print(f"{count}x: {item}")
        if not counter: print("None")
        print()

if __name__ == "__main__":
    main()