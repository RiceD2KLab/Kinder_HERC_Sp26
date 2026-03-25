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
    """Reads the content of a file using UTF-8 encoding."""
    return Path(path).read_text(encoding="utf-8")

def replace_numbers(text):
    """
    Finds digits in text and converts them to words.
    """
    return re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)

def normalize_text(text):
    """
    Standardizes text to minimize 'false' errors during comparison.
    
    Processing steps:
    1. Lowercase conversion.
    2. Expand contractions.
    3. Replace numbers to be written out.
    4. Removal of bracketed timestamps and non-speech tags.
    5. Removal of speaker change symbols (>>).
    6. Normalization of 'am/pm' and time formats.
    7. Stripping of filler words (uh, um, etc.) and punctuation.
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
    """Load reference and hypothesis files, normalize, compute WER, and print results."""
    if len(sys.argv) != 3:
        print("Usage: python wer_norm.py reference.txt hypothesis.txt")
        return

    # Load and Normalize
    reference = normalize_text(load_text(sys.argv[1]))
    hypothesis = normalize_text(load_text(sys.argv[2]))

    # Process words (This returns a WordCleanReport in v4.0.0)
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