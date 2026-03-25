# Word Error Rate — Transcription Quality Evaluation

Calculates Word Error Rate (WER) and detailed error breakdowns between a reference (human) transcript and a hypothesis (ASR-generated) transcript.

## Setup

```bash
cd word_error_rate
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Usage

```bash
python wer_norm.py reference.txt hypothesis.txt
```

### Inputs

- **reference.txt** — Ground truth transcript (human-labeled). Plain text, UTF-8 encoded.
- **hypothesis.txt** — ASR-generated transcript to evaluate. Plain text, UTF-8 encoded.

### Output

Printed to stdout:

- **Summary metrics** — WER score, substitution/deletion/insertion counts.
- **Top 10 errors** per category — most frequent substitutions, deletions, and insertions.

Example:

```
===== SUMMARY METRICS =====
Word Error Rate: 0.0832
Substitutions: 45 | Deletions: 12 | Insertions: 8

--- Top 10 Substitutions (Ref -> Hyp) ---
3x: the -> a
2x: board -> bored
...

--- Top 10 Deletions (Missing) ---
2x: the
1x: of
...

--- Top 10 Insertions (Extra) ---
1x: uh
1x: the
...
```

## Normalization Steps

Both reference and hypothesis texts are normalized before comparison to minimize false errors from formatting differences:

1. Lowercase conversion
2. Contraction expansion (`won't` → `will not`)
3. Number-to-word conversion (`50` → `fifty`)
4. Removal of bracketed timestamps and non-speech tags
5. Removal of speaker indicators (`>>`)
6. Normalization of time formats (`a.m.` → `am`)
7. Removal of filler words (`uh`, `um`, `erm`, `ah`)
8. Percent symbol expansion (`%` → `percent`)
9. Punctuation removal
10. Title normalization (`ms` → `miss`)

These normalizations ensure the WER reflects genuine transcription errors rather than stylistic differences between human and ASR formatting conventions.
