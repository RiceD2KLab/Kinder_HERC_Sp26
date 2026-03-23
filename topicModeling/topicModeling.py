from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import re

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

stopwords = list(ENGLISH_STOP_WORDS) + [
    "yeah","okay","ok","right","um","uh","like","said",
    "madam","chair","motion","second","agenda","minutes",
    "district","board","meeting", "just","know","think","got","people","really","time", "want","see","well","way","make","take","look","good","new","could","also"
]

vectorizer_model = CountVectorizer(
    stop_words=stopwords,
    ngram_range=(1, 2),
    min_df=2
)

# from __future__ import annotations
# from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
# from __future__ import annotations
# # ran into some issues with newer versions of pandas and BERTopic, so pinning to older versions for now
# # pandas 1.5.3 and BERTopic 0.15.0 are known to work well together as of mid-2024
# # if you run into issues, try creating a new virtual environment and installing these specific versions:
# # pip install pandas==1.5.3 bertopic==0.15.0 sentence-transformers==2.2.0
# # standard imports for data structures, typing, and text processing
# from dataclasses import dataclass
# from typing import List, Optional, Tuple, Dict, Any
# import re

# # bertopic and embedding model
# from bertopic import BERTopic
# from sentence_transformers import SentenceTransformer



# stopwords = list(ENGLISH_STOP_WORDS) + [
#     "yeah","okay","ok","right","um","uh","like","said"
# ]
# vectorizer_model = CountVectorizer(stop_words=stopwords)

# utility functions for cleaning and chunking transcript text
def normalize_text(s: str) -> str:
    """
    perform light normalization on transcript text while preserving meaning
    """
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_transcript(
    transcript: str,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[str]:
    """
    split a long transcript into smaller overlapping chunks

    topic modeling works better when documents are not extremely long
    each chunk becomes a document for bertopic
    """

    transcript = normalize_text(transcript)

    if not transcript:
        return []

    # attempt paragraph splitting first
    paras = [p.strip() for p in transcript.split("\n\n") if p.strip()]

    # if transcript has no clear paragraphs, fall back to sentence splitting
    if len(paras) == 1:
        paras = re.split(r"(?:\n(?=[A-Z][A-Z .'-]{2,}:))|(?<=[.!?])\s+(?=[A-Z])", transcript)
        paras = [p.strip() for p in paras if p and p.strip()]

    chunks: List[str] = []
    buf = ""

    # helper function to store completed chunks
    def flush_buffer(b: str):
        b = b.strip()
        if b:
            chunks.append(b)

    for p in paras:

        # if paragraph is larger than chunk size, split directly
        if len(p) > max_chars:
            start = 0
            while start < len(p):
                end = min(start + max_chars, len(p))
                flush_buffer(p[start:end])
                start = max(0, end - overlap_chars)
            continue

        if not buf:
            buf = p

        elif len(buf) + 2 + len(p) <= max_chars:
            buf = buf + "\n\n" + p

        else:
            flush_buffer(buf)

            # keep overlap from previous chunk for context
            tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
            buf = (tail + "\n\n" + p).strip()

    flush_buffer(buf)

    return chunks


# dataclass used to store topic modeling summary results
@dataclass
class TopicSummary:
    docs_used: int
    topic_info: Any
    top_topics: List[Dict[str, Any]]


def run_meeting_topics(
    *,
    transcript: Optional[str] = None,
    chunks: Optional[List[str]] = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    min_topic_size: int = 8,
    top_n_words: int = 8,
    nr_repr_docs: int = 3,
) -> Tuple[BERTopic, TopicSummary, List[int], Optional[List[float]]]:

    """
    run topic modeling on a meeting transcript

    supports two workflows:
    1. transcript only -> auto chunk
    2. chunks already created -> use chunks directly
    """

    if chunks is not None:
        docs = [normalize_text(c) for c in chunks if normalize_text(c)]
    else:
        if not transcript:
            raise ValueError("provide either chunks or transcript")
        docs = chunk_transcript(transcript)

    if not docs:
        raise ValueError("no usable text found")

    # load sentence embedding model
    embedding_model = SentenceTransformer(embedding_model_name)

    # initialize bertopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        calculate_probabilities=True,
        verbose=True
    )

    # run topic modeling
    topics, probs = topic_model.fit_transform(docs)

    # retrieve topic statistics
    info = topic_model.get_topic_info()

    # remove outlier topic (-1)
    top_k = min(8, (info["Topic"] != -1).sum())
    info_non_outliers = info[info["Topic"] != -1].head(top_k)

    top_topics: List[Dict[str, Any]] = []

    # build readable topic summaries
    for _, row in info_non_outliers.iterrows():
        t = int(row["Topic"])

        words = topic_model.get_topic(t) or []
        top_words = [w for (w, _) in words[:top_n_words]]

        repr_docs = topic_model.get_representative_docs(t)
        repr_docs = (repr_docs or [])[:nr_repr_docs]

        repr_excerpts = [
            d[:250].replace("\n", " ").strip() + ("…" if len(d) > 250 else "")
            for d in repr_docs
        ]

        top_topics.append(
            {
                "topic_id": t,
                "count": int(row["Count"]),
                "label": str(row.get("Name", "")),
                "top_words": top_words,
                "representative_excerpts": repr_excerpts,
            }
        )

    summary = TopicSummary(
        docs_used=len(docs),
        topic_info=info,
        top_topics=top_topics,
    )

    return topic_model, summary, topics, probs


# run example if script executed directly
if __name__ == "__main__":

    # load transcript from file in the same folder
    with open("sampleTranscript.txt", "r", encoding="utf-8") as f:
        transcript_text = f.read()

    # run topic modeling
    model, summary, topics, probs = run_meeting_topics(transcript=transcript_text)

    # print summary results
    print(f"docs fed to bertopic: {summary.docs_used}")

    for t in summary.top_topics:
        print("\n")
        print(f"topic {t['topic_id']} (count={t['count']}): {', '.join(t['top_words'])}")

        for ex in t["representative_excerpts"]:
            print(f"  - {ex}")
    
