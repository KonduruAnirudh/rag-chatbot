# app/rag/sparse.py
import re

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Words so common they carry no signal. BM25's IDF already down-weights them;
# removing them also stops a question like "what is the..." matching every chunk.
STOPWORDS = frozenset("""
    a an and are as at be been but by can do does for from had has have how
    i if in into is it its of on or so than that the their them then there
    these they this to was were what when where which who why will with you your
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase, split on anything that isn't a letter or digit, drop stopwords."""
    return [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in STOPWORDS]