# app/rag/sparse.py
import re
import unicodedata

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Words so common they carry no signal. BM25's IDF already down-weights them;
# removing them also stops a question like "what is the..." matching every chunk.
STOPWORDS = frozenset("""
    a an and are as at be been but by can do does for from had has have how
    i if in into is it its of on or so than that the their them then there
    these they this to was were what when where which who why will with you your
""".split())


def tokenize(text: str) -> list[str]:
    """
    Normalise, lowercase, split on anything that isn't a letter or digit, drop stopwords.

    NFKC turns PDF ligatures back into letters ("traﬃc" -> "traffic"). Without it the
    ligature is not in [a-z0-9], so "traﬃc" became the tokens "tra" and "c", which no
    query matches: 171 of the 214 corpus chunks contain one. Normalising here, not at
    extraction, fixes both the index and the queries with no re-index - BM25 is
    rebuilt from the stored text at every start.
    """
    text = unicodedata.normalize("NFKC", text)
    return [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in STOPWORDS]