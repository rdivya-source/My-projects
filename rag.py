"""Retrieval layer: chunk the resume, index it, retrieve evidence, ground the LLM prompt.

Embedding = TF-IDF (scikit-learn), store = in-memory sparse matrix, similarity = cosine.
Runs fully offline, so resume text never leaves the device during retrieval.
"""
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel

import engine as E

ENGLISH_STOP = set(ENGLISH_STOP_WORDS)

CHUNK_SIZE = 200       # max characters per chunk (one bullet / line)
CHUNK_OVERLAP = 0      # lines shared between consecutive chunks
TOP_K = 3
MIN_SCORE = 0.15       # below this, report "no supporting evidence"

TOKEN = r"[A-Za-z0-9+#]+(?:\.[A-Za-z0-9]+)*"


HEADINGS = {"summary", "skills", "experience", "education", "projects", "certifications", "achievements"}
NOISE = {"candidate", "resume", "experience", "does", "did", "have", "has", "used", "use", "using", "shows", "show"}


def _stem(w):
    for suf in ("ments", "ment", "ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def analyzer(text):
    import re
    toks = re.findall(TOKEN, text.lower())
    return [_stem(t) for t in toks if t not in NOISE and t not in ENGLISH_STOP]


def chunk_resume(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Structure-aware chunking: one line/bullet per chunk, prefixed with its section heading."""
    chunks, section = [], ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().rstrip(":") in HEADINGS:
            section = line.rstrip(":").upper()
            continue
        line = line.lstrip("-\u2022* ").strip()
        chunks.append((f"[{section}] " if section else "") + line[: max(size, 200)])
    return chunks


class ResumeIndex:
    def __init__(self, resume_text):
        self.chunks = chunk_resume(resume_text)
        self.vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(self.chunks)

    def retrieve(self, query, k=TOP_K):
        scores = linear_kernel(self.vec.transform([query]), self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def answer(self, query, k=TOP_K):
        hits = self.retrieve(query, k)
        supported = bool(hits) and hits[0][1] >= MIN_SCORE
        return {"supported": supported, "hits": hits}


def skill_query(skill):
    return " ".join(E.SKILLS.get(skill, [skill]) + [skill])


def gather_evidence(index, skills):
    """Evidence chunks for each skill the job asks for."""
    return {s: index.answer(skill_query(s)) for s in skills}


def evidence_block(evidence):
    lines = []
    for skill, res in evidence.items():
        if res["supported"]:
            lines.append(f"- {skill}: \"{res['hits'][0][0]}\"")
        else:
            lines.append(f"- {skill}: NO EVIDENCE IN RESUME (do not claim it)")
    return "\n".join(lines)
