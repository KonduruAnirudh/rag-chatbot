# RAG Chatbot

A document-grounded question-answering system built with FastAPI and the OpenAI API. Upload PDF or TXT files, ask questions in natural language, and receive answers drawn strictly from your documents — with the source passages shown alongside every response.

---

## Problem statement

Large language models answer from parametric knowledge acquired during training. That creates three problems for anyone who needs answers about their own documents:

1. **The model has never seen them.** Internal policies, technical specifications, and private reports aren't in any training set.
2. **Answers can't be verified.** A fluent response gives no indication of where its claims came from, so a wrong answer is indistinguishable from a right one.
3. **Knowledge is frozen.** Anything after the training cutoff is invisible, and updating a model's knowledge means retraining it.

Retrieval-Augmented Generation addresses all three. Instead of relying on what the model memorized, the system searches a document collection for relevant passages at query time, supplies those passages as context, and instructs the model to answer only from them. The knowledge base becomes something you control — updatable by uploading a file, and auditable because every answer ships with its sources.

---

## How RAG is used in this project

The system runs two pipelines that meet at a shared vector index.

**Ingestion** happens once per document. Text is extracted, split into overlapping chunks, converted to embedding vectors, and stored.

**Querying** happens per question. The question is converted to a vector using the *same* embedding model, compared against every stored chunk by cosine similarity, and the highest-scoring passages become the context for the language model.

The critical property is that documents and questions are projected into the same vector space. That's what makes semantic similarity meaningful — the question *"How should I handle a DDoS incident?"* retrieves a passage about incident response runbooks despite sharing almost no vocabulary with it.

Three mechanisms keep answers grounded, applied in layers:

| Layer | Mechanism | Guarantee |
|---|---|---|
| Retrieval gate | Chunks below the similarity threshold are discarded. If none survive, the LLM is never called | **Structural.** Cannot fail — no context, no generation |
| Prompt instruction | The model is told to answer only from the supplied passages and to say when they don't cover the question | Strong, but probabilistic |
| Source display | Every response returns the passages used, with scores | Makes any drift visible and verifiable |

---

## Features

- Upload PDF and TXT documents through a REST API
- Incremental indexing — new documents are appended to the existing index, nothing is rebuilt
- Semantic search across the entire knowledge base, regardless of which document holds the answer
- Grounded answers with numbered inline citations
- Structured source attribution: filename, section index, similarity score, and a text snippet
- Refuses to answer when the documents don't cover the question, rather than guessing
- Query rewriting so conversational follow-ups retrieve correctly
- Duplicate detection by content hash
- Document deletion that removes vectors from retrieval immediately
- Full persistence — documents, metadata, and vectors survive a server restart
- Automatic interactive API documentation via Swagger UI
- Reads scanned PDFs and image files (PNG, JPG) using vision-model OCR
- Handles mixed documents: pages with a text layer are extracted, image-only pages are OCR'd
- Answers cite page numbers, and passages read by OCR are marked as such

---

## Architecture

```
                          ┌──────────────────────────┐
                          │   Browser (HTML/CSS/JS)  │
                          │  library + chat, served  │
                          │      by FastAPI          │
                          └────────────┬─────────────┘
                                       │  REST / JSON + multipart
                                       ▼
                          ┌──────────────────────────┐
                          │         FastAPI          │
                          │   routes/documents.py    │
                          │   routes/chat.py         │
                          └──────┬────────────┬──────┘
                                 │            │
              INGEST PATH        │            │        QUERY PATH
                                 ▼            ▼
                  ┌────────────────────┐   ┌──────────────────────┐
                  │ validate + SHA-256 │   │ generate.py          │
                  │ duplicate check    │   │ rewrite_question     │
                  └─────────┬──────────┘   │ (only with history)  │
                            ▼              └──────────┬───────────┘
              ┌──────────── per page ───────────┐     │
              │  extract.py                     │     │
              │    text layer (pypdf)           │     │
              │            │                    │     │
              │    under OCR_MIN_CHARS?         │     │
              │            │ yes                │     │
              │            ▼                    │     │
              │    render page (pypdfium2)      │     │
              │            ▼                    │     │
              │    ocr.py  ─ vision model       │     │
              │            ─ or Tesseract       │     │
              └────────────┬────────────────────┘     │
                           ▼                          │
              joined text + page character spans      │
                           │                          │
                           ▼                          ▼
              ┌────────────────────────┐   ┌──────────────────────┐
              │ chunk.py               │   │ embed.py (query)     │
              │ LangChain recursive    │   │ text-embedding-      │
              │ splitter, sentence-    │   │   3-small            │
              │ first separators       │   └──────────┬───────────┘
              │ → page_start/page_end  │              │
              │ → ocr flag             │              │
              └────────────┬───────────┘              │
                           ▼                          │
              ┌────────────────────────┐              │
              │ embed.py (documents)   │              │
              │ batched, normalised    │              │
              └────────────┬───────────┘              │
                           ▼                          │
        ┌──────────────────────────────────┐          │
        │            store.py              │          │
        │  vectors.npy      dense index    │◄─────────┤
        │  metadata.json    chunk records  │          │
        │  BM25             rebuilt in RAM │◄─────────┤
        │            registry.py           │          │
        │  documents.json   what exists    │          │
        └──────────────────────────────────┘          │
                           │                          │
                  ┌────────┴────────┐                 │
                  ▼                 ▼                 │
          dense ranking       sparse ranking          │
          (cosine, NumPy)     (BM25 keywords)         │
                  └────────┬────────┘                 │
                           ▼                          │
              Reciprocal Rank Fusion (k=60)  ◄────────┘
                           │
                           ▼
              cosine ≥ SIMILARITY_THRESHOLD, top K
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      nothing passes              chunks retrieved
              │                         │
              ▼                         ▼
      fixed refusal            generate.py builds prompt
      (no LLM call)            rules + <context> + history
              │                         │
              │                         ▼
              │              gpt-5.6-luna, temperature 0.2
              │                  reasoning disabled
              └────────────┬────────────┘
                           ▼
              answer + sources (filename, page range,
              similarity score, OCR flag) + search_query
```

### Reading the diagram

**`embed.py` appears on both paths.** Documents and questions must be turned
into vectors by the identical model for their coordinates to be comparable, so
both call the same function. A mismatch is structurally impossible rather than
a rule someone has to remember.

**OCR is a page-level branch, not a separate pipeline.** A page with a usable
text layer is extracted; a page without one is rendered to an image and
transcribed. A 68-page report containing three scans pays for three pages of
OCR. Both outputs join the same string and are chunked identically.

**Page character spans are what survive the flattening.** Chunking sees one
long string with no notion of pages, so extraction records where each page
begins and ends. Chunks are then mapped back to the pages they overlap, which
is how a source can cite "Pages 12–13".

**Retrieval ranks two ways and gates one way.** Dense search ranks by meaning,
BM25 by keywords, and Reciprocal Rank Fusion combines their positions — not
their scores, which are on different scales. The relevance gate stays on cosine
similarity, because fused ranks say which chunk is *better*, never whether any
chunk is *good enough*. That is what makes refusal possible.

**Refusal skips the model entirely.** When nothing clears the threshold, a fixed
reply is returned with no LLM call: zero tokens, zero latency, and hallucination
is impossible on that path rather than merely discouraged.

`embed.py` appears on both paths deliberately. Documents and questions must be embedded by the identical model for their vectors to be comparable, so both call the same function — making a mismatch structurally impossible rather than a rule someone has to remember.

---

## Tech stack

| Component | Technology | Rationale |
|---|---|---|
| Language | Python 3.14 | — |
| Backend | FastAPI + Uvicorn | Type hints drive validation and documentation from a single declaration; ASGI handles the I/O-bound LLM calls without blocking |
| Frontend | HTML + CSS + vanilla JS | No build step, no framework. Keeps the REST boundary explicit and visible |
| LLM | OpenAI Responses API (`gpt-5.6-luna`) | OpenAI's recommended interface for new projects. `output_text` is simpler than navigating the Chat Completions response shape |
| Embeddings | `text-embedding-3-small` | 1536 dimensions, same SDK and credentials as generation, no additional dependency |
| Vector store | NumPy array + cosine similarity | See engineering decisions below |
| PDF extraction | pypdf | Pure Python, actively maintained, single-function API |
| Chunking | Custom (~40 lines) | Recursive boundary-seeking splitter written in-project so every decision is explainable |
| Persistence | JSON + `.npy` files | No service to run; matches the scale of the data |
| Config | python-dotenv | Standard, minimal |
| PDF rendering | pypdfium2 | Page → image for OCR. Chosen over PyMuPDF, which is AGPL-licensed |
| OCR (default) | OpenAI vision via the Responses API | Layout-aware: preserves tables and describes diagram relationships |
| OCR (alternative) | Tesseract + pytesseract | Local and free, behind an `OCR_ENGINE` switch |

---

## Engineering decisions

### No orchestration framework

LangChain and LlamaIndex would collapse this project into roughly fifteen lines. That was the reason to avoid them: the pipeline stages are the substance of the work, and hiding them behind `RetrievalQA.from_chain_type()` would leave nothing to explain and no visibility into what to tune. Each stage here is a small module with an obvious responsibility.

### No vector database

Similarity search over normalized vectors is a single matrix multiplication. On this corpus it completes in well under a millisecond, and it's **exact**.

FAISS, Chroma, and Pinecone solve a problem that appears at a much larger scale. Their approximate nearest-neighbour indexes — HNSW graphs, IVF partitioning — trade recall for speed, which only pays off somewhere around 100,000 vectors. Below that, a vector database adds a dependency and an abstraction layer for no measurable benefit.

Deletion is also simpler here. NumPy fancy indexing rebuilds the matrix without the removed rows, so deleted content genuinely ceases to exist. Several production vector stores mark-and-filter instead, leaving deleted vectors physically present.

The switching point is when a linear scan starts appearing in latency measurements.

### JSON rather than SQLite for metadata

The document registry is a small list read in full on every access. There is no query to optimize, so SQLite's indexes, partial reads, and transactions address problems this project doesn't have. JSON also keeps one persistence pattern in the codebase instead of two.

The boundary: thousands of documents, where whole-file rewrites get slow, or more than one server process — JSON has no write locking, so concurrent writes would corrupt the file.

---

## Project structure

```
rag-chatbot/
│
├── app/
│   ├── main.py              FastAPI instance, CORS, router registration
│   ├── config.py            Environment loading and pipeline parameters
│   ├── schemas.py           Pydantic request/response models
│   │
│   ├── routes/
│   │   ├── documents.py     Upload, list, delete
│   │   └── chat.py          Question answering, session reset
│   │
│   ├── rag/
│   │   ├── extract.py       File bytes → plain text
│   │   ├── chunk.py         Text → overlapping chunks with metadata
│   │   ├── embed.py         Text → normalized vectors (batched)
│   │   ├── store.py         Vector matrix + aligned metadata, persisted
│   │   ├── registry.py      Document metadata, persisted
│   │   ├── retrieve.py      Question → top-k chunks above threshold
│   │   └── generate.py      Prompt construction, query rewriting, LLM call
│   │
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
│
├── data/
│   ├── uploads/             Stored source files (gitignored)
│   └── index/               vectors.npy, metadata.json, documents.json
│
├── .env                     Secrets — never committed
├── .env.example             Variable names, no values
├── .gitignore
├── requirements.txt
└── README.md
```

The split between `routes/` and `rag/` is the load-bearing structural decision. `rag/` knows nothing about HTTP — it could be driven from a command-line script or a Slack bot without modification. `routes/` handles only the HTTP contract: parse, validate, delegate, shape the response.

Within `rag/`, one file per pipeline stage means the directory listing is the architecture diagram.

---

## Installation

Requires Python 3.10 or higher.

```bash
git clone https://github.com/KonduruAnirudh/rag-chatbot.git
cd rag-chatbot
```

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

```bash
python -m pip install -r requirements.txt
```

---

## Environment variables

```bash
cp .env.example .env
```

Then edit `.env`:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API credential |
| `CHAT_MODEL` | No | `gpt-5.6-luna` | Model used for generation and query rewriting |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Model used for all embeddings |

The key is loaded once at startup and read by a single module. It never appears in source, in logs, or in an API response — upstream errors are logged server-side and returned to the client as a generic message so SDK exception text can't leak configuration details.

`.env` is listed in `.gitignore`. `.env.example` documents the variable names with no values, so the repository is complete without containing a secret.

Changing `EMBEDDING_MODEL` invalidates every stored vector. Vectors from different models are not comparable, so the index must be rebuilt — delete `data/index/` and re-upload.

---

## Running the application

```bash
uvicorn app.main:app --reload
```

| URL | What it serves |
|---|---|
| `http://127.0.0.1:8000/` | Frontend |
| `http://127.0.0.1:8000/docs` | Swagger UI — interactive API testing |
| `http://127.0.0.1:8000/api/health` | Health check |

The frontend is served as static files by FastAPI itself. One process, one port, same origin as the API — no separate server and no CORS configuration needed for local use.

---

## API reference

### `GET /api/health`
Liveness check. Returns `{"status": "ok"}`.

### `POST /api/documents`
Uploads and indexes a document. Body: `multipart/form-data` with a `file` field.

```bash
curl -X POST http://127.0.0.1:8000/api/documents -F "file=@whitepaper.pdf"
```

```json
{
  "doc_id": "0a567db7-6bee-4d5e-b265-e4ac2b9038e0",
  "filename": "whitepaper.pdf",
  "size_bytes": 1927448,
  "char_count": 118218,
  "chunk_count": 142,
  "total_chunks_indexed": 186,
  "message": "Uploaded and processed."
}
```

| Status | Condition |
|---|---|
| `400` | Unsupported file type |
| `409` | Identical file already uploaded |
| `413` | Exceeds the 10 MB limit |
| `422` | Empty file, or no extractable text (e.g. a scanned PDF) |
| `502` | Embedding API failure |

### `GET /api/documents`
Lists indexed documents with their metadata and chunk counts.

### `DELETE /api/documents/{doc_id}`
Removes a document's vectors from the index, deletes the stored file, and clears its registry entry. Returns `404` if the document doesn't exist.

Deletion order is deliberate: vectors first (the user-visible effect), registry last (the source of truth, so a failed delete can be retried).

### `POST /api/chat`
Asks a question against the knowledge base.

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the security pillar recommend for incident response?","session_id":"demo"}'
```

```json
{
  "answer": "The security pillar recommends developing a DDoS incident response strategy and runbooks before an attack. Base the playbook on NIST's steps: gather evidence, mitigate, recover, and conduct a post-incident analysis. [1][2]",
  "sources": [
    {
      "doc_id": "0a567db7-...",
      "filename": "whitepaper.pdf",
      "chunk_index": 130,
      "score": 0.542,
      "snippet": "A recommended approach is to model your response playbook based on NIST's suggested steps..."
    }
  ],
  "session_id": "demo",
  "search_query": "What does the security pillar recommend for incident response?"
}
```

`search_query` shows the question actually used for retrieval, which differs from `question` when a follow-up has been rewritten.

`400` if no documents have been uploaded. `502` if retrieval or generation fails.

### `POST /api/chat/reset`
Clears conversation history for a session. Query parameter: `session_id`.

---

## How the RAG pipeline works

### Ingestion

```
upload → validate → hash → store → extract → chunk → embed → index → register
```

**Validation** checks extension, size, and non-emptiness before anything is written.

**Content hashing** rejects byte-identical re-uploads with `409` before any work is done — no disk write, no extraction, no embedding calls.

**Storage** uses a generated UUID as the filename rather than the client-supplied one. A filename like `../../etc/passwd` would escape the upload directory if joined to a path directly; the original name is kept as display metadata only.

**Extraction** reads each page's text layer with pypdf. A page yielding fewer
than 100 characters has no usable text layer — it is a scan or a photograph —
so that page is rendered to an image and transcribed by the OCR engine. Only
pages that need it are OCR'd, capped at 20 per document. Standalone image files
go straight to OCR. If nothing can be read even with OCR, the upload is
rejected and the stored file deleted.

While the pages are joined into one string, each page's character range is
recorded. That is what allows a chunk to be mapped back to the page or pages it
covers, since chunking itself is page-blind.

**Chunking** splits at paragraphs, then sentences, then lines, targeting 1000
characters with 150 of overlap. Each chunk carries `page_start`, `page_end`,
and an `ocr` flag. Chunks may span a page boundary — deliberately, so a
sentence split across pages stays intact — which is why a chunk records a page
range rather than a single page.

**Embedding** batches up to 64 chunks per API call and normalizes each vector to unit length, so cosine similarity later reduces to a plain dot product.

**Indexing** appends the new vectors to the existing matrix. Nothing is recomputed — each chunk's vector depends only on its own text, so adding a document can never invalidate existing vectors.

### Querying

```
question → rewrite → embed → cosine search → top-k → threshold → prompt → LLM → answer + sources
```

**Rewriting** runs only when conversation history exists. It resolves references — "tell me more about that" becomes "tell me more about SpendWise, the personal finance application" — before embedding.

**Search** multiplies the entire vector matrix by the query vector in one operation, producing a similarity score per chunk. Because vectors are unit length, the dot product *is* the cosine similarity.

**Top-k and threshold** do different jobs. `TOP_K` caps how much context is sent — a budget. `SIMILARITY_THRESHOLD` decides whether any of it is good enough to use — a quality gate. Without a threshold, K would also be a *minimum*, and the system would always return K chunks however irrelevant, making refusal impossible.

**Prompt construction** wraps the passages in `<context>` tags, numbers them, and labels each with its source file. The question comes last, since instructions placed before a long context block get diluted.

**Generation** calls the Responses API. If retrieval returned nothing, this step is skipped entirely and a constant refusal is returned — zero tokens, zero latency, and hallucination is structurally impossible rather than merely discouraged.

---

## Configuration

Set in `app/config.py`:

| Parameter | Value | Reasoning |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Roughly a paragraph — large enough to be self-contained, small enough to be about one thing |
| `CHUNK_OVERLAP` | 150 | 15%. Prevents loss of facts spanning a chunk boundary |
| `TOP_K` | 5 | Raised from 4 after a real question's answer ranked exactly 5th — see findings |
| `SIMILARITY_THRESHOLD` | 0.20 | Calibrated — see findings below |
| `MAX_CONTEXT_CHARS` | 8000 | Backstop against `TOP_K` being raised without bound |
| `OCR_ENGINE` | vision | Layout-aware. `tesseract` is available for offline use |
| `OCR_MIN_CHARS` | 100 | Below this, a page has no usable text layer |
| `OCR_MAX_PAGES` | 20 | Bounds a worst-case upload to ~40s and about 2 cents |
| `OCR_CONCURRENCY` | 4 | Pages OCR'd in parallel |
| `VISION_DPI` / `VISION_MAX_PIXELS` | 150 / 1600 | See findings — image size dominated latency |
| `TESSERACT_DPI` | 300 | Classic OCR needs the detail |

These interact. Smaller chunks carry less context each, so they need a higher `TOP_K` to cover the same ground. They are not independent knobs, which is why tuning them means measuring rather than reasoning.

---

## Measurements and findings

### Vision OCR: image size dominated latency

A full page rendered at 150 DPI (1240x1753) caused the transcription call to
hang past 60 seconds. Capping the longest side at 1600 pixels before sending
brought the same page to 3.7 seconds with no loss of legibility — vision models
are billed by image dimensions, so the cap reduces cost as well as latency.

The hang was isolated by calling the OCR function directly, which returned
normally while HTTP requests stalled: that ruled out the route and pointed at
the payload.

Five image-only pages OCR at 9.8 seconds with four running in parallel, against
roughly 20 seconds sequentially.

Classic OCR (Tesseract) was implemented first and measured at 88-96% word
recovery, but it reads characters only: a flowchart becomes a list of
disconnected labels. The vision model returns the same page as
"[Diagram: A user connects to an Application Load Balancer (ALB), which
supports HTTP and HTTPS...]" — relationships, not just words. Both engines
remain available behind an OCR_ENGINE switch.

### OCR: classic versus vision, and why image size mattered

Tesseract was implemented and measured first. Rendering pages that already had
a text layer and comparing its output against pypdf's gave a ground truth:
**88–96% of words recovered** at 300 DPI, about 1 second per page.

That accuracy is adequate for prose but insufficient for this project's goal.
Classic OCR reads characters, not layout: a flowchart becomes a list of
disconnected labels, and the relationships drawn as arrows are lost entirely —
not degraded, absent. The same page read by a vision model returned:

> [Diagram: A user connects to an Application Load Balancer (ALB), which
> supports HTTP and HTTPS. The ALB connects to application targets represented
> by AWS service icons.]

Relationships, not just words. Both engines remain available behind an
`OCR_ENGINE` switch.

**Image size, not page complexity, dominated latency.** A full page rendered at
150 DPI (1240×1753) caused the transcription call to hang past 60 seconds.
Capping the longest side at 1600 pixels brought the same page to **3.7 seconds**
with no loss of legibility. Vision models are billed by image dimensions, so
the cap reduces cost as well as time. The cause was isolated by calling the OCR
function directly, which returned normally while HTTP requests stalled — ruling
out the route and pointing at the payload.

Five image-only pages OCR in **9.8 seconds** with four running in parallel,
against roughly 20 sequentially. Measured cost is about **$0.001 per page**.

### Page numbers survive chunking, but chunks rarely span pages

Extraction records each page's character range while joining pages into one
string; chunking then maps every chunk back to the pages it overlaps.
Verified across 214 chunks: page numbers rise monotonically with chunk index in
both documents, no chunk lacks a page number, and a chunk claiming page 47 was
confirmed by hand against the PDF.

Only **4 of 214 chunks span a page boundary**, far fewer than expected. The
reason is separator priority: pages are joined with `\n\n`, which is also the
splitter's first separator, so it almost always cuts at a page break before it
needs to merge across one. The cross-page merge only happens when a page's
trailing fragment is small enough to combine with the next page's opening.


### Threshold calibration

Rather than guessing, similarity scores were recorded for questions known to be answerable and questions known to be outside the corpus (186 chunks across three documents):

| Query type | Top similarity score |
|---|---|
| Answerable — incident response | 0.542 |
| Answerable — IAM permissions | 0.446 |
| In-corpus but off-topic | ~0.283 |
| Outside the corpus (three queries) | 0.121 – 0.138 |

The threshold was set at 0.20 rather than higher because the costs are asymmetric: an unnecessary LLM call is cheap, whereas wrongly refusing an answerable question looks like a broken system. The prompt-level refusal instruction acts as a second filter for weak matches that pass the gate — verified by an in-corpus off-topic question, where four chunks were retrieved and the model correctly reported that the documents didn't cover the subject.

**This number is not portable.** It is a property of the embedding model and the corpus together. A different document set or a different embedding model requires re-running the calibration.

### Extraction quality is the pipeline's ceiling

The whitepaper's table of contents extracted as dot leaders and page numbers. Those chunks are semantically empty but share vocabulary with the whole document, so they scored around 0.26–0.52 against almost any question about it — high enough to displace real content in the top-k. In one test a contents line scored 0.517 purely because it lexically matched the question.

Filtering chunks that are more than 60% dot leaders reduced the document from 149 to 142 chunks and removed them from results entirely. Page headers and footers still appear inline in body chunks; stripping those would need layout-aware extraction.

No amount of prompt engineering or retrieval tuning compensates for bad extraction, because every downstream stage reports success while operating on garbage.

### Follow-up questions fail without query rewriting

Retrieval is stateless while generation is stateful — the model sees conversation history, but the embedding step sees only the current question. So "tell me more about that" carries no topic and retrieves near-randomly, even though the model would understand it perfectly.

Demonstrated with two sessions given identical input:

| Session | History | `search_query` after rewriting | Result |
|---|---|---|---|
| A | Yes | "Tell me more about SpendWise — Personal Finance & Expense Management Application" | Correct answer, top score 0.379 |
| B | No | "Tell me more about that" | Nothing above threshold — refused |

The rewrite costs one extra LLM call and about half a second, and only runs when history exists. It falls back to the original question on failure: retrieval quality is optional, availability is not.

### Vector search under-ranks structured content

Asking which programming languages a résumé listed returned the summary paragraph first (0.607) and the actual skills list — containing the literal answer — third (0.346). Terse structured content embeds weakly compared to flowing prose, even when it is the more exact answer.

`TOP_K` is what saves this: retrieval doesn't need to rank the answer first, it needs to get the answer into the window. Closing the gap properly would require hybrid search, combining keyword matching with vector similarity.

### Prompt injection

### Vocabulary mismatch: a retrieval miss traced to its cause

Asking *"What HTTP status code does a recipient return when it accepts a Security Event Token?"* was refused, although the answer — `202 (Accepted)` — is stated in RFC 8935.

Diagnosis, step by step:

| Check | Finding |
|---|---|
| Does the answer exist in the index? | Yes — in two chunks, because the sentence sits on a chunk boundary and the overlap preserved it in both |
| Where did it rank? | 7th (score 0.523), outside `TOP_K = 5` |
| What outranked it? | Four chunks about the *"Security Event Token Error Codes"* registry (top score 0.608) |
| Why? | The question spelled out "Security Event Token". The registry chunks repeat that exact phrase; the answer chunk uses the RFC's acronym, "SET" |

Rephrasing the question in the document's own vocabulary — *"What status code does a SET Recipient respond with when a SET is valid?"* — raised every score (top result 0.608 → 0.735) and brought the answer chunk to rank 5 (0.671). The system answered `202 (Accepted)` and cited that passage directly.

Three conclusions:

- **The failure was in retrieval, not generation.** Given only error-code passages, the model reported that the documents did not cover the question rather than supplying the answer from its own training data. That is the intended behaviour.
- **`TOP_K = 5` is justified by measurement.** In the successful case the answer chunk ranked exactly 5th. With the earlier setting of 4, the correctly phrased question would also have been refused.
- **The same fact can be retrievable or not depending on phrasing.** Pure vector search is sensitive to whether the question uses the document's terminology. Hybrid search, which adds keyword matching, or query expansion, which rewrites terms like "Security Event Token" to "SET" before searching, would address this.

The same chunk exposed an extraction artefact. The RFC renders its normative keywords (MUST, SHALL) and cross-references in a distinct style, and pypdf emits those text runs after the paragraph rather than in place. So *"the body of the response MUST be empty"* extracts as *"the body of the response be empty … MUST"* — the obligation detached from the sentence it governs. In a specification, MUST versus MAY is the meaning of the sentence, and every downstream stage reported success regardless. A layout-aware extractor would likely fix this.

## Example questions

Against a technical whitepaper:

- "What does the security pillar recommend for incident response?"
- "How should IAM permissions be managed?"
- "What is the difference between infrastructure layer and application layer attacks?"

Follow-up, in the same session:

- "Tell me more about that"

Testing refusal — a question with no answer in the corpus:

- "What is the recipe for chicken biryani?"

Expected: *"I couldn't find anything in the uploaded documents that answers that."* — with an empty `sources` array and no LLM call made.

---

## Testing

The interactive Swagger UI at `/docs` covers every endpoint without additional tooling.

**Document handling**

| Test | Expected |
|---|---|
| Valid PDF | `200`, chunk count returned |
| Valid TXT | `200` |
| `.exe` or other unsupported type | `400` |
| Zero-byte file | `422` |
| Over 10 MB | `413` |
| Scanned PDF (no text layer) | `422`, and the stored file is removed |
| Same file uploaded twice | `409` naming the existing document |

**Retrieval and generation**

| Test | Expected |
|---|---|
| Question answerable from one document | Grounded answer with citations |
| Question spanning two documents | Sources from both filenames |
| Question with no answer in the corpus | Refusal, empty sources, no LLM call |
| Follow-up using a pronoun | `search_query` shows the resolved question |
| Same follow-up in a fresh session | Poor retrieval or refusal |
| Question before any upload | `400` |

**Persistence**

Upload a document, stop the server, restart it, then call `GET /api/documents`. The document should still be listed and still answerable. Delete it, restart again, and it should remain absent.

**Error paths**

Set `CHAT_MODEL` to an invalid value and restart: chat requests should return `502` with a generic message, while the real error appears only in the server log.

---

## Known limitations

1. **OCR text is a transcription, not the source.** Text read from an image is
   a model's interpretation of pixels. It can paraphrase, and on unclear input
   it may produce a plausible word rather than an error. Affected passages are
   marked `OCR` in the sources panel so the weaker guarantee is visible.
2. **Image-based prompt injection.** Instructions hidden inside an image now
   enter the context, and unlike text-layer injection they cannot be found by
   searching the file. The `<context>` delimiters and data-not-instructions
   prompt rule mitigate but do not eliminate this.
3. **Near-duplicate content is not detected.** Duplicate detection compares
   file bytes, so the same page uploaded as a PDF and again as an image is
   indexed twice and competes with itself for retrieval slots.
4. **OCR is capped at 20 pages per document.** Beyond that, remaining pages are
   skipped and the count reported in `ocr_skipped`.
5. **Summarisation is poor.** "What is this document about?" retrieves nothing,
   because the question contains no topic to match against.
6. **BM25 does not stem.** "signature" and "sign" are different tokens, so
   keyword search can miss a morphological variant.
7. **Conversation history is in memory.** Documents and vectors persist across
   restarts; chat sessions do not.
8. **Single-process only.** JSON persistence has no write locking.
9. **No startup reconciliation.** A crash between indexing and registration
   would leave orphaned vectors.
10. **No authentication or rate limiting.**
11. **Evaluation sets are small** — 9 retrieval questions, 3 OCR pages sampled.
    Results are indicative rather than conclusive.

## Future improvements

**Retrieval quality**
- Hybrid search combining BM25 keyword matching with vector similarity, to close the gap on structured content that embeds weakly
- Cross-encoder re-ranking of the top-k candidates, which scores query and passage jointly rather than comparing independent embeddings
- Layout-aware extraction to strip headers, footers, and page furniture before chunking

**Scale**
- A dedicated vector database once linear scanning appears in latency measurements — roughly 100,000 vectors
- PostgreSQL for document metadata, replacing JSON, once concurrent writes are needed
- Object storage for uploaded files instead of the local filesystem
- Background ingestion so large uploads don't block the request

**Product**
- Streaming responses, so answers appear progressively rather than after a multi-second wait
- Per-document filtering, letting a user scope a question to a subset of the corpus
- Persistent conversation history in a database
- Authentication and per-user document isolation

**Evaluation**
- A labelled set of question–expected-source pairs, measuring retrieval precision and recall to replace manual threshold calibration with a reproducible metric

- **Structured diagram extraction.** Vision OCR describes diagram relationships
  in prose. Returning them as structured entity–relation pairs would let them
  populate a knowledge graph directly rather than passing through text.
- **Hybrid retrieval over a knowledge graph**, so questions about connections
  and multi-hop relationships can traverse entities rather than matching
  passages.
  
---

## Security notes

- The API key lives in `.env`, which is gitignored, and is read by a single module. It never reaches source control, logs, or an API response.
- Upstream SDK errors are logged server-side and returned to the client as generic messages, so exception text can't leak request IDs or configuration.
- Uploads are validated for extension, size, and non-emptiness before any processing.
- Files are stored under a generated UUID, never the client-supplied filename, which prevents path traversal.
- Failed ingestion rolls back — the stored file is deleted rather than left orphaned.
- Retrieved context is delimited with XML tags and explicitly marked as untrusted data in the system prompt.
- CORS is configured permissively for local development and would be restricted to the frontend origin in a deployment.
