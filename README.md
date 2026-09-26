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
| Retrieval gate | Chunks below the similarity threshold are discarded. If none survive, the LLM is never called | **Structural.** Cannot fail — no context, no generation. But an off-topic question that names something in the documents usually passes the gate (Known limitation 17), so for those the prompt instruction is what refuses |
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
- Experimental graph mode, off by default: search passages plus passages found by following the entities a question names, each shown with the path and sentence that reached it. A demonstration, not an improvement (see "Graph answers (Step 8)")

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

### The graph holds multi-hop paths, not lookup facts

The knowledge graph exists for questions that need several hops across documents or sections — how a Shield Advanced finding reaches Security Hub CSPM, and what happens next. A fact stated plainly in one sentence ("Enterprise Support includes a designated Technical Account Manager") is already found by vector and keyword search, so the graph does not need to hold it.

This is why plan-to-feature relationships deliberately have no verb. `supports` was removed after it caused two-thirds of the labelled errors, and `contains` was narrowed so "for X" and "access to X" no longer count as inclusion; those relationships now fall to `related_to` or are skipped. No verb was added for them, because every verb added during the extraction pilot lowered agreement between runs, and the information is easy to retrieve without the graph.

### Graph evidence is quoted, not asserted

*Built in Step 8 as an experimental path (see "Graph answers (Step 8)"), with one change decided then: the relationship label is not given to the model at all. The reason follows the original design below.*

Decided before answer generation uses the graph. Measured edge precision on untuned text is 48%, so a two-hop path is fully right only about a quarter of the time. Every edge, though, carries the sentence it came from, verbatim — and the sentence is right even when the relationship label is not.

Showing the quote alongside the verb is not enough at this precision: a wrong verb still frames how the model reads a correct sentence. So graph evidence enters the prompt with the **quoted sentence as the primary evidence** and its source cited, and the relationship label as a secondary hint that the prompt explicitly calls unreliable — the model is told the sentence is what counts. Never a bare triple stated as fact.

If this works, the graph's value does not depend on its labels being right: it becomes a way of surfacing sentences that vector search would not have retrieved, reached by following entities from one passage to another. That value survives a wrong verb.

Evidence from a diagram is kept distinct. When vision OCR reads a diagram, it writes its own description ("[Diagram: A user connects to a VPC endpoint …]"). An edge quoting that description rests on machine-written text, not on a transcribed sentence, and is shown as weaker evidence than either a text-layer sentence or an OCR'd one.

**Superseded in Step 8 (owner, 2026-09-26): the label is withheld, not flagged.** Relationship labels are 48% correct, and prompt-only warnings failed every time they were tried in this project — the extraction prompt's rule 3 and its no-substitute rule both. So the model gets the two entity names and the verbatim sentence, the reliable part, and never the label, the unreliable one. This is enforced in code and asserted in `eval/step8_check.py`. The label stays in the API response and the UI, so a person can still see and check it.

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
│   │   ├── chat.py          Question answering, session reset
│   │   └── graph_chat.py    Graph mode (Step 8, experimental; off by default)
│   │
│   ├── rag/
│   │   ├── extract.py       File bytes → plain text
│   │   ├── chunk.py         Text → overlapping chunks with metadata
│   │   ├── embed.py         Text → normalized vectors (batched)
│   │   ├── store.py         Vector matrix + aligned metadata, persisted
│   │   ├── registry.py      Document metadata, persisted
│   │   ├── retrieve.py      Question → top-k chunks above threshold
│   │   ├── generate.py      Prompt construction, query rewriting, LLM call
│   │   ├── graph_extract.py Chunk → entities and quoted relations (Step 5, frozen)
│   │   ├── graph.py         Extraction records → the graph; resolution, admission, storage (Step 6)
│   │   ├── graph_retrieve.py  Question → graph passages: Step 7's G1 method (Step 8)
│   │   └── graph_answer.py  The graph-mode prompt, without relationship labels (Step 8)
│   │
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
│
├── data/
│   ├── uploads/             Stored source files (gitignored)
│   ├── index/               vectors.npy, metadata.json, documents.json
│   └── graph/               extractions.json: the graph's source of truth
│
├── eval/                    Evaluation scripts and saved runs (see Testing)
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
| `GRAPH_RAG_ENABLED` | No | `false` | Turns on the experimental graph mode (`POST /api/chat/graph` and the UI switch). `/api/chat` never reads it |

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
  "graph": "extracting in the background",
  "message": "Uploaded and processed."
}
```

The document is searchable as soon as this returns. Knowledge-graph extraction runs afterwards as a background task (three model calls per chunk), because it is slow and the graph is an addition: a failure there never fails an upload. If the document is deleted before extraction finishes, nothing is written to the graph.

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
Removes a document's vectors from the index, its records from the knowledge graph, deletes the stored file, and clears its registry entry. Returns `404` if the document doesn't exist. The response reports `chunks_removed` and `graph_chunks_removed`.

Deletion order is deliberate: vectors first (the user-visible effect), then the graph — which is rebuilt from its remaining records and reads chunk text from the vector store, so the vectors must already be gone — and the registry last (the source of truth, so a failed delete can be retried). The graph after a delete is exactly the graph that would exist had the document never been uploaded; `python -m eval.graph_lifecycle_check` verifies this on a copy of `data/`.

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

### `POST /api/chat/graph` (experimental)
The same request as `/api/chat`. Returns `403` unless `GRAPH_RAG_ENABLED=true`. Answers from the search passages plus up to two graph passages. The response adds `retrieval_mode: "vector+graph"`. Each source gains `retrieval` (`"vector"` or `"graph"`), and graph sources gain `graph`: the question's entities that reached the passage, and each link with `reached_from`, `to`, the extracted `relation` (for people; never sent to the model), the verbatim `quote` and the names as written. Sources are in prompt order, so `[n]` in the answer is `sources[n-1]`. Graph mode keeps its own conversation history.

### `POST /api/chat/graph/reset` (experimental)
Clears graph mode's history for a session. Query parameter: `session_id`.

### `GET /api/chat/graph/status`
`{"enabled": true | false, "detail": …}`: whether this server has graph mode on, so the UI can say so instead of failing.

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

**Graph extraction** then runs in the background: entities and quoted relationships are extracted from each chunk and stored in `data/graph/`. Deleting a document removes them. Answers do not use the graph (see Known limitation 18).

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
| `MAX_CONTEXT_CHARS` | 8000 | Backstop against `TOP_K` being raised without bound. Enforced in both modes by `generate.fit_context`: whole passages are dropped from the end, never cut, and the first is always kept. The prompt and the sources drop the same passages. At 5 chunks it never triggers (about 5,300 characters) |
| `GRAPH_EVIDENCE_MAX` | 2 | Graph passages added after the vector top 5 in graph mode: Step 7's G1 cap, unchanged |
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

The same chunk exposed an extraction artefact (see also Known limitation 13). The RFC renders its normative keywords (MUST, SHALL) and cross-references in a distinct style, and pypdf emits those text runs after the paragraph rather than in place. So *"the body of the response MUST be empty"* extracts as *"the body of the response be empty … MUST"* — the obligation detached from the sentence it governs. In a specification, MUST versus MAY is the meaning of the sentence, and every downstream stage reported success regardless. A layout-aware extractor would likely fix this.

### Graph extraction: precision only means something under one labelling standard

Edges from the extraction pilot were labelled by hand after each prompt change. Between prompt v4 and v5 the labelling standard tightened: "Enterprise Support **for** a designated Technical Account Manager" was accepted as `contains` in v4 and rejected in v5, because the sentence does not state inclusion. Seven edges were identical in both runs — same verb, same quote — and changed label.

| Prompt | Precision as first reported | Precision under the v5 standard |
|---|---|---|
| v4 | 21/35 = 60% | 14/35 = 40% |
| v5 | — | 21/30 = 70% |

The honest trend is 40% → 70%, not 60% → 70%. The standard is now written at the top of every labelling file and held fixed, so later iterations are comparable.

### Graph extraction: about half the edges are correct on untuned text

**Headline: 48% of the graph's edges are correct on untuned text** (28/58, 95% range 36–61%). That is the expected precision of the full graph, because 207 of the 214 indexed chunks were never used to tune the extraction. It pools every hand-labelled edge from untuned text: the v6 pilot's holdout and a random sample of the finished graph.

| Labelled by hand | Precision | 95% range |
|---|---|---|
| **Pooled, untuned text (the expected figure)** | **28/58 = 48%** | **36–61%** |
| Random sample of the finished graph — 40 edges, seeded, after resolution and admission | 17/40 = 42.5% | 29–58% |
| v6 pilot holdout — 3 chunks drawn after the prompt was fixed | 11/19 = 58% | 36–77% |
| v6 pilot known chunks — tuned on, so optimistic | 20/23 = 87% | 68–95% |

The holdout's 58% was the first estimate. The 40-edge sample is twice its size and measures the graph actually traversed; the two ranges overlap, so the pooled figure is the best estimate. Admission did not raise precision — it makes the graph smaller, not more correct. Votes did not predict correctness either: edges found by all three extraction runs scored 38%, those found by two of three 47%.

**The edges traversal depends on are the least reliable.** An edge can sit in the middle of a multi-hop path only if both of its entities have at least two neighbours. Those edges scored 32% (6/19, 95% range 15–54%) — the lowest of any group — against 52% for edges with an entity at a dead end (11/21). The sample is small, so this is a direction rather than a conclusion. Combined with the pooled figure, a two-hop path is fully right about 23% of the time (0.48²), and less if its middle edges are the weaker ones.

The known-chunk figure alone would be misleading: precision on the chunks the prompt was tuned against rose from 70% to 87%, while on unseen text it was below v5's 70%. The holdout's errors took forms the known chunks barely contained — `conforms_to` for "delivered over" and "validated using", `receives`/`uses` inferred from "processes" and "provides", and `related_to` for entities that are merely listed together.

What the pilot established:

1. **Perfect agreement, 42% wrong.** On the holdout, two independent extractions agreed exactly (1.00), and 8 of 19 edges were still wrong. Voting measures consistency, not correctness: a mistake the prompt causes is made by every run.
2. **Code checks worked; prompt rules did not.** The `verb_not_stated` check caught 73% of errors on the known chunks and 0% on the holdout. The prompt rules "never infer a purpose" and "never substitute a nearby entity" failed repeatedly. No prompt-only rule held in this project; every lasting fix was enforced in code.
3. **Word lists do not generalise.** Tuned for `contains`, `creates` and `mitigates`, they score 7/7 — and the errors moved to verbs they don't cover. Extending them to `receives` and `uses` would have rejected correct edges, including the planned demo path's middle hop. The one holdout rejection ("Global Accelerator *has an inbuilt* SYN proxy") was a correct edge; its words were deliberately not added, which would have fitted the list to the holdout.
4. **The holdout is spent.** Any future change to the extraction prompt or checks needs a fresh holdout to be tested honestly.

The demo path planned at the start — Shield Advanced → finding → Security Hub CSPM — is not fully supported by explicit text. Its first hop comes from a passive sentence ("anomalous traffic is surfaced as a Shield Advanced finding") that names no agent, so under the strict standard it is not an edge.

### Graph extraction over the full corpus

All 214 chunks were extracted with the frozen v6 prompt (2-of-3 voting): no chunk failed, 642 model calls, 9.3 minutes, about $0.31 at list prices. 96% of input tokens were served from the prompt cache, which is why the pilot's projection ($0.45) was high.

| | Count |
|---|---|
| Entity mentions / distinct names | 1,880 / 1,149 |
| Edges extracted | 545 |
| Edges after dropping generic endpoints (rule R2) | 514 (−31, 5.7%) |
| Distinct entities admitted (rules R1–R3), exact names | 416 |
| … with "AWS"/"Amazon" prefixes merged | 396 |
| Edges between admitted entities | 385–389 (71% of extracted) |
| Chunks with no edge | 42 |

The admitted counts are upper bounds: entity resolution (aliases, embedding similarity) is not built yet and will merge further.

### Graph retrieval experiment: fixed before any data

Written down before any retrieval arm exists, so the thresholds cannot drift towards the results.

**Outcome (recorded after the single held-out run, 2026-09-25): the stop rule fired.** On the held-out half no graph configuration fully recalled more multi-hop questions than today's retrieval: every arm scored 4 of 5. The graph found 1–2 more gold chunks, but the primary metric gained 0 points against the +15 required. As pre-registered, the experiment stops and reports that the graph does not help on this corpus. Details are in "Phase 3 result" below. Step 7 was measured at commit `44f32db`, before the close-out ligature fix; see "Close-out fixes" for why re-running it today gives slightly different search results.

**The question.** Does graph retrieval help on this corpus, and if so, when should it run?

**What the graph is for on this corpus: promoting evidence, not finding it.** This was measured on the development half before any graph arm existed. Every gold chunk that today's retrieval (A0) leaves out of its top 5 still passes the 0.20 cosine gate: it is ranked lower, not rejected (6 multi-hop chunks at ranks 7, 7, 8, 13, 14 and 19, and 1 plain chunk at rank 8). The gate passes 48–209 of the 214 chunks for these questions, so it works as a refusal switch, not a relevance filter. The graph's possible value here is therefore moving evidence that retrieval ranked too low into the context, not reaching evidence retrieval cannot see. That has an obvious cheaper alternative: raise `TOP_K`. The A0@7 control partly tests it. On the development half, 7 chunks would have completed none of the 3 multi-hop questions A0 misses, because each has a gold chunk ranked 13th or lower. If A0@7 alone closes most of the gap on the held-out half, the honest finding is that the graph's value is small next to a one-line configuration change.

**Step zero — linking.** Before any arm is built, measure whether the entities a question is about can be found in the graph at all (`python -m eval.linking_recall`). If they can't, no traversal policy matters.

**Arms**

| Arm | Policy |
|---|---|
| A0 | Today: dense + BM25 + Reciprocal Rank Fusion |
| A0@7 | A0 with the top 7 instead of 5: the control for showing the model two more chunks *(added 2026-09-25)* |
| A2 | Always run graph retrieval too, and fuse its chunks into the ranking; the cosine gate still decides refusal |
| A3 | A deterministic trigger decides when graph retrieval runs, then fuses as A2 |
| Ceiling | A2's fusion applied only to questions a person labelled as needing the graph — the most any trigger could gain |

A classifier arm (graph-only when a model says so) was dropped: it is the most complex, needs its own labelled set, loses the vector evidence whenever it misroutes, and model decisions varied between runs during extraction (agreement 0.44–0.62). It returns only if the ceiling shows a large gain that A3's rules cannot capture.

**Questions.** 40, written and labelled by the project owner — 12 plain, 10 relationship, 10 multi-hop, 8 off-topic (4 naming nothing in the corpus, 4 naming a corpus entity) — split 20 development / 20 held out, the held-out half run once. 60 candidates were drafted from the documents, not from the graph, to keep the set from favouring what the graph happens to contain.

**Pass/fail, per arm**

| Criterion | Threshold |
|---|---|
| No regression on the existing 9 retrieval questions | MRR within 0.03 of 0.704, Recall@5 stays 100% |
| Refusal | All 8 off-topic questions refused; any failure fails the arm outright. *Amended 2026-09-25: absolute for the 4 naming nothing only; see "How refusal is evaluated"* |
| All-hops recall on multi-hop questions, held-out half | At least +15 points over A0, *and over A0@7 (tightened 2026-09-25, before any graph arm or held-out question ran)* |
| Added latency | Under 500 ms when the graph is used |
| Noise | Non-gold share of the context no worse than A0 + 10 points |

**Stop rule.** If the ceiling arm does not clear +15 points of all-hops recall, the experiment stops and reports that the graph does not help on this corpus.

**Found while preparing.** Off-topic questions that name a corpus entity are not refused by the cosine gate today: 7 of 7 candidates scored 0.33–0.70 against the 0.20 threshold ("What is the monthly price of AWS Shield Advanced?" scored 0.695) and reach the model, which then has to decline on its own. Off-topic questions naming nothing scored at most 0.17 and are refused before any model call. So what "refused" means in the refusal criterion — no retrieval, or an answer that declines — has to be fixed before the experiment runs.

**How refusal is evaluated (owner decision, 2026-09-25).** Production refusal is unchanged: retrieval keeps only chunks at or above the 0.20 cosine gate, and when none pass, the fixed reply is returned without calling the model. A graph arm keeps that rule across both of its sides: it refuses without calling the model only when neither side finds sufficiently relevant evidence, meaning nothing passes the cosine gate and the graph side contributes nothing. What counts as sufficiently relevant graph evidence is fixed with each arm, before its development run. Refusal is scored from the retrieval output alone (no evidence means refused), so no answer text has to be judged. Measured on the 8 kept off-topic questions with A0 today:

| Questions | Top cosine | A0 today | Graph linking |
|---|---|---|---|
| Naming nothing: Q48, Q50, Q51, Q52 | 0.10–0.17 | Refused without the model, all 4 | Links nothing in any of them |
| Naming a corpus entity: Q54, Q55, Q56, Q59 | 0.33–0.70 | 5 chunks pass the gate for each; all 4 reach the model | Each links to its entity |

Every arm must refuse the 4 that name nothing; one failure fails the arm. The 4 that name a corpus entity are not refused by A0 under this definition, so "all 8 refused" cannot be met by any arm, A0 included, while production refusal stays as it is. Decided by the owner before any arm runs: the absolute criterion covers only the 4 that name nothing. For the 4 that name a corpus entity, A0's behaviour is the baseline, production refusal stays unchanged, and each arm records separately whether graph retrieval contributed evidence to them, with no pass/fail.

**Change to the plan: the assistant labelled the questions.** At the owner's request (2026-09-25) the labels in `eval/graph_questions.yaml` were written by the assistant, not the owner; owner review is pending. The safeguards: every answer fact is a verbatim phrase of its gold chunks, checked in code; the 40 kept questions and the 20/20 split were drawn with a fixed seed, not chosen. Labelling against the text moved 5 of the 16 drafted multi-hop questions to relationship (one passage answers each) and marked 2 more as not needing the graph (the evidence is adjacent chunks of one section), which leaves 9 drafted questions that need passages from different sections, including one that is borderline and one (the demo question) that is only partly answerable. The 10 kept multi-hop questions are those 9 plus one of the 2 same-section questions, because a plain random draw had dropped a clean cross-section question. On owner review the multi-hop questions were re-split so both halves hold several clean graph questions: held out Q39, Q41, Q44, Q45 (clean) and Q42 (borderline); development Q38, Q46, Q47 (clean), Q32 (partly answerable) and Q33 (does not need the graph). The held-out multi-hop half is therefore all graph-needed, but it is still 5 questions, so a single question moves all-hops recall by 20 points. `python -m eval.check_graph_questions` re-checks every field, gold chunk and answer fact after any edit to the labels.

**Step zero result: linking recall 41/56 = 73%.** Every question that needs the graph linked at least one of its entities, so traversal has a starting point for all 9. Of the 15 intended entities that did not link, 4 are not in the graph under any name ("incident response" twice, "cost protection", "DNS"), 2 were rejected by admission while a shorter admitted node carries the concept ("challenge", "application layer"), and 9 are misses. Most misses are the question's wording differing from the documents ("recipient" for SET Recipient, "DNS reflection" for UDP reflection attacks). Two are a longer, more specific node taking the words ("waf rules", "set delivery"), and one is a singular question word against a plural node ("CloudWatch metric"). The first run counted a tenth miss that was a labelling error: Q34 listed the Shield Response Team, which is the answer rather than something the question names. It was removed on owner review (41/57 = 72% before). All 4 off-topic questions that name a corpus entity link to it, and the 4 naming nothing link to nothing.

**Linking by question type** (the 40 kept questions)

| Type | Intended entities linked | Recall | Every entity linked | At least one linked |
|---|---|---|---|---|
| Plain | 6 of 15 | 40% | 3 of 9 | 5 of 9 |
| Relationship | 15 of 18 | 83% | 7 of 10 | 9 of 10 |
| Multi-hop | 16 of 19 | 84% | 7 of 10 | 10 of 10 |
| Needing the graph (9) | 15 of 18 | 83% | 6 of 9 | 9 of 9 |
| Off-topic naming an entity | 4 of 4 | 100% | 4 of 4 | 4 of 4 |

Linking is weakest on plain questions, which do not need the graph. Every question that needs the graph links at least one of its entities. The 3 that miss one are "incident response" and "cost protection", which are not in the graph, and "AWS WAF", which the more specific "waf rules" took. Phase 1 reports linking next to each question's retrieval result, so a multi-hop failure can be put down to linking or to traversal.

**Step 7 design, fixed before Phase 1 (2026-09-25).** Every graph arm runs the same pipeline; only the traversal differs.

1. **Link:** the step-zero linker, unchanged. The entities it finds are the starting points.
2. **Reach:** the chunks whose quotes support an edge reached from them (four configurations, below).
3. **Relevance:** a graph chunk counts only if it passes the production cosine gate for the question and is not already in A0's top 5.
4. **Rank:** chunks reached from the most starting entities come first, then the configuration's score decides, then cosine.
5. **Add:** the top 2 go after A0's unchanged top 5.

*Why add rather than fuse.* Adding makes the graph's contribution exactly two chunks, so it can be attributed and removed. Fusing the graph into RRF as a third ranked list would give a source that is 48% correct an equal vote over the top 5, and make its effect inseparable from the baseline. Adding also means neither refusal nor the existing 9 questions' top 5 can change: if nothing passes the gate, A0 refuses, and no graph chunk can pass it either. The cost is a context of 7 chunks instead of 5, which is why A0@7 is a control.

*Why no relation verbs.* Verbs are 48% correct. "This quote names both entities" is true by construction, because the quote check rejects any edge whose quote does not name both ends. Retrieval uses the reliable part and ignores the unreliable part.

| Configuration | Reach | Chunks reached per entity (median / 90th percentile) |
|---|---|---|
| G1 (the default) | 1 hop | 1 / 3 |
| G2-U | 2 hops | 5 / 27 |
| G2-H | 2 hops, never through a hub (10 or more neighbours) | 2 / 10 |
| G2-W | 2 hops, each step weighted by 1 / neighbour count | 5 / 27, weighted |

Each configuration runs as A2 (on every question) and as the ceiling (only on questions labelled as needing the graph), next to A0 and A0@7.

**Phase 1** covers only the development half and measures retrieval only. It reports, per arm:
- multi-hop questions with every gold chunk in the context, over all 5 and over the 4 that need the graph
- gold recall
- graph-chunk precision: the share of added chunks that are gold
- noise: the non-gold share of the context, pooled over the questions that have gold chunks
- whether the existing questions' top 5 is identical to A0's
- refusal of the questions naming nothing
- chunks added to the entity-naming off-topic questions (recorded, with no pass or fail)
- latency of the graph side (link, reach, rank): median and 95th percentile
- candidate chunks per question

"The context" is what the model would see: 5 chunks for A0, 7 for A0@7, and up to 7 for a graph arm.

**Selection rule, fixed now.** A configuration is out if, under A2, it changes the refusal of a question naming nothing, changes the existing questions' top 5, or has a 95th-percentile latency of 500 ms or more. Among the rest, the configuration that fully recalls the most development multi-hop questions wins. Ties go to the higher graph-chunk precision, then to the simpler configuration, in the order G1, G2-U, G2-H, G2-W. G1 is the default that a 2-hop configuration has to beat. The script applies the rule; nobody picks.

**A3.** If the chosen configuration passes the noise criterion under A2 on the development half, A3 is skipped and the result says so: a trigger that only saves cost is not worth its brittleness. Otherwise its trigger is designed from development results only, and fixed before the held-out run.

**Phase 3, the held-out half, run once.** It runs A0, A0@7, and the chosen configuration under A2 (and A3 if built) and as the ceiling, against the pass/fail table. The multi-hop gain must be at least 15 points over both A0 and A0@7.

**Phase 1 result, development half (2026-09-25; `python -m eval.graph_retrieval_eval`).** No configuration fully recalled a multi-hop question that A0 misses. A0, A0@7 and all four configurations each score 2 of 5, and 1 of the 4 that need the graph. Gold chunks found: A0 17 of 24; A0@7, G1 and G2-H 19; G2-U and G2-W 18.

- **What G1 added.** 2 of the 24 chunks it added are gold (8%), both on questions that need the graph. AWS:154 for Q32 was ranked 7th by A0, so A0@7 reaches it too. RFC:20 for Q38 was ranked 19th, which A0@7 does not reach: that is the one promotion only the graph made. A0@7 in turn reached RFC:26 (ranked 7th), which the graph missed.
- **Why the rest was missed.** A0 left 6 gold chunks out of its top 5 on the 3 incomplete graph questions. G1 added 2 of them. AWS:153 (Q32) is reachable only at 2 hops and ranked 8th to 13th among the candidates, below the cap. RFC:26 (Q38) has edges, but none involving the linked entity. AWS:115 and AWS:37 (Q46) have no edge at all. None was lost to linking, so on this half the limit is how much of the text extraction turned into edges, not linking or traversal.
- **Selection.** The rule chose G1: all configurations tied on multi-hop, and G1 had the highest graph-chunk precision.
- **A3 is skipped, as pre-registered.** G1's noise under A2 is 82%, against A0's 79%, which is within 10 points.
- **Hard criteria.** Every configuration refused both questions naming nothing and left the existing questions' top 5 unchanged. The graph side takes 2.6 ms at the 95th percentile.
- **Entity-naming off-topic questions.** Every configuration added 2 chunks to each of the 2 in this half (recorded, with no pass or fail).

**Phase 3 result, held-out half, run once (2026-09-25; `eval/graph_runs/heldout-20260925-230948.json`).** This is the final Step 7 retrieval evaluation, recorded as measured.

| Arm | Multi-hop fully recalled | Of those needing the graph | Gold chunks found | Noise | Added chunks that are gold | Graph side, 95th percentile |
|---|---|---|---|---|---|---|
| A0 | 4 of 5 | 4 of 5 | 19 of 23 | 76.2% | | |
| A0@7 | 4 of 5 | 4 of 5 | 19 of 23 | 83.0% | | |
| G1 | 4 of 5 | 4 of 5 | 20 of 23 | 81.7% | 1 of 29 | 2.7 ms |
| G2-U | 4 of 5 | 4 of 5 | 21 of 23 | 80.9% | 2 of 30 | 2.5 ms |
| G2-H | 4 of 5 | 4 of 5 | 21 of 23 | 80.7% | 2 of 29 | 2.7 ms |
| G2-W | 4 of 5 | 4 of 5 | 20 of 23 | 81.8% | 1 of 30 | 2.5 ms |
| G1, ceiling | 4 of 5 | 4 of 5 | 19 of 23 | 78.9% | 0 of 10 | 2.7 ms |
| G2-U, G2-H, G2-W, ceiling | 4 of 5 each | 4 of 5 | 20 of 23 | 77.8% | 1 of 10 | 2.5–2.7 ms |

- **The graph improved gold-chunk retrieval but not the primary metric.** Every graph configuration found more gold chunks than A0 and A0@7 (19 of 23): G1 and G2-W found 20, and G2-U and G2-H found 21. None fully recalled more multi-hop questions: every arm scores 4 of 5, including A0@7 and every ceiling.
  - The one held-out multi-hop question A0 misses is Q39. It needs AWS:19 (A0 rank 16) and AWS:91 (rank 34). The 2-hop configurations added AWS:19; no configuration reached AWS:91.
  - G1's one extra gold chunk is RFC:14 for Q01, a plain question. On the questions that need the graph, G1 added no gold chunk: its ceiling finds 19 of 23, the same as A0.
  - A0 already fully recalled 4 of the 5, so the largest gain possible on this half was one question (20 points).
- **Which configuration was tested.** The pre-registered rule selected G1 on the development half, and Phase 3 tests that choice. The script also applies the rule to whichever half it runs on, so this run printed "chosen: G2-H": tied at 4 of 5, with higher graph-chunk precision on the held-out half. That would be a choice made on held-out data, so it is reported here but not used. Every configuration gained 0 points, so the outcome does not depend on it. The script has since been fixed: the selection function refuses any half but the development half, and a held-out run evaluates only G1. Re-scoring both saved runs with the fixed code reproduces every recorded number.
- **A3 was skipped** because the pre-registered noise condition was met on the development half (G1: 82% against A0's 79% + 10 points). On the held-out half G1's noise, 81.7%, is also within A0's 76.2% + 10.

| Criterion, for G1 | Result | |
|---|---|---|
| No regression on the existing questions | Top 5 identical to A0's on every one | Pass |
| Refusal, the 4 naming nothing | All refused (Q48 and Q50 here, Q51 and Q52 in Phase 1) | Pass |
| Multi-hop fully recalled, held-out half | +0 points over A0 and over A0@7; +15 required | **Fail** |
| Added latency | 2.7 ms at the 95th percentile; under 500 ms required | Pass |
| Noise | 81.7%, against a limit of 86.2% | Pass |
| Entity-naming off-topic questions (recorded only) | G1 added 0 chunks to Q55 and 2 to Q56 | — |

**Stop rule.** The ceiling arm (G1 applied only to questions needing the graph) gained 0 points over A0 and over A0@7, short of the +15 required. As pre-registered, the experiment stops and reports that the graph does not help on this corpus. Production retrieval never called the graph during Step 7 and is unchanged.

**Decisions after Step 7 (owner, 2026-09-25).**

- **The graph improved some retrieval-level metrics, not the pre-registered primary one.** It found 1–2 more gold chunks on the held-out half, but did not fully recall more multi-hop questions.
- **Step 8 is not built.** Graph evidence is not fused into production answers, following the stop rule. *Superseded 2026-09-26: the owner chose to build Step 8 as a separate, experimental demonstration path, leaving the Step 7 result and `/api/chat` unchanged. See "Graph answers (Step 8)".*
- **The graph code, its stored data and extraction on upload are kept,** so the graph stays current and available for future experiments. The chatbot does not read it when answering.
- **Steps 9–13** (comparing vector-only with vector-plus-graph retrieval) take this Step 7 evaluation as their primary evidence. Nothing is re-run after the held-out result, and the held-out evaluation is not modified.

**Damaged source text costs an edge; it does not create a false one.** 15 of 3,187 proposed relations (0.5%) quoted text that was not in the chunk. The most common cause (8 of 15) was PDF text damaged in extraction — words split by a stray space ("connectio ns") or a word pypdf moved out of its sentence — which the model silently repaired when quoting. The rest were a quote stitched from a lead-in and a non-adjacent bullet, an abbreviated name, and an added full stop. The quote check rejected all 15, so the damage became a recall loss rather than a precision loss — the right direction to fail in. Nothing was changed.

### Graph answers (Step 8): an experimental demonstration, not evidence

**Step 8 demonstrates graph RAG end to end: question → entity linking → vector retrieval plus graph passages → an answer whose citations reach back to the source sentences. Step 7 measured whether graph retrieval improves retrieval on this corpus, and the answer was no.** The passages the graph adds are the evidence a question needs about one time in twenty: 2 of 24 on the development half, 1 of 29 held out, and 2 of 48 in the demonstration below. Most graph passages are noise. Nothing in this section is evidence that graph RAG helps on this corpus. It was built at the owner's request, overriding the Step 7 stop rule, to show the architecture working.

**How it is built.**
- **A separate endpoint.** `POST /api/chat/graph` is off unless `GRAPH_RAG_ENABLED=true`, and `/api/chat` is untouched.
- **Step 7's G1, unchanged.**
  1. Link the entities the question names.
  2. Take the chunks one hop away.
  3. Keep those that pass the production cosine gate and are not already in the vector top 5.
  4. Add at most 2, after the vector top 5.
- **Refusal stays structural.** When vector retrieval finds nothing, the graph is never consulted, so it can never be the only evidence.
- **The model never sees the relationship label.** It gets each graph passage with the two entity names and the verbatim sentence that linked it. Labels are 48% correct, so the label is withheld in code (see "Graph evidence is quoted, not asserted"). The API and the UI still show it, marked "machine-extracted, ~48% accurate on this corpus".
- **Graph mode keeps its own conversation history,** so neither mode's answers can leak into the other's rewrites.
- **The UI marks graph mode everywhere.**
  - The "Graph mode (experimental)" switch is off on every page load.
  - Every answer is tagged with the endpoint that produced it, read from the response.
  - Graph passages carry a badge, the path from the question and the linking sentence.

**What the checks prove.** `python -m eval.step8_check` passes 25 of 25. Each key check was also shown to fail on a planted fault.
- **Step 8's code leaves `/api/chat` byte-identical.** The same three-turn session runs through the current code and through a copy with Step 8's files removed. Embeddings are replayed, and the model is replaced by a stub whose answer is a hash of the whole request. It passes with the graph flag off and on.
  - **At the Step 8 merge (`c9feb8a`),** `/api/chat` was also byte-identical to the Step 7 commit.
  - **The close-out fixes then changed the vector path on purpose,** so the check now isolates Step 8's own contribution.
  - **Shown to fail:** a one-space prompt change was detected, and so was a planted Step 8 router that altered `/api/chat`'s snippets.
- **The relationship label never reaches the model.** This is asserted on the exact request sent, for every label that is not also a word of the passages or instructions. It is asserted again with every label replaced by a sentinel string: 99 sentinels, and none reached the prompt.
- **The rest match Step 7 and the refusal rules.**
  - The graph passages are exactly what Step 7's own G1 code chooses on all 20 development questions, run live on the same search passages.
  - Every quote is verbatim in its passage.
  - Refusals make no model call and no graph lookup.

**Latency.** The graph step costs one extra embedding call; the traversal itself is cheap.

| Over 36 graph-mode answers | p50 | p95 |
|---|---|---|
| Graph step, whole | 288 ms | 422 ms |
| of which the second embedding call | 284 ms | 418 ms |
| of which the traversal itself | 3.7 ms | 4.7 ms |

The extra embedding call is an integration choice: it kept `retrieve.py`, and the Step 7 baselines it carries, untouched. An earlier run saw one embedding call take 1,668 ms, above Step 7's 500 ms threshold.

**The demonstration.** `python -m eval.step8_demo`, saved as `eval/step8_runs/demo-20260926-080526.json`.
- **What was asked.** Both modes answered the 40 questions outside the held-out half, plus the standing refusal question.
- **"What is the capital of France?" is also held-out Q48.** It is used here only as a refusal check and is excluded from any held-out scoring.
- **Whose readings.** The readings below are the assistant's, from the saved answers. The owner's labels are pending; the labelling file sits beside the run.

**One gain and one harm, and the harm is the more instructive.** The gain is Q38, described below. The harm is Q32, "How do AWS Shield Advanced findings reach incident response?"
- **What the answer said.** "These findings support incident response by providing evidence for the response playbook." It cites a graph passage, AWS:156, which gives only the playbook's steps.
- **Why it is unsupported.** The link from findings to incident response is the unsupported hop recorded in Known limitation 16, now in a live answer. It is stated as fact, with a citation that makes it look sourced.
- **How the passage arrived.** Through the generic entity "AWS", one of the broad names the linker matches.
- **The prompt told the model not to do this:** "Never state a connection between two things unless a passage's text states it." By the owner's count, it is the fifth prompt-only rule to fail in this project. This README records two others by name: extraction's "never infer a purpose" and "never substitute a nearby entity".

A citation that looks like evidence is a worse failure than a refusal, and a rule in a prompt did not prevent it.

*Refusal flips: the same mechanism can produce a gain or a failure.*

| Questions | Vector-only declined, graph mode answered | Vector-only answered, graph mode declined |
|---|---|---|
| 31 on-topic | 1 (Q38), a gain | 0 |
| 5 off-topic, naming a corpus entity | 0 (a failure would count here) | 0 |
| 4 off-topic naming nothing, and "capital of France" | 0: refused without a model call in both modes | 0 |

The automatic reading first counted one flip each way among the entity-naming questions (Q59 and Q60). Reading the answers, both modes declined both questions; one answer also added a sentence of context with a citation. So the count is one gain and no failures, from only 5 entity-naming questions.

*Citations: does the model use the graph passages?*

| Measure | Result |
|---|---|
| Answers given at least one graph passage | 32 |
| Answers citing a graph passage | 4 (12%) |
| Graph passages cited, of those shown | 5 of 57 (9%), against 39 of 160 search passages (24%) in the same answers |
| Graph passages that are gold, of those shown to on-topic questions | 2 of 48 (4%) |

- **Mostly ignored.** The model cites a graph passage less than half as often as a search passage, so most of the noise is tolerated rather than used.
- **Cited and gold: 2.** They are the only two gold graph passages shown, and both were used (Q38 RFC:20, Q32 AWS:154).
- **Cited and not gold: 3.**
  - Two support true details: Q06 (rate limits scoped down for dynamic endpoints), and Q16 (SYN cookies at the edge, alongside the search passages).
  - **One is harm, the headline above.** In Q32, "These findings support incident response by providing evidence for the response playbook" cites AWS:156, which gives only the playbook's steps. The connection from findings to incident response is the model's own inference, the unsupported hop recorded in Known limitation 16. It is stated as fact, with a citation, despite the prompt's rule against stating connections no passage states. AWS:156 was reached through the generic entity "AWS".

**The gain, one question as an illustration rather than a result: Q38.** "What two different purposes does mutual TLS serve in RFC 8935?"
- **Vector-only declined.** Its top 5 held neither gold chunk.
- **Graph mode followed "mutual TLS" to RFC:20.** That passage was above the cosine gate (0.33) but ranked 19th by vector search, and it is gold. Graph mode answered with two purposes, both citing it:
  - authorization based on the transmitter's identity, which RFC:20 states
  - authenticating the transmitter, a fair reading of RFC:20's "or via other employed authentication methods"
- **The answer is partly correct.** The labelled second purpose, mitigating denial of service by authenticating transmitters (RFC:26), is missing, because no edge links RFC:26 to mutual TLS.
- **What it shows.** This is the clearest single picture of what the graph does on this corpus: it promotes evidence that retrieval ranked above the gate but too low to use.

### Close-out fixes: the context budget and PDF ligatures

Two gaps found earlier, fixed after Step 8, each measured before and after.

**The context budget was declared but never enforced.** `MAX_CONTEXT_CHARS` (8,000) was never checked by `/api/chat`. At 5 chunks of at most 999 characters the context is about 5,300 characters, so nothing broke. Raise `TOP_K`, though, and the budget silently did nothing.
- **The fix.** `generate.fit_context` keeps the leading whole passages that fit, and both chat routes apply it before building the prompt *and* the sources. Citations are positional (`[n]` is `sources[n-1]`), so trimming only the prompt would have misnumbered every citation after the cut.
- **Checked three ways:**
  - Twelve 999-character passages trim to 7 (7,222 characters).
  - Through `/api/chat` with retrieval forced to 12 passages, the result was 8 passages, a 7,498-character context, and every citation still aligned.
  - At today's settings `/api/chat` is byte-identical to before.

**PDF ligatures were invisible to keyword search.** PDF text keeps typographic ligatures as single characters ("ﬁ", "ﬂ", "ﬃ"), and the BM25 tokenizer keeps only `a–z` and `0–9`. So "traﬃc" became the tokens `tra` and `c`, which no query matches. **171 of the 214 chunks contain a ligature**, in 123 distinct words.
- **The fix is in `sparse.py`'s tokenizer only.** It applies NFKC normalisation before lowercasing. Extraction and the stored text are untouched, and no re-index is needed: BM25 is rebuilt from the stored text at every start.

| Query word | Chunks it matched before | After |
|---|---|---|
| traffic | 0 | 65 |
| specific | 0 | 21 |
| flood | 1 | 17 |
| configure | 0 | 13 |
| reflection | 0 | 10 |

- **The measured baseline did not move.** On the original 9 questions, MRR is still 0.587 dense, 0.713 BM25 and 0.704 hybrid, and Recall@5 is 100%.
- **Few search results changed.** Of the 40 questions outside the held-out half, 5 have a different top 5 (Q02, Q13, Q15, Q33, Q38). Gold chunks in the top 5 are unchanged at 34 of 41, with none entering or leaving. The graph passages are unchanged on all 20 development questions.
- **The honest reading.** The bug was real and large: words used dozens of times were unsearchable by keyword. But none of the labelled questions depends on one of them, so no measured number improves.
- **Step 7's numbers still stand.** They were measured at commit `44f32db`, before this fix, and the saved runs in `eval/graph_runs/` are the record. Re-running `eval.graph_retrieval_eval` today measures the current system, where 4 of the 20 development questions have a different top 5. To reproduce Step 7 exactly, check out `44f32db`.

### Entity resolution: connectivity comes from extraction, not merging

Resolution turns the 1,170 entity names the extraction produced into one key per real-world thing. Its effect on what multi-hop retrieval can use — two-hop paths whose two edges come from different chunks:

| Resolution | Admitted entities | Cross-chunk two-hop paths | Across documents |
|---|---|---|---|
| None (exact names) | 372 | 1,514 | 18 |
| Layer 1: "AWS"/"Amazon" prefixes, acronyms defined in brackets | 347 | 1,846 (+22%) | 26 |
| + all layer-3 similarity proposals, if every one were right | 334 | 1,894 | 26 |
| + folding all 53 singular/plural pairs | 329 | 1,884 | 26 |

Almost all of the gain comes from layer 1. Layer 3 and plural folding add 2–3% between them, so the graph's connectivity is set by what extraction finds, not by how cleverly names are merged — worth knowing for anyone who expects entity resolution to be where graph quality comes from. Plural folding was not adopted: 2% is not worth a new merge rule with a known trap ("HTTPS" is not the plural of "HTTP").

**Embedding similarity proposes candidates; it cannot decide them.** At a cosine similarity of 0.90 or more, layer 3 proposed 16 pairs. A person labelled 14 the same thing and 2 different — 87.5% precision (95% range 64–97%). No threshold separates the two: "CDN cache" / "CDN caching" (different) scored 0.946, higher than 11 of the 14 correct pairs, so the only error-free threshold would catch 4 of the 14. That is why layer 3 stays human-labelled rather than automated: the 14 labelled pairs are applied as recorded decisions, the 2 different pairs are kept apart and never proposed again, and new proposals wait for a person.

| | Admitted entities | Components | Cross-chunk two-hop paths |
|---|---|---|---|
| Layers 1–2 | 347 | 42 | 1,846 |
| + the 14 labelled layer-3 merges (the graph now used) | 335 | 39 | 1,892 (+2.5%) |

*Observation, not generalised:* all 14 pairs labelled the same were singular/plural variants ("SET Recipient" / "SET Recipients"), and both proposals that were not plurals were labelled different. That is consistent with a plural rule, but it is one sample of 16, and the known trap ("HTTPS" is not the plural of "HTTP") still stands — so no plural rule was added; plurals merge only case by case, when a person has labelled the pair.

Over-merging makes a graph confidently wrong, so the resolution report checks named families on every run:

| Must stay apart | Result |
|---|---|
| Shield / Shield Standard / Shield Advanced / Shield Response Team / Origin Shield | Pass |
| SET / SET Recipient / SET Transmitter / SET Issuer | Pass |
| AWS endpoint / endpoint | Pass — after a fix. The first version stripped "AWS" from any name, merging the whitepaper's AWS endpoint with the RFC's HTTP endpoint; the prefix is now removed only when what follows is itself a name ("AWS WAF") |
| Amazon VPC / virtual private cloud (VPC) | Pass — by a hand-written decision: Amazon VPC is the service, a VPC is a resource created with it |
| Security Hub / Security Hub CSPM | **Untested.** Only "Security Hub CSPM" occurs in the corpus; the rule keeps the two apart, but no real data has tested it |

**A small pilot misrepresents more than precision.** `related_to` was 21% of edges on the seven-chunk pilot and 14% (76/545) across the corpus. The pilot's chunks were unusually heavy in support-plan and escalation language, which has no verb in the vocabulary, so they overstated the catch-all rate. Figures measured on a hand-picked sample describe that sample; only the full run is large enough to judge the vocabulary.

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

**Evaluation scripts**

Run from the project root. The expected results are the current measured values.

| Command | What it verifies | Expected | Notes |
|---|---|---|---|
| `python -m eval.retrieval_eval` | Retrieval ranking on the original 9 questions: dense, BM25 and hybrid | MRR 0.587 / 0.713 / 0.704; Recall@5 100% for all three | The regression baseline. 9 embedding calls |
| `python -m eval.ocr_check data/test_documents/rfc8935.pdf` | Tesseract OCR against pypdf's text for the same rendered pages | One line per page: word counts and the share of pypdf's words recovered | Needs Tesseract installed |
| `python -m eval.graph_lifecycle_check` | Uploading and deleting with the graph: the delete cascade, a delete during extraction, and documents containing planted instructions | `17 of 17 checks passed.` | Runs on a temporary copy of `data/` and checks the real one is untouched. About 2 cents |
| `python -m eval.graph_report` | Entity resolution: merges, over-merge checks and the graph's shape | Each over-merge check prints `PASS`; Security Hub prints `UNTESTED` | Rewrites the layer-3 labelling files. A fraction of a cent |
| `python -m eval.check_graph_questions` | The Step 7 question labels: required fields, gold chunks that exist, answer facts found verbatim | `All checks passed.` | Run after any label edit; exits 1 on a problem. No API calls |
| `python -m eval.linking_recall` | How many of each question's entities the linker finds in the graph | `linking recall: 41/56 = 73%` | No API calls |
| `python -m eval.graph_retrieval_eval` | The Step 7 graph retrieval arms against A0 and A0@7, on the development half | Every arm fully recalls 2 of 5 multi-hop questions; `-> chosen: G1` | Saves a run to `eval/graph_runs/`. 20 embedding calls |
| `python -m eval.step8_check` | Step 8 wiring: `/api/chat` byte-identical with and without Step 8's code, the label never reaching the model, refusal, provenance, G1 parity, citation numbering, separate histories | `25 of 25 checks passed.` | The model is stubbed. About 110 embedding calls |
| `python -m eval.step8_demo` | Step 8 demonstration: both modes answer the 40 questions outside the held-out half; refusal flips, citation of graph passages, latency | Summary tables, then a run and a labelling file saved to `eval/step8_runs/` | A demonstration, not evidence. Real answers, a few cents; answers vary from run to run |

**Not for re-running.** These are records of steps that are closed:

- `python -m eval.graph_retrieval_eval --heldout-once`: the held-out half was run once (`eval/graph_runs/heldout-20260925-230948.json`), and a second run would not be an independent test.
- `python -m eval.extract_pilot` and `python -m eval.graph_precision_sample`: they produced the Step 5 and Step 6 labelling files. Extraction is frozen at v6, and the labels refer to saved runs; testing a changed extraction needs fresh, unlabelled text.

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
12. **Graph extraction is consistent, which is not the same as correct.**
    With the final prompt (v6), two independent extractions agree on 81% of
    edges on the seven tuned chunks and 100% on the holdout; earlier prompts
    ranged 37–72%. Removing the vaguest verb, `supports`, did more for
    agreement than any voting rule. But the holdout agreed perfectly while 42%
    of its edges were wrong: voting filters random disagreement, and a mistake
    the prompt causes is made by every run. Building the graph once and saving
    it makes it fixed, not right.
13. **pypdf moves the RFC's normative keywords out of their sentences.** As
    recorded under "Vocabulary mismatch" above, styled text runs (MUST,
    SHOULD, cross-references) are emitted after the paragraph, not in place.
    RFC 8935 has 45 resulting gaps across 22 of its 45 chunks, e.g. "the SET
    Recipient respond with" and "as defined in ."; 20 of the 25 MUSTs in the
    index sit in detached keyword runs rather than inside a clause. This
    corrupts chunk text and also the evidence quotes stored on graph edges,
    so an edge's quote can silently lose the requirement level of the
    sentence it came from.
14. **About half the graph's edges are wrong** — measured precision 48% on
    untuned text (95% range 36–61%, 58 labelled edges). Errors compound along
    a path: a two-hop path is fully right about 23% of the time, and the edges
    that can sit mid-path scored lowest of all (32%, small sample). Every
    edge's quote is verbatim document text (100% in every pilot run), so the
    Step 8 design (not built) shows graph evidence as quoted sentences with the
    relationship label marked as unreliable — see "Graph evidence is quoted,
    not asserted". The
    extraction prompt is frozen at v6.
15. **Graph extraction resists planted instructions only through its prompt.**
    A test document with an instruction to add a fake edge, placed between two
    real technical statements, was extracted correctly: the real edges, none
    from the instruction. But that is one document, and nothing in code would
    stop an edge built from such a sentence — it names both entities verbatim.
    A chunk whose extraction fails on upload is kept with its error; there is
    no retry yet short of re-running the backfill.
16. **The planned demo path is not fully supported by the text.** Shield
    Advanced → finding → Security Hub CSPM: the first hop rests on a passive
    sentence ("anomalous traffic is surfaced as a Shield Advanced finding")
    that names no agent, so it is not an edge. It is recorded rather than
    demonstrated on an inferred edge.
17. **The structural refusal only covers questions that share no topic with
    the documents.** The cosine gate (0.20) refuses "What is the capital of
    France?" before any model call, but an off-topic question that names a
    corpus entity — "What is the monthly price of AWS Shield Advanced?" —
    scores up to 0.70 and reaches the model, which must decline on its own
    instructions. For those questions, refusal is a prompt behaviour, not a
    structural guarantee.
18. **The graph is built on every upload, but only the experimental endpoint
    uses it.** Graph retrieval did not improve the pre-registered multi-hop
    metric on the held-out half (4 of 5 questions fully recalled with or without
    it), so `/api/chat` does not use it. Only the experimental
    `POST /api/chat/graph` does (Step 8). Extraction still runs in the
    background after each upload, about $0.0015 per chunk.
19. **The frontend's files have no versioning.** `app.js`, `index.html` and
    `style.css` are served under fixed names, so a browser can keep stale
    copies after an update. Since Step 8 the files depend on each other: a new
    `app.js` with an old cached `index.html` stops at its startup check, which
    requires every element it uses to exist. Until the files are versioned,
    reload with Cmd+Shift+R (Ctrl+Shift+R) after an update.
20. **Graph mode can state a connection no document states.** In the Step 8
    demonstration, one answer (Q32) cited a graph passage for a link between
    Shield Advanced findings and incident response that the documents never
    make, despite the prompt's rule against it. Graph passages are gold about
    5% of the time, and a prompt rule is the only guard against using the rest
    as a bridge.

## Future improvements

**Graph retrieval — out of scope for the first experiment**

- **Questions about scanned pages and diagrams.** Answering them needs `scanned5.pdf` and `diagram.png` in the index. Uploading them mid-experiment would change the corpus and move the retrieval baseline, and `scanned5.pdf` is a scan of whitepaper pages 21–25, so its chunks would duplicate existing ones and blur which chunk counts as the gold evidence. One variable at a time: they come after the first graph retrieval experiment.

**Graph extraction — candidate changes, each needing a fresh holdout**

The extraction prompt and its checks are frozen at v6, and the only unseen text has been used for testing. A change tested on the sample that motivated it is overfitting — it happened twice during the pilot — so each candidate below needs new, randomly drawn text to be tested on.

- **Reject metrics-table rows as evidence.** The whitepaper's CloudWatch metric tables extract as lines like "Network Load Balancer ProcessedBytes The total number of bytes processed…", which contain no verb; the model reads them as "Network Load Balancer produces ProcessedBytes". Both such edges in the 40-edge sample were wrong. A quote shaped like a table row (a service name, a CamelCase metric name, then "The total/number of…") is detectable in code. The 8 edges affected (9 quotes):

  | Edge | Source |
  |---|---|
  | Application Load Balancer —produces→ UnHealthyHostCount | whitepaper chunk 138, p. 53 |
  | Network Load Balancer —produces→ ActiveFlowCount | whitepaper chunk 138, p. 53 |
  | Network Load Balancer —produces→ NewFlowCount | whitepaper chunk 138, p. 53 |
  | Network Load Balancer —produces→ ProcessedBytes | whitepaper chunks 138 and 139, p. 53 |
  | AWS WAF —produces→ AllowedRequests | whitepaper chunk 134, p. 51 |
  | AWS WAF —produces→ BlockedRequests | whitepaper chunk 134, p. 51 |
  | AWS WAF —produces→ CountedRequests | whitepaper chunk 134, p. 51 |
  | AWS WAF —produces→ PassedRequests | whitepaper chunk 134, p. 51 |

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
- **A second graph retrieval experiment.** The first (Step 7) did not improve
  the primary multi-hop metric. A new attempt needs a fresh held-out question
  set, since the current one is spent. On the development half the measured
  limit was edge coverage, not linking or traversal: gold chunks with no edge
  at all, and a gold chunk reachable only past the cap.
  
---

## Security notes

- The API key lives in `.env`, which is gitignored, and is read by a single module. It never reaches source control, logs, or an API response.
- Upstream SDK errors are logged server-side and returned to the client as generic messages, so exception text can't leak request IDs or configuration.
- Uploads are validated for extension, size, and non-emptiness before any processing.
- Files are stored under a generated UUID, never the client-supplied filename, which prevents path traversal.
- Failed ingestion rolls back — the stored file is deleted rather than left orphaned.
- Retrieved context is delimited with XML tags and explicitly marked as untrusted data in the system prompt.
- CORS is configured permissively for local development and would be restricted to the frontend origin in a deployment.
