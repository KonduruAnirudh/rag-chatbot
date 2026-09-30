"use strict";

/* ================================================================
   1. Element references
   Every element the script uses is looked up once, here.
   ================================================================ */

const els = {
  // Library
  sidebar:        document.getElementById("sidebar"),
  menuBtn:        document.getElementById("menu-btn"),
  uploadBtn:      document.getElementById("upload-btn"),
  fileInput:      document.getElementById("file-input"),
  uploadError:    document.getElementById("upload-error"),
  uploadErrorTxt: document.getElementById("upload-error-text"),
  uploadErrorX:   document.getElementById("upload-error-close"),
  uploadProgress: document.getElementById("upload-progress"),
  uploadName:     document.getElementById("upload-name"),
  uploadStatus:   document.getElementById("upload-status"),
  docList:        document.getElementById("doc-list"),
  docEmpty:       document.getElementById("doc-empty"),
  stats:          document.getElementById("library-stats"),

  // Chat
  thread:         document.getElementById("thread"),
  threadInner:    document.getElementById("thread-inner"),
  emptyState:     document.getElementById("empty-state"),
  composer:       document.getElementById("composer"),
  question:       document.getElementById("question"),
  sendBtn:        document.getElementById("send-btn"),
  charCount:      document.getElementById("char-count"),
  clearBtn:       document.getElementById("clear-btn"),

  // Question jumper
  jumper:         document.getElementById("jumper"),

  // Graph mode (Step 8, experimental)
  graphToggle:    document.getElementById("graph-toggle"),
  modeToggle:     document.getElementById("mode-toggle"),
  modeToggleNote: document.getElementById("mode-toggle-note"),
  modeHint:       document.getElementById("mode-hint"),
};


// Fail loudly and clearly if index.html and this file disagree about an id.
for (const [name, node] of Object.entries(els)) {
  if (!node) {
    throw new Error(
      `app.js: element for "${name}" not found in index.html`
    );
  }
}


/* ================================================================
   2. State and settings
   ================================================================ */

const state = {
  documentCount: 0,
  uploading: false,
  asking: false,

  // Never remembered:
  // every page load starts in vector search.
  graphMode: false,

  // Whether the backend has GRAPH_RAG_ENABLED turned on.
  graphAvailable: false,
};


/* ================================================================
   3. API modes
   ================================================================ */

const MODES = {
  vector: {
    name: "Vector search",
    endpoint: "/api/chat",
    reset: "/api/chat/reset",
  },

  graph: {
    name: "Graph mode (experimental)",
    endpoint: "/api/chat/graph",
    reset: "/api/chat/graph/reset",
  },
};


// Measured graph extraction accuracy.
const RELATION_ACCURACY = "~48%";


/* ================================================================
   4. File settings
   ================================================================ */

const ALLOWED_EXTENSIONS = [
  ".pdf",
  ".txt",
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".tiff",
  ".bmp",
];

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const MAX_QUESTION_CHARS = 2000;
const COUNTER_THRESHOLD = 1800;

let sessionId = newSessionId();


/* ================================================================
   5. Generic API helper
   ================================================================ */

async function api(path, options = {}) {
  let response;

  try {
    response = await fetch(path, options);
  } catch {
    throw new Error(
      "Could not reach the server. Check that it is running."
    );
  }

  let data = null;

  try {
    data = await response.json();
  } catch {
    // Body wasn't JSON — leave data as null.
  }

  if (!response.ok) {
    const detail = data?.detail;

    // FastAPI validation errors return detail as a list.
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg).join("; ")
      : detail || `Request failed with status ${response.status}.`;

    throw new Error(message);
  }

  return data;
}


/* ================================================================
   6. Helpers
   ================================================================ */

function el(tag, className, text) {
  const node = document.createElement(tag);

  if (className) {
    node.className = className;
  }

  if (text !== undefined) {
    node.textContent = text;
  }

  return node;
}


const plural = (n, word) =>
  `${n} ${word}${n === 1 ? "" : "s"}`;


function timeAgo(isoString) {
  const seconds = Math.floor(
    (Date.now() - new Date(isoString)) / 1000
  );

  const units = [
    ["year", 31536000],
    ["month", 2592000],
    ["day", 86400],
    ["hour", 3600],
    ["minute", 60],
  ];

  for (const [unit, size] of units) {
    const n = Math.floor(seconds / size);

    if (n >= 1) {
      return `${plural(n, unit)} ago`;
    }
  }

  return "just now";
}


function newSessionId() {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : `s-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}


/* ================================================================
   7. Library: list
   ================================================================ */

async function loadDocuments() {
  try {
    const data = await api("/api/documents");
    renderDocuments(data);
  } catch (err) {
    els.stats.textContent = "Library unavailable";
    showLibraryError(err.message);
  }
}


function renderDocuments(data) {
  const docs = [...data.documents].sort(
    (a, b) =>
      new Date(b.uploaded_at) -
      new Date(a.uploaded_at)
  );

  els.docList.replaceChildren(
    ...docs.map(renderDocument)
  );

  els.docEmpty.hidden = docs.length > 0;

  els.stats.textContent =
    docs.length === 0
      ? "Nothing indexed yet"
      : `${plural(data.count, "document")}, ${plural(
          data.total_chunks_indexed,
          "passage"
        )} indexed`;

  state.documentCount = data.count;

  updateComposer();
}


function renderDocument(doc) {
  const item = el("li", "doc");

  const info = el("div", "doc-info");

  // The filename opens the original file in a new tab.
  // rel="noopener" prevents the opened page from reaching
  // back into this page through window.opener.
  const name = el(
    "a",
    "doc-name",
    doc.filename
  );

  name.href =
    `/api/documents/${encodeURIComponent(doc.doc_id)}/file`;

  name.target = "_blank";
  name.rel = "noopener";
  name.title = `Open ${doc.filename}`;

  const meta = el(
    "p",
    "doc-meta",
    `${plural(doc.chunk_count, "passage")}, added ${timeAgo(
      doc.uploaded_at
    )}`
  );

  info.append(name, meta);


  const del = el(
    "button",
    "doc-delete",
    "Delete"
  );

  del.type = "button";

  del.setAttribute(
    "aria-label",
    `Delete ${doc.filename}`
  );

  del.addEventListener(
    "click",
    () => deleteDocument(doc, del)
  );

  item.append(info, del);

  return item;
}


/* ================================================================
   8. Library: upload
   ================================================================ */

function validateFile(file) {
  const dot = file.name.lastIndexOf(".");

  const ext =
    dot === -1
      ? ""
      : file.name.slice(dot).toLowerCase();

  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return `"${file.name}" isn't a PDF, TXT or image file.`;
  }

  if (file.size === 0) {
    return `"${file.name}" is empty.`;
  }

  if (file.size > MAX_FILE_BYTES) {
    return `"${file.name}" is larger than 10 MB.`;
  }

  return null;
}


// The server doesn't report progress, so these stages are timed
// estimates that keep a long upload from looking frozen.
function startStatusMessages() {
  const stages = [
    [0, "Uploading…"],
    [1500, "Extracting text…"],
    [4000, "Generating embeddings…"],
    [15000, "Still working. Large files take longer."],
  ];

  const timers = stages.map(([delay, text]) =>
    setTimeout(() => {
      els.uploadStatus.textContent = text;
    }, delay)
  );

  return () =>
    timers.forEach(clearTimeout);
}


async function uploadFile(file) {
  const problem = validateFile(file);

  if (problem) {
    showLibraryError(problem);
    return;
  }

  els.uploadName.textContent = file.name;
  els.uploadProgress.hidden = false;

  const stopStatusMessages =
    startStatusMessages();

  const form = new FormData();

  // Field name must be exactly "file".
  form.append("file", file);

  try {
    // Do not set Content-Type manually.
    // The browser adds the multipart boundary.
    await api("/api/documents", {
      method: "POST",
      body: form,
    });

    await loadDocuments();

  } catch (err) {
    showLibraryError(err.message);

  } finally {
    stopStatusMessages();
    els.uploadProgress.hidden = true;
  }
}


async function uploadFiles(files) {
  if (
    state.uploading ||
    files.length === 0
  ) {
    return;
  }

  hideLibraryError();

  state.uploading = true;
  els.uploadBtn.disabled = true;

  try {
    // One at a time — backend accepts one file per request.
    for (const file of files) {
      await uploadFile(file);
    }

  } finally {
    state.uploading = false;
    els.uploadBtn.disabled = false;

    // Allows the same file to be selected again.
    els.fileInput.value = "";
  }
}


/* ================================================================
   9. Library: delete
   ================================================================ */

async function deleteDocument(doc, button) {
  const confirmed = confirm(
    `Remove "${doc.filename}" from the library?\n\n` +
    `Its passages will no longer be used to answer questions.`
  );

  if (!confirmed) {
    return;
  }

  hideLibraryError();

  button.disabled = true;
  button.textContent = "Removing…";

  try {
    await api(
      `/api/documents/${encodeURIComponent(doc.doc_id)}`,
      {
        method: "DELETE",
      }
    );

    await loadDocuments();

  } catch (err) {
    showLibraryError(err.message);

    button.disabled = false;
    button.textContent = "Delete";
  }
}


/* ================================================================
   10. Library: error banner
   ================================================================ */

function showLibraryError(message) {
  els.uploadErrorTxt.textContent = message;
  els.uploadError.hidden = false;
}


function hideLibraryError() {
  els.uploadError.hidden = true;
}


/* ================================================================
   11. Chat: rendering answers safely
   Rule: escape ALL HTML first, then add the few tags we allow.
   ================================================================ */

function escapeHtml(text) {
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };

  return text.replace(
    /[&<>"']/g,
    (ch) => map[ch]
  );
}


// Runs on text that is ALREADY escaped.
function formatInline(
  escaped,
  sourceCount
) {
  return escaped
    .replace(
      /\*\*(.+?)\*\*/g,
      "<strong>$1</strong>"
    )
    .replace(
      /`([^`]+)`/g,
      "<code>$1</code>"
    )
    // Citations: [1], [1][2], and [1, 2].
    // Only create buttons for real sources.
    .replace(
      /\[(\d+(?:\s*,\s*\d+)*)\]/g,
      (match, group) => {
        const numbers = group
          .split(",")
          .map((s) => Number(s.trim()));

        const valid = numbers.every(
          (n) =>
            n >= 1 &&
            n <= sourceCount
        );

        if (!valid) {
          return match;
        }

        return numbers
          .map(
            (n) =>
              `<button class="cite" type="button" data-cite="${n}">${n}</button>`
          )
          .join("");
      }
    );
}


// A table row's cells.
function splitRow(line) {
  let row = line
    .trim()
    .replace(/\\\|/g, "\u0000");

  if (row.startsWith("|")) {
    row = row.slice(1);
  }

  if (row.endsWith("|")) {
    row = row.slice(0, -1);
  }

  return row
    .split("|")
    .map(
      (cell) =>
        cell
          .replace(/\u0000/g, "|")
          .trim()
    );
}


// The row under a table's header.
function tableAlignments(
  line,
  columns
) {
  if (!line.includes("|")) {
    return null;
  }

  const cells = splitRow(line);

  if (
    cells.length !== columns ||
    !cells.every(
      (cell) =>
        /^:?-+:?$/.test(cell)
    )
  ) {
    return null;
  }

  return cells.map((cell) => {
    if (
      cell.startsWith(":") &&
      cell.endsWith(":")
    ) {
      return "center";
    }

    if (cell.endsWith(":")) {
      return "right";
    }

    return cell.startsWith(":")
      ? "left"
      : null;
  });
}


// Cells go through formatInline.
function renderTable(
  header,
  alignments,
  rows,
  sourceCount
) {
  const cell = (
    tag,
    text,
    i
  ) => {
    const align = alignments[i]
      ? ` class="align-${alignments[i]}"`
      : "";

    return `<${tag}${align}>${formatInline(
      text,
      sourceCount
    )}</${tag}>`;
  };

  const head = header
    .map((text, i) =>
      cell("th", text, i)
    )
    .join("");

  const body = rows
    .map(
      (row) =>
        `<tr>${header
          .map((_, i) =>
            cell(
              "td",
              row[i] ?? "",
              i
            )
          )
          .join("")}</tr>`
    )
    .join("");

  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>${head}</tr>
        </thead>
        <tbody>${body}</tbody>
      </table>
    </div>
  `;
}


// Paragraphs, headings, bullet lists,
// numbered lists and tables.
function renderMarkdown(
  text,
  sourceCount
) {
  const lines =
    escapeHtml(text.trim()).split("\n");

  let html = "";
  let paragraph = [];
  let listType = null;

  const flushParagraph = () => {
    if (paragraph.length) {
      html +=
        `<p>${formatInline(
          paragraph.join(" "),
          sourceCount
        )}</p>`;

      paragraph = [];
    }
  };

  const closeList = () => {
    if (listType) {
      html += `</${listType}>`;
      listType = null;
    }
  };

  for (
    let i = 0;
    i < lines.length;
    i++
  ) {
    const line = lines[i].trim();

    // Table detection.
    const header =
      line.includes("|")
        ? splitRow(line)
        : null;

    const alignments =
      header &&
      i + 1 < lines.length
        ? tableAlignments(
            lines[i + 1],
            header.length
          )
        : null;

    if (alignments) {
      flushParagraph();
      closeList();

      const rows = [];

      let next = i + 2;

      while (
        next < lines.length &&
        lines[next].trim() !== "" &&
        lines[next].includes("|")
      ) {
        rows.push(
          splitRow(lines[next])
        );

        next += 1;
      }

      html += renderTable(
        header,
        alignments,
        rows,
        sourceCount
      );

      i = next - 1;
      continue;
    }

    const heading =
      line.match(/^#{1,6}\s+(.*)/);

    const bullet =
      line.match(/^[-*]\s+(.*)/);

    const numbered =
      line.match(/^\d+[.)]\s+(.*)/);

    const item =
      bullet || numbered;

    if (heading) {
      flushParagraph();
      closeList();

      html +=
        `<p><strong>${formatInline(
          heading[1],
          sourceCount
        )}</strong></p>`;

    } else if (item) {
      flushParagraph();

      const type =
        bullet ? "ul" : "ol";

      if (listType !== type) {
        closeList();

        html += `<${type}>`;

        listType = type;
      }

      html +=
        `<li>${formatInline(
          item[1],
          sourceCount
        )}</li>`;

    } else {
      closeList();

      if (line === "") {
        flushParagraph();
      } else {
        paragraph.push(line);
      }
    }
  }

  flushParagraph();
  closeList();

  return html;
}


function scoreClass(score) {
  if (score >= 0.45) {
    return "score-strong";
  }

  if (score >= 0.30) {
    return "score-moderate";
  }

  return "score-weak";
}


/* ================================================================
   12. Source locations
   ================================================================ */

// Where a passage came from.
// Chunks can span a page boundary.
function locationLabel(source) {
  const start = source.page_start;
  const end = source.page_end;

  if (!start) {
    return `Section ${source.chunk_index}`;
  }

  return start === end
    ? `Page ${start}`
    : `Pages ${start}\u2013${end}`;
}


// Page label as a link into the original file.
function locationElement(source) {
  const label =
    locationLabel(source);

  if (!source.doc_id) {
    return el(
      "span",
      "source-meta",
      label
    );
  }

  const link = el(
    "a",
    "source-meta source-link",
    label
  );

  const page =
    source.page_start
      ? `#page=${source.page_start}`
      : "";

  link.href =
    `/api/documents/${encodeURIComponent(
      source.doc_id
    )}/file${page}`;

  link.target = "_blank";
  link.rel = "noopener";

  link.title = source.page_start
    ? `Open ${source.filename} at page ${source.page_start}`
    : `Open ${source.filename}`;

  return link;
}


/* ================================================================
   13. Graph source links
   ================================================================ */

function renderGraphLinks(graph) {
  const box = el(
    "div",
    "graph-links"
  );

  const shown = new Set();

  for (const link of graph.links) {
    const key =
      [
        link.reached_from,
        link.to,
      ]
        .sort()
        .join("\u0000") +
      "\u0000" +
      link.quote;

    if (shown.has(key)) {
      continue;
    }

    shown.add(key);

    const block = el(
      "div",
      "graph-link"
    );

    const path = el(
      "p",
      "graph-path"
    );

    path.append(
      el(
        "span",
        "graph-path-label",
        "Path"
      ),
      el(
        "span",
        null,
        `your question → ${link.reached_from} → ${link.to}`
      )
    );


    const quote = el(
      "blockquote",
      "graph-quote",
      `“${link.quote}”`
    );

    quote.title =
      "The sentence that links this passage to your question, verbatim from the document";

    if (link.from_diagram) {
      quote.append(
        el(
          "span",
          "graph-weak",
          " (a machine-written description of a diagram: weaker evidence)"
        )
      );
    }


    const relation = el(
      "p",
      "graph-relation"
    );

    relation.append(
      el(
        "span",
        "graph-relation-label",
        "Extracted relationship"
      ),

      el(
        "span",
        null,
        `${link.subject} — ${link.relation.replace(
          /_/g,
          " "
        )} → ${link.object}`
      ),

      el(
        "span",
        "graph-relation-note",
        `machine-extracted, ${RELATION_ACCURACY} accurate on this corpus · not shown to the model`
      )
    );

    relation.title =
      `Relationship labels were measured at about ${RELATION_ACCURACY.slice(
        1
      )} correct, so the model is never given them: it sees only the two entity names and the sentence above. The label is shown here so you can check it yourself.`;

    block.append(
      path,
      quote,
      relation
    );

    box.append(block);
  }

  return box;
}


/* ================================================================
   14. Chat: sources
   ================================================================ */

function renderSources(sources) {
  const details = el(
    "details",
    "sources"
  );

  const added =
    sources.filter(
      (s) => s.retrieval === "graph"
    ).length;

  details.append(
    el(
      "summary",
      null,
      added
        ? `Based on ${plural(
            sources.length,
            "passage"
          )}: ${
            sources.length - added
          } from search, ${added} added by the graph`
        : `Based on ${plural(
            sources.length,
            "passage"
          )}`
    )
  );

  const list = el(
    "ol",
    "source-list"
  );

  sources.forEach(
    (source, index) => {
      const isGraph =
        source.retrieval === "graph";

      const item = el(
        "li",
        isGraph
          ? "source source-graph"
          : "source"
      );

      item.dataset.source =
        String(index + 1);

      const head = el(
        "div",
        "source-head"
      );

      const file = el(
        "span",
        "source-file",
        source.filename
      );

      file.title =
        source.filename;

      const score = el(
        "span",
        `score ${scoreClass(
          source.score
        )}`,
        Number(source.score).toFixed(2)
      );

      score.title =
        "Similarity to your question";

      head.append(
        file,
        locationElement(source),
        score
      );


      // OCR badge.
      if (source.ocr) {
        const badge = el(
          "span",
          "ocr-badge",
          "OCR"
        );

        badge.title =
          "Read from an image by OCR";

        head.append(badge);
      }


      // Graph badge.
      if (isGraph) {
        const badge = el(
          "span",
          "graph-badge",
          "Graph"
        );

        badge.title =
          "Added by the graph: reached by following an entity named in your question";

        head.append(badge);
      }


      const snippet =
        (source.snippet || "")
          .replace(/\s+/g, " ")
          .trim();

      item.append(head);

      if (
        isGraph &&
        source.graph
      ) {
        item.append(
          renderGraphLinks(
            source.graph
          )
        );
      }

      item.append(
        el(
          "blockquote",
          "source-text",
          `…${snippet}…`
        )
      );

      list.append(item);
    }
  );

  details.append(list);

  return details;
}


/* ================================================================
   15. Answer mode tag
   ================================================================ */

function modeTag(mode) {
  const tag = el(
    "p",
    `mode-tag mode-tag-${mode}`
  );

  tag.append(
    el(
      "span",
      "mode-tag-name",
      MODES[mode].name
    ),

    el(
      "span",
      "mode-tag-endpoint",
      `answered by ${MODES[mode].endpoint}`
    )
  );

  return tag;
}


/* ================================================================
   16. Render answer
   ================================================================ */

function renderAnswer(
  node,
  question,
  data
) {
  const sources =
    Array.isArray(data.sources)
      ? data.sources
      : [];

  const mode =
    data.retrieval_mode ===
    "vector+graph"
      ? "graph"
      : "vector";

  node.className =
    `msg msg-assistant msg-${mode}`;

  node.removeAttribute(
    "aria-label"
  );


  const body = el(
    "div",
    "msg-body"
  );

  body.innerHTML =
    renderMarkdown(
      data.answer || "",
      sources.length
    );

  node.replaceChildren(
    modeTag(mode),
    body
  );


  // Show rewritten query when different.
  const normalise = (s) =>
    s
      .trim()
      .toLowerCase()
      .replace(/\s+/g, " ");

  if (
    data.search_query &&
    normalise(data.search_query) !==
      normalise(question)
  ) {
    node.append(
      el(
        "p",
        "search-note",
        `Searched for: ${data.search_query}`
      )
    );
  }


  // Don't show empty sources box.
  if (sources.length > 0) {
    node.append(
      renderSources(sources)
    );
  }
}


/* ================================================================
   17. Chat: thread
   ================================================================ */

function addMessage(className) {
  els.emptyState.hidden = true;

  const node = el(
    "article",
    `msg ${className}`
  );

  els.threadInner.append(node);

  scrollToBottom();

  // Update question jumper whenever a message is added.
  updateJumper();

  return node;
}


function scrollToBottom() {
  els.thread.scrollTop =
    els.thread.scrollHeight;
}


function loadingDots() {
  const dots = el(
    "div",
    "dots"
  );

  dots.append(
    el("span"),
    el("span"),
    el("span")
  );

  return dots;
}


/* ================================================================
   18. Ask question
   ================================================================ */

async function askQuestion(question) {
  if (
    !question ||
    state.asking ||
    state.documentCount === 0
  ) {
    return;
  }

  state.asking = true;

  els.question.value = "";

  autoGrow();
  updateComposer();


  // User message.
  addMessage(
    "msg-user"
  ).textContent = question;


  // Freeze the mode when the question is sent.
  const mode =
    state.graphMode
      ? "graph"
      : "vector";


  const pending =
    addMessage(
      "msg-assistant msg-loading"
    );


  pending.setAttribute(
    "aria-label",
    `Waiting for answer (${MODES[mode].name})`
  );


  pending.append(
    modeTag(mode),
    loadingDots()
  );


  try {
    const data =
      await api(
        MODES[mode].endpoint,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",
          },

          body: JSON.stringify({
            question,
            session_id: sessionId,
          }),
        }
      );


    renderAnswer(
      pending,
      question,
      data
    );

  } catch (err) {
    pending.className =
      "msg msg-error";

    pending.removeAttribute(
      "aria-label"
    );

    pending.textContent =
      mode === "graph"
        ? `${MODES.graph.name}: ${err.message}`
        : err.message;

  } finally {
    state.asking = false;

    updateComposer();

    scrollToBottom();

    els.question.focus();
  }
}


/* ================================================================
   19. Citation highlighting
   ================================================================ */

function toggleCitation(cite) {
  const message =
    cite.closest(
      ".msg-assistant"
    );

  if (!message) {
    return;
  }


  const n =
    cite.dataset.cite;


  const details =
    message.querySelector(
      ".sources"
    );


  const target =
    message.querySelector(
      `.source[data-source="${n}"]`
    );


  if (
    !details ||
    !target
  ) {
    return;
  }


  const wasActive =
    cite.classList.contains(
      "is-active"
    );


  message
    .querySelectorAll(
      ".cite.is-active"
    )
    .forEach(
      (c) =>
        c.classList.remove(
          "is-active"
        )
    );


  message
    .querySelectorAll(
      ".source.is-highlighted"
    )
    .forEach(
      (s) =>
        s.classList.remove(
          "is-highlighted"
        )
    );


  if (wasActive) {
    return;
  }


  details.open = true;


  message
    .querySelectorAll(
      `.cite[data-cite="${n}"]`
    )
    .forEach(
      (c) =>
        c.classList.add(
          "is-active"
        )
    );


  target.classList.add(
    "is-highlighted"
  );


  target.scrollIntoView({
    behavior: "smooth",
    block: "nearest",
  });
}


/* ================================================================
   20. Composer
   ================================================================ */

function autoGrow() {
  els.question.style.height =
    "auto";

  els.question.style.height =
    `${Math.min(
      els.question.scrollHeight,
      160
    )}px`;
}


function updateComposer() {
  const hasDocuments =
    state.documentCount > 0;

  const length =
    els.question.value.length;


  els.question.disabled =
    !hasDocuments;


  els.question.placeholder =
    !hasDocuments
      ? "Add a document to start asking questions"
      : state.graphMode
        ? "Ask in Graph mode (experimental)"
        : "Ask a question about your documents";


  els.sendBtn.disabled =
    !hasDocuments ||
    state.asking ||
    els.question.value.trim() === "";


  els.charCount.hidden =
    length < COUNTER_THRESHOLD;


  els.charCount.textContent =
    `${length} / ${MAX_QUESTION_CHARS}`;


  els.charCount.classList.toggle(
    "is-over",
    length >= MAX_QUESTION_CHARS
  );
}


/* ================================================================
   21. Wiring: library events
   ================================================================ */

els.uploadBtn.addEventListener(
  "click",
  () => els.fileInput.click()
);


els.fileInput.addEventListener(
  "change",
  () =>
    uploadFiles([
      ...els.fileInput.files,
    ])
);


els.uploadErrorX.addEventListener(
  "click",
  hideLibraryError
);


/* ================================================================
   22. Drag and drop
   ================================================================ */

let dragDepth = 0;


els.sidebar.addEventListener(
  "dragenter",
  (e) => {
    e.preventDefault();

    dragDepth += 1;

    els.sidebar.classList.add(
      "is-dragging"
    );
  }
);


els.sidebar.addEventListener(
  "dragleave",
  () => {
    dragDepth =
      Math.max(
        0,
        dragDepth - 1
      );

    if (dragDepth === 0) {
      els.sidebar.classList.remove(
        "is-dragging"
      );
    }
  }
);


els.sidebar.addEventListener(
  "drop",
  (e) => {
    e.preventDefault();

    dragDepth = 0;

    els.sidebar.classList.remove(
      "is-dragging"
    );

    uploadFiles([
      ...e.dataTransfer.files,
    ]);
  }
);


// Prevent browser from navigating away when
// files are dropped somewhere else.
window.addEventListener(
  "dragover",
  (e) => e.preventDefault()
);

window.addEventListener(
  "drop",
  (e) => e.preventDefault()
);


/* ================================================================
   23. Mobile drawer
   ================================================================ */

els.menuBtn.addEventListener(
  "click",
  (e) => {
    e.stopPropagation();

    els.sidebar.classList.toggle(
      "is-open"
    );
  }
);


document.addEventListener(
  "click",
  (e) => {
    if (
      els.sidebar.classList.contains(
        "is-open"
      ) &&
      !els.sidebar.contains(
        e.target
      )
    ) {
      els.sidebar.classList.remove(
        "is-open"
      );
    }
  }
);


document.addEventListener(
  "keydown",
  (e) => {
    if (e.key === "Escape") {
      els.sidebar.classList.remove(
        "is-open"
      );
    }
  }
);


/* ================================================================
   24. Chat events
   ================================================================ */

// Event delegation handles citations and example buttons.
els.threadInner.addEventListener(
  "click",
  (e) => {
    const cite =
      e.target.closest(".cite");

    if (cite) {
      toggleCitation(cite);
      return;
    }


    const example =
      e.target.closest(".example");

    if (
      example &&
      !els.question.disabled
    ) {
      els.question.value =
        example.textContent.trim();

      autoGrow();
      updateComposer();

      els.question.focus();
    }
  }
);


els.question.addEventListener(
  "input",
  () => {
    autoGrow();
    updateComposer();
  }
);


els.question.addEventListener(
  "keydown",
  (e) => {
    // Don't send while an input method
    // is still composing a character.
    if (
      e.key === "Enter" &&
      !e.shiftKey &&
      !e.isComposing
    ) {
      e.preventDefault();

      askQuestion(
        els.question.value.trim()
      );
    }
  }
);


els.composer.addEventListener(
  "submit",
  (e) => {
    e.preventDefault();

    askQuestion(
      els.question.value.trim()
    );
  }
);


/* ================================================================
   25. Clear chat
   ================================================================ */

els.clearBtn.addEventListener(
  "click",
  async () => {
    if (state.asking) {
      return;
    }


    try {
      // Each mode keeps its own history on the server.
      // Clearing the chat clears both.
      for (
        const mode of Object.values(MODES)
      ) {
        await api(
          `${mode.reset}?session_id=${encodeURIComponent(
            sessionId
          )}`,
          {
            method: "POST",
          }
        );
      }

    } catch {
      // If the server couldn't clear its history,
      // start a brand-new session.
      sessionId = newSessionId();
    }


    els.threadInner
      .querySelectorAll(".msg")
      .forEach(
        (node) => node.remove()
      );


    els.emptyState.hidden = false;


    // Rebuild/hide the question jumper.
    updateJumper();


    if (!els.question.disabled) {
      els.question.focus();
    }
  }
);


/* ================================================================
   25b. Jump between questions
   One tick per question. Appears from the second question, since a
   single question needs no navigation. Hidden on phones in CSS: the
   right edge is where thumbs scroll.
   ================================================================ */

const JUMPER_MIN_QUESTIONS = 2;


function updateJumper() {
  const questions = [
    ...els.threadInner.querySelectorAll(
      ".msg-user"
    ),
  ];


  els.jumper.hidden =
    questions.length <
    JUMPER_MIN_QUESTIONS;


  if (els.jumper.hidden) {
    els.jumper.replaceChildren();
    return;
  }


  els.jumper.replaceChildren(
    ...questions.map(
      (node, index) => {
        const tick = el(
          "button",
          "jumper-tick"
        );

        tick.type = "button";


        // Accessible label for screen readers.
        tick.setAttribute(
          "aria-label",
          `Question ${index + 1}`
        );


        // Tooltip for sighted users.
        tick.title =
          node.textContent.slice(
            0,
            60
          );


        tick.addEventListener(
          "click",
          () => {
            node.scrollIntoView({
              behavior: "smooth",
              block: "start",
            });
          }
        );


        return tick;
      }
    )
  );


  markCurrentTick();
}


// Highlight the tick for whichever question
// is nearest the top of the current view.
function markCurrentTick() {
  const questions = [
    ...els.threadInner.querySelectorAll(
      ".msg-user"
    ),
  ];


  const ticks = [
    ...els.jumper.querySelectorAll(
      ".jumper-tick"
    ),
  ];


  if (ticks.length === 0) {
    return;
  }


  const top =
    els.thread.getBoundingClientRect()
      .top;


  let current = 0;


  questions.forEach(
    (node, index) => {
      // The last question whose top edge is
      // at or above the viewport top.
      if (
        node.getBoundingClientRect()
          .top -
          top <=
        8
      ) {
        current = index;
      }
    }
  );


  ticks.forEach(
    (tick, index) => {
      tick.classList.toggle(
        "is-current",
        index === current
      );
    }
  );
}


// Keep the current tick synchronized with scrolling.
// requestAnimationFrame keeps work off the critical
// scroll handler path.
let jumperFrame = null;


els.thread.addEventListener(
  "scroll",
  () => {
    if (jumperFrame) {
      return;
    }


    jumperFrame =
      requestAnimationFrame(
        () => {
          jumperFrame = null;

          markCurrentTick();
        }
      );
  }
);


/* ================================================================
   26. Graph mode (Step 8, experimental)
   ================================================================ */

// The toggle, composer hint, placeholder and
// composer's colour all say which mode the next
// question goes to.
function applyMode() {
  const mode =
    state.graphMode
      ? "graph"
      : "vector";


  document.body.classList.toggle(
    "graph-mode",
    state.graphMode
  );


  els.graphToggle.checked =
    state.graphMode;


  els.modeHint.textContent =
    `${MODES[mode].name} · ${MODES[mode].endpoint}`;


  updateComposer();
}


// Each mode keeps its own history on the server,
// so switching mode mid-conversation starts that
// mode without the other's earlier turns.
function noteModeSwitch() {
  if (
    !els.threadInner.querySelector(
      ".msg-user"
    )
  ) {
    return;
  }


  const mode =
    state.graphMode
      ? "graph"
      : "vector";


  addMessage(
    `msg-divider msg-divider-${mode}`
  ).textContent =
    `Switched to ${MODES[mode].name}. It keeps its own history: earlier answers from the other mode are not part of this conversation.`;
}


async function loadGraphStatus() {
  try {
    const status =
      await api(
        "/api/chat/graph/status"
      );

    state.graphAvailable =
      Boolean(status.enabled);

  } catch {
    state.graphAvailable = false;
  }


  els.graphToggle.disabled =
    !state.graphAvailable;


  els.modeToggle.classList.toggle(
    "is-unavailable",
    !state.graphAvailable
  );


  els.modeToggleNote.textContent =
    state.graphAvailable
      ? "vector + graph"
      : "off on this server";


  els.modeToggle.title =
    state.graphAvailable
      ? "Adds up to two passages found by following entities named in your question. Step 7 found this does not improve retrieval on this corpus: it is a demonstration, not an improvement."
      : "Graph mode is off on this server. Set GRAPH_RAG_ENABLED=true in .env and restart to try it.";
}


els.graphToggle.addEventListener(
  "change",
  () => {
    if (state.asking) {
      // A question in flight keeps the mode
      // it was sent with.
      els.graphToggle.checked =
        state.graphMode;

      return;
    }


    state.graphMode =
      els.graphToggle.checked &&
      state.graphAvailable;


    applyMode();

    noteModeSwitch();
  }
);


/* ================================================================
   27. Start
   ================================================================ */

applyMode();

loadDocuments();

loadGraphStatus();