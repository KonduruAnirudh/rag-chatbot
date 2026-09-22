"use strict";

/* ================================================================
   Element references and state
   ================================================================ */

const els = {
  // Sidebar
  sidebar: document.getElementById("sidebar"),
  menuBtn: document.getElementById("menu-btn"),

  // Upload
  uploadBtn: document.getElementById("upload-btn"),
  fileInput: document.getElementById("file-input"),
  uploadError: document.getElementById("upload-error"),
  uploadErrorTxt: document.getElementById("upload-error-text"),
  uploadErrorX: document.getElementById("upload-error-close"),
  uploadProgress: document.getElementById("upload-progress"),
  uploadName: document.getElementById("upload-name"),
  uploadStatus: document.getElementById("upload-status"),

  // Library
  docList: document.getElementById("doc-list"),
  docEmpty: document.getElementById("doc-empty"),
  stats: document.getElementById("library-stats"),

  // Chat composer
  composer: document.getElementById("composer"),
  question: document.getElementById("question"),
  sendBtn: document.getElementById("send-btn"),
};

const state = {
  uploading: false,

  // Number of documents currently indexed.
  // This is used to determine whether chat should be enabled.
  documentCount: 0,
};

const ALLOWED_EXTENSIONS = [".pdf", ".txt"];
const MAX_FILE_BYTES = 10 * 1024 * 1024;


/* ================================================================
   Helpers
   ================================================================ */

// One place that talks to the backend.
// Turns every failure into an Error whose message is safe to show.
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
    // Response body wasn't JSON.
    // Leave data as null.
  }

  if (!response.ok) {
    const detail = data?.detail;

    // FastAPI validation errors return detail as an array.
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg).join("; ")
      : detail || `Request failed with status ${response.status}.`;

    throw new Error(message);
  }

  return data;
}


// Create an element with a class and optional text.
//
// textContent is used instead of innerHTML so filenames
// cannot inject HTML into the page.
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


/* ================================================================
   Chat composer
   ================================================================ */

/*
   The composer should only be usable when there is at least
   one document in the knowledge base.

   Example:

   documentCount = 0
        ↓
   Chat disabled

   documentCount = 1+
        ↓
   Chat enabled
*/
function updateComposer() {
  const hasDocuments = state.documentCount > 0;

  if (els.question) {
    els.question.disabled = !hasDocuments;

    if (!hasDocuments) {
      els.question.placeholder =
        "Add a document before asking a question";
    } else {
      els.question.placeholder =
        "Ask a question about your documents";
    }
  }

  if (els.sendBtn) {
    els.sendBtn.disabled = !hasDocuments;
  }

  if (els.composer) {
    els.composer.classList.toggle(
      "is-disabled",
      !hasDocuments
    );
  }
}


/* ================================================================
   Library: list documents
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
  // Newest documents first.
  const docs = [...data.documents].sort(
    (a, b) =>
      new Date(b.uploaded_at) -
      new Date(a.uploaded_at)
  );

  // Remove the old document list and render the new one.
  els.docList.replaceChildren(
    ...docs.map(renderDocument)
  );

  // Show empty state when there are no documents.
  els.docEmpty.hidden = docs.length > 0;

  // Update library statistics.
  els.stats.textContent =
    docs.length === 0
      ? "Nothing indexed yet"
      : `${plural(data.count, "document")}, ${plural(
          data.total_chunks_indexed,
          "passage"
        )} indexed`;

  // Store the current document count.
  state.documentCount = data.count;

  /*
     IMPORTANT:

     Whenever the document count changes,
     update the chat composer.

     For example:

     Upload document
        ↓
     renderDocuments()
        ↓
     state.documentCount = 1
        ↓
     updateComposer()
        ↓
     Chat becomes enabled
  */
  updateComposer();
}


function renderDocument(doc) {
  const item = el("li", "doc");

  const info = el("div", "doc-info");

  const name = el(
    "p",
    "doc-name",
    doc.filename
  );

  name.title = doc.filename;

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
   Library: upload
   ================================================================ */

function validateFile(file) {
  const dot = file.name.lastIndexOf(".");

  const ext =
    dot === -1
      ? ""
      : file.name.slice(dot).toLowerCase();

  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return `"${file.name}" isn't a PDF or TXT file.`;
  }

  if (file.size === 0) {
    return `"${file.name}" is empty.`;
  }

  if (file.size > MAX_FILE_BYTES) {
    return `"${file.name}" is larger than 10 MB.`;
  }

  return null;
}


/*
   The server doesn't report progress.

   These stages are timed messages so that a long upload
   doesn't look like the application has frozen.
*/
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

  return () => timers.forEach(clearTimeout);
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

  // IMPORTANT:
  // Backend expects the field name to be "file".
  form.append("file", file);

  try {
    await api("/api/documents", {
      method: "POST",
      body: form,
    });

    // Reload the library after successful upload.
    //
    // This eventually calls:
    // renderDocuments()
    //      ↓
    // updateComposer()
    await loadDocuments();

  } catch (err) {
    showLibraryError(err.message);

  } finally {
    stopStatusMessages();

    els.uploadProgress.hidden = true;
  }
}


async function uploadFiles(fileList) {
  if (
    state.uploading ||
    fileList.length === 0
  ) {
    return;
  }

  hideLibraryError();

  state.uploading = true;

  els.uploadBtn.disabled = true;

  // Upload one file at a time because the backend
  // accepts one file per request.
  for (const file of fileList) {
    await uploadFile(file);
  }

  state.uploading = false;

  els.uploadBtn.disabled = false;

  // Allows the same file to be selected again later.
  els.fileInput.value = "";
}


/* ================================================================
   Library: delete
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

    // Reload library after deletion.
    //
    // If this was the last document:
    //
    // loadDocuments()
    //      ↓
    // renderDocuments()
    //      ↓
    // state.documentCount = 0
    //      ↓
    // updateComposer()
    //      ↓
    // Chat becomes disabled.
    await loadDocuments();

  } catch (err) {
    showLibraryError(err.message);

    button.disabled = false;

    button.textContent = "Delete";
  }
}


/* ================================================================
   Library: error banner
   ================================================================ */

function showLibraryError(message) {
  els.uploadErrorTxt.textContent = message;

  els.uploadError.hidden = false;
}


function hideLibraryError() {
  els.uploadError.hidden = true;
}


/* ================================================================
   Wiring: upload buttons
   ================================================================ */

els.uploadBtn.addEventListener(
  "click",
  () => els.fileInput.click()
);


els.fileInput.addEventListener(
  "change",
  () => uploadFiles([...els.fileInput.files])
);


els.uploadErrorX.addEventListener(
  "click",
  hideLibraryError
);


/* ================================================================
   Drag and drop
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
    dragDepth -= 1;

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

    uploadFiles(
      [...e.dataTransfer.files]
    );
  }
);


// Prevent the browser from opening a dropped file
// if it is dropped somewhere outside the sidebar.
window.addEventListener(
  "dragover",
  (e) => e.preventDefault()
);


window.addEventListener(
  "drop",
  (e) => e.preventDefault()
);


/* ================================================================
   Mobile sidebar drawer
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
      els.sidebar.classList.contains("is-open") &&
      !els.sidebar.contains(e.target)
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
   Start application
   ================================================================ */

// Load documents when the page starts.
//
// This also calls renderDocuments()
// and therefore updateComposer().
loadDocuments();