"use strict";

const cfg = window.MACRO_CONFIG || { FEEDBACK_WORKER_URL: "", DATA_BASE: "data" };

// ── Auth abstraction (placeholder — swap for real auth in Phase 2) ──
function currentUser() {
  let id = localStorage.getItem("macro_user_id");
  if (!id) {
    id = "anon-" + Math.random().toString(36).slice(2, 10);
    localStorage.setItem("macro_user_id", id);
  }
  return { id, name: "Anonymous" };
}

// ── Data layer (swap for an API call in Phase 2) ──
async function fetchIndex() {
  const r = await fetch(`${cfg.DATA_BASE}/index.json?_=${Date.now()}`);
  if (!r.ok) throw new Error(`index.json: ${r.status}`);
  return r.json();
}

async function fetchBriefing(id) {
  const r = await fetch(`${cfg.DATA_BASE}/briefings/${id}.json?_=${Date.now()}`);
  if (!r.ok) throw new Error(`${id}.json: ${r.status}`);
  return r.json();
}

async function submitFeedback(payload) {
  if (!cfg.FEEDBACK_WORKER_URL) {
    throw new Error("Feedback endpoint not configured. Set FEEDBACK_WORKER_URL in config.js.");
  }
  const r = await fetch(cfg.FEEDBACK_WORKER_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`Feedback failed (${r.status}): ${t.slice(0, 200)}`);
  }
  return r.json().catch(() => ({}));
}

// ── Render helpers ──
function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

function formatDate(iso) {
  const d = new Date(iso + "T12:00:00Z");
  if (isNaN(d)) return iso;
  return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", year: "numeric" });
}

function renderMarkdown(md) {
  if (window.marked) return window.marked.parse(md, { breaks: false, gfm: true });
  return md.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/\n/g, "<br>");
}

// ── Sidebar ──
function renderSidebar(briefings, activeId) {
  const list = document.getElementById("briefing-list");
  list.innerHTML = "";
  if (!briefings.length) {
    list.appendChild(el("div", { class: "muted" }, "No briefings yet."));
    return;
  }
  for (const b of briefings) {
    const item = el(
      "div",
      {
        class: "briefing-item" + (b.id === activeId ? " active" : ""),
        onclick: () => loadBriefing(b.id),
      },
      el("div", { class: "date" }, b.date),
      el("div", { class: "day" }, formatDate(b.date)),
    );
    list.appendChild(item);
  }
}

// ── Briefing view ──
function renderBriefing(doc) {
  const view = document.getElementById("briefing-view");
  view.innerHTML = "";

  const header = el(
    "div",
    { class: "briefing-header" },
    el("div", { class: "briefing-title" }, doc.title || `Macro Briefing — ${doc.date}`),
    el("div", { class: "briefing-subtitle" }, doc.subtitle || ""),
  );
  view.appendChild(header);

  for (const section of doc.sections || []) {
    const sec = el(
      "div",
      { class: "section", "data-section-id": section.id },
      el("div", { class: "section-title" }, section.title),
      el("div", { class: "section-body", html: renderMarkdown(section.markdown) }),
    );
    sec.appendChild(buildFeedbackRow(doc.id, section));
    view.appendChild(sec);
  }
}

function buildFeedbackRow(briefingId, section) {
  const row = el("div", { class: "section-feedback" });
  row.appendChild(el("span", { class: "sf-label" }, "Feedback"));

  const upBtn   = el("button", { class: "sf-thumb", title: "Useful" }, "👍");
  const downBtn = el("button", { class: "sf-thumb", title: "Off-base" }, "👎");
  const note    = el("textarea", { class: "sf-note", placeholder: "Optional note…", rows: "1" });
  const saveBtn = el("button", { class: "sf-save" }, "Send");
  const status  = el("span", { class: "sf-status" });

  let rating = null;

  upBtn.addEventListener("click", () => {
    rating = rating === "up" ? null : "up";
    upBtn.classList.toggle("active-up", rating === "up");
    downBtn.classList.remove("active-down");
  });
  downBtn.addEventListener("click", () => {
    rating = rating === "down" ? null : "down";
    downBtn.classList.toggle("active-down", rating === "down");
    upBtn.classList.remove("active-up");
  });

  saveBtn.addEventListener("click", async () => {
    if (!rating && !note.value.trim()) {
      status.textContent = "Pick a thumb or add a note.";
      status.className = "sf-status err";
      return;
    }
    saveBtn.disabled = true;
    status.textContent = "Sending…";
    status.className = "sf-status";
    try {
      await submitFeedback({
        briefing_id: briefingId,
        section_id:  section.id,
        section:     section.title,
        rating,
        note:        note.value.trim().slice(0, 1000),
        user_id:     currentUser().id,
        ts:          new Date().toISOString(),
      });
      status.textContent = "Saved — will inform tomorrow's briefing.";
      status.className = "sf-status ok";
      saveBtn.disabled = false;
    } catch (e) {
      status.textContent = e.message || "Failed.";
      status.className = "sf-status err";
      saveBtn.disabled = false;
    }
  });

  row.append(upBtn, downBtn, note, saveBtn, status);
  return row;
}

// ── Routing (very light) ──
let currentBriefingId = null;

async function loadBriefing(id) {
  currentBriefingId = id;
  history.replaceState({}, "", `#${id}`);
  document.querySelectorAll(".briefing-item").forEach((it) => {
    it.classList.toggle("active", it.querySelector(".date")?.textContent === id);
  });
  try {
    const doc = await fetchBriefing(id);
    renderBriefing(doc);
  } catch (e) {
    document.getElementById("briefing-view").innerHTML =
      `<div class="empty-state"><div class="empty-title">Couldn't load briefing</div><div class="empty-sub">${e.message}</div></div>`;
  }
}

async function init() {
  document.getElementById("user-label").textContent = currentUser().name;
  try {
    const idx = await fetchIndex();
    const briefings = idx.briefings || [];
    const initial = location.hash.slice(1) || (briefings[0] && briefings[0].id);
    renderSidebar(briefings, initial);
    if (initial) await loadBriefing(initial);
  } catch (e) {
    document.getElementById("briefing-list").innerHTML =
      `<div class="muted">Couldn't load index: ${e.message}</div>`;
  }
}

window.addEventListener("DOMContentLoaded", init);
