"""
Macro Briefing — cloud web app.
Deploy on Railway. Accessible from any device, anywhere.
"""

import json
import os
import re
import queue
import threading
from datetime import date, datetime
from pathlib import Path
from functools import wraps

from flask import (
    Flask, Response, jsonify, render_template_string,
    request, session, redirect, url_for
)
from apscheduler.schedulers.background import BackgroundScheduler

import briefing as briefing_module

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent / "data"
BRIEFINGS_DIR  = DATA_DIR / "briefings"
FEEDBACK_FILE  = DATA_DIR / "feedback.json"
PAUSE_FILE     = DATA_DIR / ".paused"
DATA_DIR.mkdir(exist_ok=True)
BRIEFINGS_DIR.mkdir(exist_ok=True)

# ── Auth ───────────────────────────────────────────────────────────────────────
APP_PASSWORD = os.environ.get("APP_PASSWORD", "changeme")

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

@app.route("/health")
def health():
    return "ok", 200

@app.route("/login", methods=["GET", "POST"])
def login_page():
    error = ""
    if request.method == "POST":
        pw = request.form.get("password", "")
        if pw == APP_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("index"))
        error = "Incorrect password."
    return render_template_string(LOGIN_HTML, error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_status():
    today = date.today().strftime("%Y-%m-%d")
    briefings = sorted(BRIEFINGS_DIR.glob("macro-briefing-*.md"), reverse=True)
    return {
        "paused": PAUSE_FILE.exists(),
        "today_done": (BRIEFINGS_DIR / f"macro-briefing-{today}.md").exists(),
        "today_date": today,
        "briefing_count": len(briefings),
        "latest_briefing": briefings[0].name if briefings else None,
    }

def load_feedback():
    if not FEEDBACK_FILE.exists():
        return {}
    with open(FEEDBACK_FILE) as f:
        return json.load(f)

def save_feedback_data(data):
    FEEDBACK_FILE.write_text(json.dumps(data, indent=2))

def list_briefings():
    return [f.name for f in sorted(BRIEFINGS_DIR.glob("macro-briefing-*.md"), reverse=True)]

# ── Generation (streaming via SSE) ────────────────────────────────────────────
_gen_lock = threading.Lock()  # only one generation at a time

@app.route("/api/generate", methods=["GET"])
@login_required
def api_generate():
    if PAUSE_FILE.exists():
        def paused():
            yield "data: Briefings are paused. Resume in Settings.\n\n"
            yield "data: __DONE_ERROR__\n\n"
        return Response(paused(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    today = date.today().strftime("%Y-%m-%d")
    output_file = BRIEFINGS_DIR / f"macro-briefing-{today}.md"

    q = queue.Queue()

    def run():
        try:
            full_text = briefing_module.generate_briefing(
                stream_callback=lambda chunk: q.put(("chunk", chunk))
            )
            output_file.write_text(full_text)
            q.put(("done", output_file.name))
        except Exception as e:
            q.put(("error", str(e)))

    t = threading.Thread(target=run, daemon=True)
    t.start()

    def stream():
        buffer = ""
        while True:
            try:
                kind, payload = q.get(timeout=120)
            except queue.Empty:
                yield "data: __DONE_ERROR__\n\n"
                return

            if kind == "chunk":
                # Buffer and flush line by line so SSE frames stay clean
                buffer += payload
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    yield f"data: {line}\n\n"
            elif kind == "done":
                if buffer.strip():
                    yield f"data: {buffer}\n\n"
                yield f"data: __DONE_OK__{payload}\n\n"
                return
            elif kind == "error":
                yield f"data: Error: {payload}\n\n"
                yield "data: __DONE_ERROR__\n\n"
                return

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Standard routes ────────────────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    return render_template_string(HTML)

@app.route("/api/status")
@login_required
def api_status():
    return jsonify(get_status())

@app.route("/api/briefings")
@login_required
def api_briefings():
    return jsonify(list_briefings())

@app.route("/api/briefing/<filename>")
@login_required
def api_briefing(filename):
    if not re.match(r"^macro-briefing-\d{4}-\d{2}-\d{2}\.md$", filename):
        return jsonify({"error": "Invalid filename"}), 400
    path = BRIEFINGS_DIR / filename
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    return jsonify({"content": path.read_text()})

@app.route("/api/delete-today", methods=["POST"])
@login_required
def api_delete_today():
    today = date.today().strftime("%Y-%m-%d")
    path = BRIEFINGS_DIR / f"macro-briefing-{today}.md"
    if path.exists():
        path.unlink()
    return jsonify({"ok": True})

@app.route("/api/pause", methods=["POST"])
@login_required
def api_pause():
    reason = (request.json or {}).get("reason", f"Paused {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    PAUSE_FILE.write_text(reason)
    return jsonify({"ok": True})

@app.route("/api/resume", methods=["POST"])
@login_required
def api_resume():
    if PAUSE_FILE.exists():
        PAUSE_FILE.unlink()
    return jsonify({"ok": True})

@app.route("/api/feedback", methods=["GET"])
@login_required
def api_get_feedback():
    filename = request.args.get("filename", "")
    if not re.match(r"^macro-briefing-\d{4}-\d{2}-\d{2}\.md$", filename):
        return jsonify({"error": "Invalid"}), 400
    date_key = filename.replace("macro-briefing-", "").replace(".md", "")
    return jsonify(load_feedback().get(date_key, []))

@app.route("/api/feedback", methods=["POST"])
@login_required
def api_save_feedback():
    body = request.json
    filename = body.get("filename", "")
    if not re.match(r"^macro-briefing-\d{4}-\d{2}-\d{2}\.md$", filename):
        return jsonify({"error": "Invalid"}), 400
    date_key = filename.replace("macro-briefing-", "").replace(".md", "")
    entries = [
        {
            "trade": str(e.get("trade", ""))[:500],
            "rating": e.get("rating") if e.get("rating") in ("up", "down", None) else None,
            "note": str(e.get("note", ""))[:1000],
        }
        for e in body.get("entries", [])
    ]
    data = load_feedback()
    data[date_key] = entries
    save_feedback_data(data)
    return jsonify({"ok": True})

# ── Scheduler (auto-generate at 6 AM ET Mon-Fri) ──────────────────────────────
def scheduled_generate():
    if PAUSE_FILE.exists():
        return
    today = date.today().strftime("%Y-%m-%d")
    output_file = BRIEFINGS_DIR / f"macro-briefing-{today}.md"
    if output_file.exists():
        return
    try:
        text = briefing_module.generate_briefing()
        output_file.write_text(text)
        print(f"[scheduler] Briefing generated: {output_file.name}")
    except Exception as e:
        print(f"[scheduler] Generation failed: {e}")

try:
    scheduler = BackgroundScheduler(timezone="America/New_York")
    scheduler.add_job(scheduled_generate, "cron", day_of_week="mon-fri", hour=6, minute=0)
    scheduler.start()
except Exception as e:
    print(f"[scheduler] Failed to start: {e}")

# ── Login page HTML ────────────────────────────────────────────────────────────
LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Macro Briefing — Login</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; color: #e8e8e8; font-family: -apple-system, sans-serif;
         display: flex; align-items: center; justify-content: center; min-height: 100vh; }
  .box { background: #161616; border: 1px solid #2a2a2a; border-radius: 12px;
         padding: 40px 36px; width: 100%; max-width: 360px; }
  h1 { font-size: 20px; color: #c8a96e; margin-bottom: 6px; }
  p  { font-size: 13px; color: #666; margin-bottom: 28px; }
  input[type=password] { width: 100%; background: #0d0d0d; border: 1px solid #2a2a2a;
    border-radius: 8px; padding: 12px 14px; color: #e8e8e8; font-size: 15px; margin-bottom: 14px; }
  input[type=password]:focus { outline: none; border-color: #c8a96e; }
  button { width: 100%; background: #c8a96e; color: #000; border: none; border-radius: 8px;
           padding: 12px; font-size: 15px; font-weight: 600; cursor: pointer; }
  .error { color: #e05252; font-size: 13px; margin-top: 12px; }
</style>
</head>
<body>
<div class="box">
  <h1>Macro Briefing</h1>
  <p>Arjun Parikh // QIS</p>
  <form method="post">
    <input type="password" name="password" placeholder="Password" autofocus>
    <button type="submit">Sign in</button>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
  </form>
</div>
</body>
</html>"""

# ── Main app HTML ──────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Macro Briefing</title>
<style>
  :root {
    --bg: #0d0d0d; --surface: #161616; --border: #2a2a2a;
    --accent: #c8a96e; --accent-dim: #8a6e3e;
    --text: #e8e8e8; --muted: #666; --green: #4caf6e; --red: #e05252; --radius: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         font-size: 15px; min-height: 100vh; }
  .app { display: flex; flex-direction: column; min-height: 100vh; }
  header { padding: 16px 20px; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; justify-content: space-between;
           background: var(--surface); position: sticky; top: 0; z-index: 10; }
  .logo { font-size: 17px; font-weight: 600; letter-spacing: 0.02em; color: var(--accent); }
  .logo span { color: var(--muted); font-weight: 400; }
  .header-right { display: flex; gap: 10px; align-items: center; }
  main { flex: 1; padding: 20px; max-width: 900px; width: 100%; margin: 0 auto; }

  .btn { display: inline-flex; align-items: center; justify-content: center; gap: 7px;
         padding: 10px 18px; border-radius: var(--radius); font-size: 14px; font-weight: 500;
         cursor: pointer; border: none; transition: opacity 0.15s, transform 0.1s; white-space: nowrap; }
  .btn:active { transform: scale(0.97); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .btn-primary { background: var(--accent); color: #000; }
  .btn-secondary { background: var(--surface); color: var(--text); border: 1px solid var(--border); }
  .btn-sm { padding: 7px 13px; font-size: 13px; }
  .btn-large { padding: 14px 28px; font-size: 16px; font-weight: 600; width: 100%; }

  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: var(--radius); padding: 20px; margin-bottom: 16px; }
  .card-title { font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
                text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }

  .pill { display: inline-flex; align-items: center; gap: 5px;
          padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; }
  .pill-green { background: rgba(76,175,110,0.15); color: var(--green); }
  .pill-red   { background: rgba(224,82,82,0.15);  color: var(--red); }
  .pill-gold  { background: rgba(200,169,110,0.15); color: var(--accent); }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

  .generate-area { text-align: center; padding: 10px 0 4px; }
  .generate-area .sub { color: var(--muted); font-size: 13px; margin-top: 8px; }

  .log-box { background: #0a0a0a; border: 1px solid var(--border); border-radius: 8px;
             padding: 14px 16px; font-family: "SF Mono", "Menlo", monospace;
             font-size: 12px; color: #aaa; max-height: 220px; overflow-y: auto;
             display: none; margin-top: 14px; line-height: 1.6; }
  .log-box.visible { display: block; }
  .log-line-ok  { color: var(--green); }
  .log-line-err { color: var(--red); }

  .briefing-content { font-family: "Georgia", serif; font-size: 15px; line-height: 1.75;
                      color: #d8d8d8; white-space: pre-wrap; word-break: break-word; }
  .briefing-content h1 { font-size: 20px; color: var(--accent); margin-bottom: 16px;
                         font-family: -apple-system, sans-serif; }
  .briefing-content h2 { font-size: 16px; color: var(--text); margin: 22px 0 8px;
                         font-family: -apple-system, sans-serif;
                         border-bottom: 1px solid var(--border); padding-bottom: 5px; }
  .briefing-content h3 { font-size: 14px; color: var(--accent-dim); margin: 16px 0 6px;
                         font-family: -apple-system, sans-serif; }

  .history-item { display: flex; align-items: center; justify-content: space-between;
                  padding: 10px 0; border-bottom: 1px solid var(--border); gap: 10px; }
  .history-item:last-child { border-bottom: none; }
  .history-date { font-size: 14px; font-weight: 500; }
  .history-actions { display: flex; gap: 8px; }

  /* Feedback */
  .feedback-card { background: #0f0f0f; border: 1px solid var(--border);
                   border-radius: var(--radius); padding: 18px 20px; margin-bottom: 16px; }
  .feedback-card-title { font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
                         text-transform: uppercase; color: var(--accent); margin-bottom: 14px;
                         display: flex; align-items: center; gap: 8px; }
  .trade-feedback-item { border: 1px solid var(--border); border-radius: 8px;
                         padding: 14px 16px; margin-bottom: 10px; background: var(--surface); }
  .trade-text { font-size: 13px; color: #ccc; line-height: 1.5; margin-bottom: 10px; white-space: pre-wrap; }
  .feedback-controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .thumb-btn { background: none; border: 1px solid var(--border); border-radius: 6px;
               padding: 5px 12px; cursor: pointer; font-size: 16px; transition: all 0.15s; color: var(--muted); }
  .thumb-btn:hover { border-color: var(--accent); }
  .thumb-btn.active-up   { background: rgba(76,175,110,0.15); border-color: var(--green); color: var(--green); }
  .thumb-btn.active-down { background: rgba(224,82,82,0.15);  border-color: var(--red);   color: var(--red); }
  .feedback-note { flex: 1; min-width: 160px; background: #0d0d0d; border: 1px solid var(--border);
                   border-radius: 6px; padding: 6px 10px; color: var(--text); font-size: 12px;
                   font-family: inherit; resize: none; height: 34px; line-height: 1.4; transition: border-color 0.15s; }
  .feedback-note:focus { outline: none; border-color: var(--accent); height: 60px; }
  .feedback-save-btn { background: var(--accent); color: #000; border: none; border-radius: 6px;
                       padding: 7px 16px; font-size: 13px; font-weight: 600; cursor: pointer; }
  .feedback-save-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .feedback-saved-msg { font-size: 12px; color: var(--green); display: none; }

  /* Settings form */
  .form-group { margin-bottom: 14px; }
  .form-group label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .form-group input { width: 100%; background: #0d0d0d; border: 1px solid var(--border);
                      border-radius: 8px; padding: 10px 12px; color: var(--text); font-size: 14px; }
  .form-group input:focus { outline: none; border-color: var(--accent); }
  .form-note { font-size: 12px; color: var(--muted); margin-top: 4px; }

  /* Tab bar */
  .tab-bar { display: none; position: fixed; bottom: 0; left: 0; right: 0;
             background: var(--surface); border-top: 1px solid var(--border);
             padding: 8px 0 env(safe-area-inset-bottom, 8px); z-index: 20; }
  .tab-bar-inner { display: flex; justify-content: space-around; }
  .tab-btn { display: flex; flex-direction: column; align-items: center; gap: 3px;
             background: none; border: none; color: var(--muted); font-size: 10px; cursor: pointer; padding: 4px 16px; }
  .tab-btn.active { color: var(--accent); }
  .tab-btn svg { width: 22px; height: 22px; }

  @media (max-width: 640px) {
    header { padding: 12px 16px; }
    main { padding: 14px 14px 80px; }
    .tab-bar { display: block; }
    .desktop-nav { display: none; }
    .btn-large { font-size: 15px; }
  }

  .section { display: none; }
  .section.active { display: block; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .spacer { flex: 1; }
  .empty { color: var(--muted); font-size: 14px; text-align: center; padding: 30px 0; }
  .spinner { width: 16px; height: 16px; border: 2px solid rgba(200,169,110,0.3);
             border-top-color: var(--accent); border-radius: 50%;
             animation: spin 0.7s linear infinite; display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="app">

<header>
  <div class="logo">Macro Briefing <span>// Arjun Parikh</span></div>
  <div class="header-right desktop-nav">
    <button class="btn btn-secondary btn-sm" onclick="showSection('history')">History</button>
    <button class="btn btn-secondary btn-sm" onclick="showSection('settings')">Settings</button>
    <a href="/logout" class="btn btn-secondary btn-sm">Sign out</a>
  </div>
</header>

<main>

  <!-- HOME -->
  <div id="sec-home" class="section active">
    <div class="card">
      <div class="card-title">Today's Status</div>
      <div class="row" id="status-row"><div class="spinner"></div></div>
    </div>
    <div class="card">
      <div class="card-title">Generate Briefing</div>
      <div class="generate-area">
        <button id="gen-btn" class="btn btn-primary btn-large" onclick="generateBriefing()">
          Generate Today's Briefing
        </button>
        <div class="sub" id="gen-sub">Searches live macro news + streams directly to you</div>
      </div>
      <div id="log-box" class="log-box"></div>
    </div>
    <div id="briefing-card" class="card" style="display:none">
      <div class="card-title">
        <div class="row">
          <span id="briefing-card-title">Today's Briefing</span>
          <div class="spacer"></div>
        </div>
      </div>
      <div id="briefing-content" class="briefing-content"></div>
    </div>
    <div id="feedback-card" class="feedback-card" style="display:none">
      <div class="feedback-card-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
        Trade Feedback — help improve future briefings
      </div>
      <p style="font-size:12px;color:var(--muted);margin-bottom:14px;line-height:1.5">
        Rate each trade idea. Thumbs down = wrong direction or poorly structured. Your notes teach future briefings.
      </p>
      <div id="feedback-items"></div>
      <div style="display:flex;align-items:center;gap:12px;margin-top:6px">
        <button class="feedback-save-btn" onclick="saveFeedback()">Save Feedback</button>
        <span class="feedback-saved-msg" id="feedback-saved-msg">Saved — next briefing will learn from this</span>
      </div>
    </div>
  </div>

  <!-- HISTORY -->
  <div id="sec-history" class="section">
    <div class="card">
      <div class="card-title">Past Briefings</div>
      <div id="history-list"><div class="empty">No briefings yet.</div></div>
    </div>
  </div>

  <!-- SETTINGS -->
  <div id="sec-settings" class="section">
    <div class="card">
      <div class="card-title">Schedule</div>
      <div class="row" id="schedule-row" style="margin-bottom:14px"><div class="spinner"></div></div>
      <div class="row" style="gap:10px">
        <button class="btn btn-secondary" onclick="pauseBriefings()">Pause</button>
        <button class="btn btn-primary" onclick="resumeBriefings()">Resume</button>
      </div>
    </div>
    <div class="card">
      <div class="card-title">About</div>
      <p style="font-size:13px;color:var(--muted);line-height:1.6">
        Briefings auto-generate at <strong style="color:var(--text)">6 AM ET Mon–Fri</strong> using the Anthropic API + live web search.<br><br>
        Feedback you leave on trade ideas is permanently stored and injected into future prompts — the system learns your preferences over time.
      </p>
    </div>
  </div>

</main>

<nav class="tab-bar">
  <div class="tab-bar-inner">
    <button class="tab-btn active" id="tab-home" onclick="showSection('home')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 9.5L12 3l9 6.5V20a1 1 0 01-1 1H4a1 1 0 01-1-1V9.5z"/></svg>
      Home
    </button>
    <button class="tab-btn" id="tab-history" onclick="showSection('history')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/></svg>
      History
    </button>
    <button class="tab-btn" id="tab-settings" onclick="showSection('settings')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="3"/><path d="M12 2v2m0 16v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M2 12h2m16 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
      Settings
    </button>
  </div>
</nav>

</div>
<script>
let currentSection = 'home';
let currentBriefingFile = null;
let statusData = null;

function showSection(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('sec-' + name).classList.add('active');
  const tab = document.getElementById('tab-' + name);
  if (tab) tab.classList.add('active');
  currentSection = name;
  if (name === 'history') loadHistory();
  if (name === 'settings') loadSettings();
}

async function loadStatus() {
  const row = document.getElementById('status-row');
  try {
    const r = await fetch('/api/status');
    if (!r.ok) throw new Error(`${r.status}`);
    statusData = await r.json();
    let pills = '';
    pills += statusData.paused
      ? `<span class="pill pill-red"><span class="dot"></span>Paused</span>`
      : `<span class="pill pill-green"><span class="dot"></span>Active</span>`;
    pills += statusData.today_done
      ? `<span class="pill pill-gold"><span class="dot"></span>Today's briefing ready</span>`
      : `<span class="pill pill-red"><span class="dot"></span>Not yet generated today</span>`;
    row.innerHTML = pills;
    if (statusData.today_done) {
      const fname = `macro-briefing-${statusData.today_date}.md`;
      await loadBriefing(fname);
      document.getElementById('gen-btn').textContent = "Regenerate Today's Briefing";
      document.getElementById('gen-sub').textContent = `Briefing exists for ${statusData.today_date}. Click to regenerate.`;
    }
  } catch(e) {
    row.innerHTML = `<span style="color:var(--red);font-size:13px">Error: ${e.message}</span>`;
  }
}

async function generateBriefing() {
  const btn = document.getElementById('gen-btn');
  const logBox = document.getElementById('log-box');
  const sub = document.getElementById('gen-sub');
  if (statusData && statusData.today_done) {
    await fetch('/api/delete-today', { method: 'POST' });
  }
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Generating...';
  sub.textContent = 'Searching live macro news — takes 1-3 minutes...';
  logBox.innerHTML = '';
  logBox.classList.add('visible');
  const evtSource = new EventSource('/api/generate');
  evtSource.onmessage = (e) => {
    const msg = e.data;
    if (msg.startsWith('__DONE_OK__')) {
      evtSource.close();
      const fname = msg.replace('__DONE_OK__', '').trim();
      btn.disabled = false;
      btn.innerHTML = "Generate Today's Briefing";
      sub.textContent = 'Done!';
      appendLog(logBox, 'Complete.', 'ok');
      loadBriefing(fname);
      loadStatus();
      return;
    }
    if (msg === '__DONE_ERROR__') {
      evtSource.close();
      btn.disabled = false;
      btn.innerHTML = "Generate Today's Briefing";
      sub.textContent = 'Generation failed. Check log.';
      appendLog(logBox, 'Failed.', 'err');
      return;
    }
    appendLog(logBox, msg, msg.startsWith('Error') ? 'err' : '');
  };
  evtSource.onerror = () => {
    evtSource.close();
    btn.disabled = false;
    btn.innerHTML = "Generate Today's Briefing";
    appendLog(logBox, 'Connection lost.', 'err');
  };
}

function appendLog(box, text, type) {
  const line = document.createElement('div');
  if (type === 'ok') line.className = 'log-line-ok';
  if (type === 'err') line.className = 'log-line-err';
  line.textContent = text;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

async function loadBriefing(filename) {
  currentBriefingFile = filename;
  const r = await fetch(`/api/briefing/${filename}`);
  if (!r.ok) return;
  const data = await r.json();
  const dateStr = filename.replace('macro-briefing-', '').replace('.md', '');
  document.getElementById('briefing-card-title').textContent = `Briefing — ${dateStr}`;
  document.getElementById('briefing-content').innerHTML = renderMarkdown(data.content);
  document.getElementById('briefing-card').style.display = 'block';
  await renderFeedback(filename, data.content);
}

function renderMarkdown(md) {
  return md
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n{2,}/g, '</p><p>')
    .replace(/^/, '<p>').replace(/$/, '</p>');
}

async function loadHistory() {
  const r = await fetch('/api/briefings');
  const files = await r.json();
  const list = document.getElementById('history-list');
  if (!files.length) { list.innerHTML = '<div class="empty">No briefings yet.</div>'; return; }
  list.innerHTML = files.map(f => {
    const d = f.replace('macro-briefing-', '').replace('.md', '');
    return `<div class="history-item">
      <span class="history-date">${d}</span>
      <div class="history-actions">
        <button class="btn btn-secondary btn-sm" onclick="viewHistory('${f}')">View</button>
      </div>
    </div>`;
  }).join('');
}

async function viewHistory(fname) {
  await loadBriefing(fname);
  showSection('home');
  window.scrollTo(0, document.getElementById('briefing-card').offsetTop - 80);
}

async function loadSettings() {
  const r = await fetch('/api/status');
  const s = await r.json();
  const schedRow = document.getElementById('schedule-row');
  schedRow.innerHTML = s.paused
    ? `<span class="pill pill-red"><span class="dot"></span>Paused</span>`
    : `<span class="pill pill-green"><span class="dot"></span>Active — Mon-Fri 6 AM ET</span>`;
}

async function pauseBriefings() {
  await fetch('/api/pause', { method: 'POST', headers: {'Content-Type':'application/json'}, body: '{}' });
  loadStatus(); loadSettings();
}

async function resumeBriefings() {
  await fetch('/api/resume', { method: 'POST', headers: {'Content-Type':'application/json'}, body: '{}' });
  loadStatus(); loadSettings();
}

// Feedback
function extractTrades(md) {
  const section = md.match(/##\s*Trade Construction Context([\s\S]*?)(?=\n##\s|\n#\s|$)/i);
  if (!section) return [];
  const text = section[1].trim();
  if (!text) return [];
  const raw = text.split(/\n(?=\d+\.|[-*]|\*\*|Given\s)/i)
    .map(s => s.replace(/^[\d.\-*]+\s*/, '').trim())
    .filter(s => s.length > 30);
  return raw.length ? raw : [text];
}

async function renderFeedback(filename, md) {
  const trades = extractTrades(md);
  const card = document.getElementById('feedback-card');
  const container = document.getElementById('feedback-items');
  if (!trades.length) { card.style.display = 'none'; return; }
  const r = await fetch(`/api/feedback?filename=${filename}`);
  const existing = r.ok ? await r.json() : [];
  container.innerHTML = '';
  trades.forEach((trade, i) => {
    const saved = existing[i] || {};
    const upActive   = saved.rating === 'up'   ? 'active-up'   : '';
    const downActive = saved.rating === 'down' ? 'active-down' : '';
    const item = document.createElement('div');
    item.className = 'trade-feedback-item';
    item.dataset.index = i;
    item.dataset.trade = trade;
    item.innerHTML = `
      <div class="trade-text">${trade.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>
      <div class="feedback-controls">
        <button class="thumb-btn ${upActive}"   data-dir="up"   onclick="toggleThumb(this)">👍</button>
        <button class="thumb-btn ${downActive}" data-dir="down" onclick="toggleThumb(this)">👎</button>
        <textarea class="feedback-note" placeholder="Why? e.g. wrong tenor, direction off, crowded...">${saved.note||''}</textarea>
      </div>`;
    container.appendChild(item);
  });
  card.style.display = 'block';
}

function toggleThumb(btn) {
  const item = btn.closest('.trade-feedback-item');
  const dir = btn.dataset.dir;
  const wasActive = btn.classList.contains(`active-${dir}`);
  item.querySelectorAll('.thumb-btn').forEach(b => b.classList.remove('active-up','active-down'));
  if (!wasActive) btn.classList.add(`active-${dir}`);
}

async function saveFeedback() {
  if (!currentBriefingFile) return;
  const saveBtn = document.querySelector('.feedback-save-btn');
  const savedMsg = document.getElementById('feedback-saved-msg');
  saveBtn.disabled = true;
  const entries = [];
  document.querySelectorAll('.trade-feedback-item').forEach(item => {
    const upBtn   = item.querySelector('[data-dir="up"]');
    const downBtn = item.querySelector('[data-dir="down"]');
    const note    = item.querySelector('.feedback-note').value.trim();
    let rating = null;
    if (upBtn.classList.contains('active-up'))     rating = 'up';
    if (downBtn.classList.contains('active-down')) rating = 'down';
    entries.push({ trade: item.dataset.trade, rating, note });
  });
  const r = await fetch('/api/feedback', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ filename: currentBriefingFile, entries })
  });
  saveBtn.disabled = false;
  if (r.ok) {
    savedMsg.style.display = 'inline';
    setTimeout(() => { savedMsg.style.display = 'none'; }, 4000);
  }
}

loadStatus();
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
