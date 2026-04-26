# Macro Briefing

Self-contained daily macro briefing for rates, FX, and cross-currency basis. Generates structured morning briefings from live data, learns from user feedback, and serves them as a free static site — no servers, no LLM API costs, no monthly bill.

---

## What it does

Every weekday at ~6 AM ET, a GitHub Action runs the briefing pipeline:

1. Pulls live data from **FRED** (~20 macro series), **~14 RSS feeds** (central banks, wires, exchanges), and optionally **Brave Search**.
2. Classifies the macro regime for 10 regions using a **Bayesian Markov model** (5 states per region).
3. Synthesizes an **8-section briefing** with directional views, trade ideas, and basis signals — deterministically, with zero LLM API calls.
4. Commits the result as JSON into `docs/data/`. **GitHub Pages** auto-publishes it.

A static SPA in `docs/` displays the briefing in a dark, sidebar-driven UI with thumbs-up/down feedback per section. Feedback is POSTed to a tiny **Cloudflare Worker** (free) which triggers a second workflow that appends to `data/feedback.json`. The next morning's run reads that file and feeds it into `MacroLLM.process_section_feedback()` — closing the learning loop.

**Everything is free, forever:** GitHub Actions (2,000 min/mo, generation takes <1 min), GitHub Pages (unlimited public bandwidth), Cloudflare Workers (100,000 req/day).

---

## Architecture

```
                          GitHub Actions cron (Mon–Fri ~10 UTC)
                                       │
                                       ▼
                              python generate.py
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
            FRED API            14 RSS feeds          Brave Search (optional)
                                       │
                                       ▼
                          MacroLLM.generate_daily_briefing()
                                       │
                                       ▼
                  docs/data/briefings/YYYY-MM-DD.json  (committed back to repo)
                                       │
                                       ▼
                            GitHub Pages  →  static SPA
                                                 │
                                                 ▼  (POST feedback)
                                       Cloudflare Worker
                                                 │
                                                 ▼  (repository_dispatch: feedback)
                              feedback-handler workflow
                                                 │
                                                 ▼
                            data/feedback.json   →   read by next briefing
```

---

## One-time setup

You'll do these once. After that the system runs itself.

### 1. Enable GitHub Pages

1. Push to `main`.
2. GitHub repo → **Settings** → **Pages** → **Source: Deploy from a branch** → branch `main`, folder `/docs`. Save.
3. After ~1 minute the site is live at `https://arjunparikh12.github.io/macro-briefing/`.

### 2. Deploy the Cloudflare Worker (for feedback)

You need a free Cloudflare account (no credit card required for Workers free tier).

```bash
cd worker/
npm install
npx wrangler login                            # one-time browser auth
npx wrangler secret put GITHUB_TOKEN          # paste a fine-grained PAT (see below)
npx wrangler deploy
```

The deploy prints a URL like `https://macro-feedback.<your-subdomain>.workers.dev`. Copy it into `docs/config.js`:

```js
window.MACRO_CONFIG = {
  FEEDBACK_WORKER_URL: "https://macro-feedback.<your-subdomain>.workers.dev",
  DATA_BASE: "data",
};
```

Commit + push. Done.

**Creating the GitHub PAT** (for the Worker secret):
GitHub → Settings → Developer settings → **Personal access tokens** → **Fine-grained tokens** → Generate new.
- Repository access: only `macro-briefing`
- Permissions: **Contents = Read and write** (this is the only scope needed for `repository_dispatch`)
- Expiration: whatever you're comfortable with (the longer you set it, the less often you re-roll)

### 3. (Optional) Add the Brave Search API key

The pipeline runs fine without Brave (RSS + FRED is plenty). If you want the extra search-augmented context:

1. Get a free key at [brave.com/search/api](https://brave.com/search/api/) (free tier: 2,000 queries/mo — plenty for daily runs).
2. GitHub repo → Settings → Secrets and variables → **Actions** → **New repository secret**.
3. Name: `BRAVE_API_KEY`. Value: your key.

Future runs automatically pick it up.

### 4. Trigger the first briefing manually

GitHub repo → Actions → **Daily Briefing** → **Run workflow** → optionally check "force". This generates today's briefing immediately so the site has something to show.

---

## Daily flow

- **~6 AM ET, Mon–Fri:** Action runs `generate.py`, commits `docs/data/briefings/YYYY-MM-DD.json` and `index.json`. Pages republishes within ~1 min.
- **You read the briefing**, click 👍 / 👎, optionally type a note, click **Send**.
- The Worker forwards it; the feedback workflow commits the entry to `data/feedback.json` (~5 sec end-to-end).
- **Next morning's run** reads that feedback, updates the regime model, and tailors the next briefing.

The model's persistent state (`data/macro_memory.json`, `data/macro_llm_learnings.json`) is also committed by the daily workflow, so learning compounds over time and is fully version-controlled.

---

## Future: opening it up to others

Phase 1 (current) is single-user, public-read, zero cost. To make it real-multi-user:

- **Auth:** Cloudflare Access (free, up to 50 users) in front of the Pages domain — sign in via Google/email magic link.
- **Per-user feedback:** the schema already includes `user_id`. The feedback workflow keys entries by user, so multi-user is a frontend change only.
- **Chat / uploads / other dynamic features:** add new routes to the same Worker. State goes to Cloudflare KV or D1 (free tiers). The MacroLLM Python engine stays in GitHub Actions; on-demand chat would either run as a fast cold-start on a Cloudflare Container or be invoked via `repository_dispatch` for cheap-but-slow async responses.

The read path (Action → JSON → CDN → browser) doesn't need to change. The write path (Worker → repo) is already in place. Multi-user is mostly auth + UI work.

---

## Repo layout

```
macro-briefing/
├── .github/workflows/
│   ├── daily-briefing.yml       # cron + commit
│   └── feedback-handler.yml     # repository_dispatch handler
├── docs/                        # GitHub Pages root
│   ├── index.html, app.js, styles.css, config.js
│   └── data/
│       ├── index.json           # list of briefings
│       └── briefings/*.json     # one per day
├── worker/                      # Cloudflare Worker (feedback proxy)
│   ├── src/index.js
│   ├── wrangler.toml
│   └── package.json
├── data/                        # model state + feedback (committed by workflows)
│   ├── feedback.json
│   ├── macro_memory.json
│   └── macro_llm_learnings.json
├── generate.py                  # CLI entry point — runs MacroLLM, writes JSON
├── briefing.py                  # thin wrapper around MacroLLM
├── macro_llm.py                 # the engine
├── regime_model.py              # Markov regime classifier
├── data_access.py               # JSON I/O helpers
├── daily_briefing_runner.py     # FRED + RSS + Brave data pipeline
└── requirements.txt
```

---

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate a briefing
python generate.py --force

# Serve the static site
cd docs && python3 -m http.server 5051
# Visit http://localhost:5051
```

Worker dev:
```bash
cd worker && npx wrangler dev
# Worker available at http://localhost:8787
# Set FEEDBACK_WORKER_URL in docs/config.js to "http://localhost:8787" for local end-to-end testing.
```

---

## Why this stack

| Concern              | Choice                | Why                                                                |
| -------------------- | --------------------- | ------------------------------------------------------------------ |
| Cron + compute       | GitHub Actions        | 2,000 free min/mo. Briefing runs in ~1 min. ~22 runs/mo = trivial. |
| Hosting              | GitHub Pages          | Free, unlimited bandwidth, global CDN, auto-deploys from `/docs`.  |
| Persistent state     | The repo itself       | Versioned, free, auditable. Workflows commit back.                 |
| Dynamic write proxy  | Cloudflare Worker     | 100k req/day free. ~10ms cold start. Keeps GH PAT off the browser. |
| LLM                  | None — `MacroLLM`     | Deterministic in-repo engine. Zero API spend.                      |

If any single piece becomes a bottleneck, each can be swapped without touching the others (e.g., move the briefing JSON to R2, swap Pages for Cloudflare Pages, swap the Worker for a Lambda).
