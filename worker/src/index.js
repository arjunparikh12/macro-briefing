// Macro Briefing — Feedback Worker
//
// Receives feedback POSTs from the static site, validates them, and triggers
// a GitHub repository_dispatch event of type "feedback". A workflow on the
// repo (.github/workflows/feedback-handler.yml) appends the entry to
// data/feedback.json so the next briefing run can learn from it.
//
// Required secrets (set with `wrangler secret put <NAME>`):
//   - GITHUB_TOKEN  → fine-grained PAT, repo-scoped, contents:write
//
// Required vars (set in wrangler.toml):
//   - GITHUB_OWNER  → "arjunparikh12"
//   - GITHUB_REPO   → "macro-briefing"
//   - ALLOWED_ORIGIN → "https://arjunparikh12.github.io" (or "*" while debugging)

const CORS_HEADERS = (origin) => ({
  "Access-Control-Allow-Origin":  origin,
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
  "Access-Control-Max-Age":       "86400",
  "Vary":                         "Origin",
});

function json(body, status, origin) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS(origin) },
  });
}

function pickOrigin(req, allowed) {
  const origin = req.headers.get("Origin") || "";
  if (allowed === "*" || !allowed) return origin || "*";
  const list = allowed.split(",").map((s) => s.trim());
  return list.includes(origin) ? origin : list[0];
}

export default {
  async fetch(request, env) {
    const origin = pickOrigin(request, env.ALLOWED_ORIGIN);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS(origin) });
    }

    if (request.method !== "POST") {
      return json({ error: "Method not allowed" }, 405, origin);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "Invalid JSON" }, 400, origin);
    }

    // Validate
    const briefingId = String(body.briefing_id || "").trim();
    if (!/^\d{4}-\d{2}-\d{2}$/.test(briefingId)) {
      return json({ error: "Invalid briefing_id" }, 400, origin);
    }
    const rating = body.rating === "up" || body.rating === "down" ? body.rating : null;
    const note = String(body.note || "").slice(0, 1000);
    const section = String(body.section || "").slice(0, 200);
    if (!rating && !note.trim()) {
      return json({ error: "Rating or note required" }, 400, origin);
    }
    const userId = String(body.user_id || "anon").slice(0, 64);
    const ts = String(body.ts || new Date().toISOString()).slice(0, 32);

    // Trigger GitHub repository_dispatch
    const url = `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/dispatches`;
    const ghResp = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type":  "application/json",
        "User-Agent":    "macro-briefing-feedback-worker",
      },
      body: JSON.stringify({
        event_type: "feedback",
        client_payload: {
          briefing_id: briefingId,
          section,
          rating,
          note,
          user_id: userId,
          ts,
        },
      }),
    });

    if (!ghResp.ok) {
      const text = await ghResp.text().catch(() => "");
      return json(
        { error: "GitHub dispatch failed", status: ghResp.status, detail: text.slice(0, 300) },
        502,
        origin,
      );
    }

    return json({ ok: true }, 200, origin);
  },
};
