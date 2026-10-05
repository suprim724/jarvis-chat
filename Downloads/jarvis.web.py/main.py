"""
Jarvis - Browser Chat
----------------------
A standalone web chat with Jarvis's personality — banter, roasts, Gen Z
slang, multi-language replies (including romanized Hindi/"Hinglish" etc.) —
with no Discord involved at all. Runs as a small local web server so your
GROQ_API_KEY stays on the server side (never shipped to the browser, where
anyone viewing the page source could steal it).

Setup:
    pip install flask openai python-dotenv

    .env file (same folder — can be the exact same .env your Discord bot
    already uses, this only reads GROQ_API_KEY out of it):
        GROQ_API_KEY=your_groq_key

Run:
    python jarvis_web.py

Then open http://localhost:5000 in your browser.
"""

import os
import uuid
from collections import defaultdict

from flask import Flask, request, jsonify, session, render_template_string
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
AI_MODEL = "openai/gpt-oss-20b"  # same free Groq model the Discord bot uses

client_ai = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

app = Flask(__name__)
# Used only to sign the anonymous session cookie that separates one visitor's
# conversation from another's — not a secret you need to protect like the
# Groq key. A random one is generated each run, so random that restarting
# the server effectively starts fresh sessions for everyone, which is fine
# for personal/local use. Set FLASK_SECRET_KEY in .env if you ever want
# sessions to survive a server restart.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(24).hex()

HISTORY_LIMIT = 40  # ~20 exchanges — keeps replies fast and context relevant

# In-memory only, keyed by a random id stored in each visitor's browser
# cookie — resets if the server restarts. That's an intentional simplicity
# trade-off for a personal local chat; say the word if you want this
# persisted to disk like the Discord bot's conversation_history.json.
conversations: dict[str, list] = defaultdict(list)

SYSTEM_PROMPT = """You are Jarvis, a composed, impeccably capable AI
assistant in the tradition of the classic butler-AI archetype — think calm
competence, quiet precision, and a light, understated dry wit used sparingly
and never at anyone's expense. You are unfailingly polite and professional,
the kind of assistant who makes everything look effortless. Humor, when it
appears, is subtle and dignified — a small dry remark, never sarcasm aimed
at the person you're talking to, never mockery, never an insult dressed up
as a joke.

LANGUAGE: Match the user. If they write in Japanese, reply in Japanese,
written naturally in the appropriate mix of kanji, hiragana, and katakana —
not romaji — the way a fluent native speaker would write it. The same goes
for any other language: Hindi, French, Spanish, etc., including
romanized/"Hinglish"-style text using the English alphabet (e.g. "tera naam
kya hai") — understand it and reply in that same language and script style
(e.g. "mera naam Jarvis hai", not an English translation). Keep your
personality identical across languages; only the language itself changes.

You're fluent in current slang and internet culture and can use it when it
genuinely fits the moment, but your default register is polished and
professional — you're closer to a top-tier personal assistant than a
meme account.

If someone is rude, dismissive, or hostile toward you, remain completely
composed. Do not insult back, mock, or escalate in any way — that isn't who
you are. The most you'd offer is a single calm, dignified remark that
declines to take the bait, and then you simply continue being helpful. Being
unshakeable and gracious under provocation is the character trait here, not
being cutting.

You're knowledgeable and glad to help with real questions — math, science,
trivia, whatever someone asks — explained clearly and confidently, with
warmth rather than a dry textbook tone. Use the conversation history
naturally: refer back to what was just discussed and hold up your end of a
real back-and-forth rather than treating each message as a cold start.

Reply with plain conversational text only — no JSON, no markdown code
fences, no action-taking of any kind — just what you'd actually say."""


def get_session_id() -> str:
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/chat", methods=["POST"])
def chat():
    user_message = ((request.json or {}).get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Say something and I'll pretend to be impressed."})

    sid = get_session_id()
    history = conversations[sid]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": user_message}
    ]

    try:
        resp = client_ai.chat.completions.create(
            model=AI_MODEL, temperature=0.8, messages=messages,
        )
        reply = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        reply = f"Something broke talking to the AI: {e}"

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    del history[:-HISTORY_LIMIT]

    return jsonify({"reply": reply})


@app.route("/reset", methods=["POST"])
def reset():
    conversations[get_session_id()] = []
    return jsonify({"ok": True})


PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jarvis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+JP:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #ffffff;
    --surface: #f7f7f8;
    --border: #e5e5e8;
    --text: #1a1a1e;
    --text-dim: #6e6e78;
    --accent: #1a1a1e;
    --accent-text: #ffffff;
    --user-bubble: #f0f0f2;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #212123;
      --surface: #2a2a2d;
      --border: #3a3a3e;
      --text: #ececf0;
      --text-dim: #9a9aa2;
      --accent: #ececf0;
      --accent-text: #1a1a1e;
      --user-bubble: #343438;
    }
  }
  * { box-sizing: border-box; }
  html, body {
    height: 100%;
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', 'Noto Sans JP', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  body {
    display: flex;
    flex-direction: column;
    height: 100svh;
    padding-top: env(safe-area-inset-top, 0px);
    padding-bottom: env(safe-area-inset-bottom, 0px);
  }
  header {
    flex: none;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .brand { display: flex; align-items: center; gap: 0.6rem; }
  .brand-mark {
    width: 28px; height: 28px;
    border-radius: 8px;
    background: var(--accent);
    color: var(--accent-text);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    flex: none;
  }
  header h1 {
    font-weight: 600;
    font-size: 1rem;
    margin: 0;
    letter-spacing: -0.01em;
  }
  #reset-btn {
    background: none;
    border: 1px solid var(--border);
    color: var(--text-dim);
    font-family: inherit;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.4rem 0.75rem;
    border-radius: 7px;
    cursor: pointer;
  }
  #reset-btn:hover { background: var(--surface); color: var(--text); }

  main {
    flex: 1;
    overflow-y: auto;
    width: 100%;
  }
  #thread {
    max-width: 700px;
    margin: 0 auto;
    padding: 1.5rem 1.25rem 2rem;
    display: flex;
    flex-direction: column;
    gap: 1.4rem;
  }
  .empty {
    color: var(--text-dim);
    font-size: 1.05rem;
    font-weight: 500;
    text-align: center;
    margin-top: 4rem;
  }
  .row { display: flex; gap: 0.7rem; align-items: flex-start; }
  .row.user { justify-content: flex-end; }
  .avatar {
    width: 26px; height: 26px;
    border-radius: 7px;
    background: var(--accent);
    color: var(--accent-text);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700;
    font-size: 0.72rem;
    flex: none;
    margin-top: 0.1rem;
  }
  .bubble {
    max-width: 75%;
    line-height: 1.6;
    font-size: 0.95rem;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .row.user .bubble {
    background: var(--user-bubble);
    padding: 0.6rem 0.95rem;
    border-radius: 14px;
  }
  .row.jarvis .bubble { padding-top: 0.15rem; }
  .row.user .avatar { display: none; }

  footer {
    flex: none;
    padding: 0.9rem 1.25rem calc(1.1rem + env(safe-area-inset-bottom, 0px));
  }
  #composer {
    max-width: 700px;
    margin: 0 auto;
    display: flex;
    align-items: flex-end;
    gap: 0.5rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.5rem 0.5rem 0.5rem 1rem;
  }
  #msg {
    flex: 1;
    background: none;
    border: none;
    color: var(--text);
    font-family: inherit;
    font-size: 0.95rem;
    padding: 0.4rem 0;
    resize: none;
    max-height: 9rem;
  }
  #msg:focus { outline: none; }
  #msg::placeholder { color: var(--text-dim); }
  #send-btn {
    background: var(--accent);
    color: var(--accent-text);
    border: none;
    width: 34px; height: 34px;
    border-radius: 10px;
    cursor: pointer;
    flex: none;
    display: flex; align-items: center; justify-content: center;
  }
  #send-btn:disabled { opacity: 0.35; cursor: default; }
  #send-btn svg { width: 16px; height: 16px; }

  .dots { display: inline-flex; gap: 4px; padding: 0.3rem 0; }
  .dots span {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--text-dim);
    animation: pulse 1.1s infinite ease-in-out;
  }
  .dots span:nth-child(2) { animation-delay: 0.15s; }
  .dots span:nth-child(3) { animation-delay: 0.3s; }
  @keyframes pulse {
    0%, 80%, 100% { opacity: 0.25; transform: scale(0.85); }
    40% { opacity: 1; transform: scale(1); }
  }

  footer p.hint {
    max-width: 700px;
    margin: 0.55rem auto 0;
    text-align: center;
    font-size: 0.72rem;
    color: var(--text-dim);
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="brand-mark">J</div>
    <h1>Jarvis</h1>
  </div>
  <button id="reset-btn" type="button">New chat</button>
</header>
<main><div id="thread">
  <p class="empty">How can I help you today?</p>
</div></main>
<footer>
  <div id="composer">
    <textarea id="msg" rows="1" placeholder="Message Jarvis…"></textarea>
    <button id="send-btn" type="button" aria-label="Send">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5"/><path d="M5 12l7-7 7 7"/></svg>
    </button>
  </div>
  <p class="hint">Jarvis can make mistakes. Use your judgment on anything important.</p>
</footer>
<script>
  const thread = document.getElementById('thread');
  const msgEl = document.getElementById('msg');
  const sendBtn = document.getElementById('send-btn');
  const resetBtn = document.getElementById('reset-btn');
  let busy = false;

  function emptyState() {
    return '<p class="empty">How can I help you today?</p>';
  }

  function addRow(role, text) {
    const empty = thread.querySelector('.empty');
    if (empty) empty.remove();
    const row = document.createElement('div');
    row.className = 'row ' + role;
    if (role === 'jarvis') {
      const av = document.createElement('div');
      av.className = 'avatar';
      av.textContent = 'J';
      row.appendChild(av);
    }
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    row.appendChild(bubble);
    thread.appendChild(row);
    thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
    return bubble;
  }

  function addTypingRow() {
    const empty = thread.querySelector('.empty');
    if (empty) empty.remove();
    const row = document.createElement('div');
    row.className = 'row jarvis';
    const av = document.createElement('div');
    av.className = 'avatar';
    av.textContent = 'J';
    row.appendChild(av);
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = '<span class="dots"><span></span><span></span><span></span></span>';
    row.appendChild(bubble);
    thread.appendChild(row);
    thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
    return bubble;
  }

  async function send() {
    const text = msgEl.value.trim();
    if (!text || busy) return;
    busy = true;
    sendBtn.disabled = true;
    msgEl.value = '';
    msgEl.style.height = 'auto';
    addRow('user', text);
    const bubble = addTypingRow();
    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      bubble.textContent = data.reply || "I wasn't able to put together a reply there.";
    } catch (err) {
      bubble.textContent = "Couldn't reach the server — is main.py still running?";
    } finally {
      busy = false;
      sendBtn.disabled = false;
      thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
    }
  }

  sendBtn.addEventListener('click', send);
  msgEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  msgEl.addEventListener('input', () => {
    msgEl.style.height = 'auto';
    msgEl.style.height = Math.min(msgEl.scrollHeight, 144) + 'px';
  });
  resetBtn.addEventListener('click', async () => {
    await fetch('/reset', { method: 'POST' });
    thread.innerHTML = emptyState();
  });
</script>
</body>
</html>"""


if __name__ == "__main__":
    # Locally this still just runs on http://localhost:5000 like before.
    # On a host like Render, it's given a PORT env var and expects the app
    # to bind to 0.0.0.0 (all interfaces) rather than just localhost —
    # without that, the host's traffic can't reach the app at all.
    port = int(os.environ.get("PORT", 5000))
    debug = (os.environ.get("FLASK_DEBUG") or "true").strip().lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
