// ─── Tab Navigation ─────────────────────────────────────────
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document
      .querySelectorAll(".tab")
      .forEach((t) => t.classList.remove("active"));
    document
      .querySelectorAll(".tab-content")
      .forEach((c) => c.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("tab-" + tab.dataset.tab).classList.add("active");
  });
});

// ─── Helpers ────────────────────────────────────────────────
function showLoading(el) {
  el.style.display = "block";
  el.innerHTML =
    '<h3>Result</h3><div class="loading"><span class="spinner"></span> Processing...</div>';
}
function showError(el, msg) {
  el.style.display = "block";
  el.innerHTML = `<h3>Result</h3><div class="error-msg">❌ ${msg}<br><small>Make sure the server is running.</small></div>`;
}

async function checkConnection(url) {
  const dot = document.querySelector(".status-dot");
  const txt = document.querySelector(".status-text");
  try {
    const r = await fetch(url + "/");
    if (r.ok) {
      dot.classList.add("connected");
      txt.textContent = "Connected";
      return true;
    }
  } catch (e) {}
  dot.classList.remove("connected");
  txt.textContent = "Disconnected";
  return false;
}

// ─── Sentiment Analysis ────────────────────────────────────
const quickTexts = {
  positive: "I absolutely love this product! It's amazing and wonderful!",
  negative:
    "This is the worst experience I've ever had. Terrible and horrible.",
  neutral: "The meeting is scheduled for 3 PM in the conference room.",
};

function quickSentiment(type) {
  document.getElementById("sentimentText").value = quickTexts[type];
  analyzeSentiment();
}

async function analyzeSentiment() {
  const url = document.getElementById("sentimentUrl").value;
  const text = document.getElementById("sentimentText").value;
  const el = document.getElementById("sentimentResult");
  showLoading(el);
  await checkConnection(url);
  try {
    const r = await fetch(url + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const d = await r.json();
    const cls =
      d.prediction === "positive"
        ? "positive"
        : d.prediction === "negative"
          ? "negative"
          : "neutral";
    const emoji = cls === "positive" ? "😊" : cls === "negative" ? "😠" : "😐";
    const color =
      cls === "positive"
        ? "var(--green)"
        : cls === "negative"
          ? "var(--red)"
          : "var(--yellow)";
    el.innerHTML = `<h3>Result</h3>
            <div style="text-align:center;margin:1rem 0"><span class="sentiment-badge sentiment-${cls}">${emoji} ${d.prediction.toUpperCase()}</span></div>
            <div class="confidence-bar"><div class="confidence-fill" style="width:${d.confidence * 100}%;background:${color}"></div></div>
            <div class="result-row"><span class="result-label">Confidence</span><span class="result-value">${(d.confidence * 100).toFixed(1)}%</span></div>
            <div class="result-row"><span class="result-label">Latency</span><span class="result-value">${d.latency_ms} ms</span></div>
            <div class="result-row"><span class="result-label">Input</span><span class="result-value" style="max-width:60%;text-align:right">${d.text.substring(0, 80)}</span></div>`;
  } catch (e) {
    showError(el, e.message);
  }
}

// ─── LLM Generation ────────────────────────────────────────
async function generateLLM() {
  const url = document.getElementById("llmUrl").value;
  const prompt = document.getElementById("llmPrompt").value;
  const max_tokens = parseInt(document.getElementById("maxTokens").value);
  const temperature =
    parseInt(document.getElementById("temperature").value) / 10;
  const el = document.getElementById("llmResult");
  showLoading(el);
  await checkConnection(url);
  try {
    const r = await fetch(url + "/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_tokens, temperature }),
    });
    const d = await r.json();
    el.innerHTML = `<h3>🤖 Response</h3>
            <div class="llm-response">${d.response}</div>
            <div class="result-row"><span class="result-label">Tokens Generated</span><span class="result-value">${d.tokens_generated}</span></div>
            <div class="result-row"><span class="result-label">Latency</span><span class="result-value">${d.latency_ms.toFixed(0)} ms</span></div>
            <div class="result-row"><span class="result-label">Speed</span><span class="result-value">${d.tokens_per_second} tok/s</span></div>`;
  } catch (e) {
    showError(el, e.message);
  }
}

async function compareTemps() {
  const url = document.getElementById("llmUrl").value;
  const prompt = document.getElementById("llmPrompt").value;
  const max_tokens = parseInt(document.getElementById("maxTokens").value);
  const el = document.getElementById("llmResult");
  showLoading(el);
  try {
    const r = await fetch(
      url +
        "/compare-temperatures?prompt=" +
        encodeURIComponent(prompt) +
        "&max_tokens=" +
        max_tokens,
      { method: "POST" },
    );
    const d = await r.json();
    let html = `<h3>🌡️ Temperature Comparison</h3><p style="color:var(--text-dim);margin-bottom:1rem">Prompt: "${d.prompt}"</p>`;
    d.comparisons.forEach((c) => {
      html += `<div style="margin-bottom:1.25rem"><div style="font-size:.82rem;color:var(--text-dim);margin-bottom:.35rem">🌡️ ${c.temperature} — ${c.label} (${c.latency_ms.toFixed(0)}ms)</div><div class="llm-response">${c.response}</div></div>`;
    });
    el.innerHTML = html;
    el.style.display = "block";
  } catch (e) {
    showError(el, e.message);
  }
}

// ─── Batch Test ─────────────────────────────────────────────
async function runBatchTest() {
  const url = document.getElementById("batchUrl").value;
  const count = parseInt(document.getElementById("batchCount").value);
  const el = document.getElementById("batchResult");
  showLoading(el);
  await checkConnection(url);
  const items = Array.from({ length: count }, (_, i) => ({
    text: `Sample review number ${i + 1} with great content`,
  }));
  try {
    // Single requests
    const t1 = performance.now();
    for (const item of items) {
      await fetch(url + "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(item),
      });
    }
    const singleMs = performance.now() - t1;
    // Batch request
    const t2 = performance.now();
    await fetch(url + "/predict/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(items),
    });
    const batchMs = performance.now() - t2;
    const speedup = (singleMs / batchMs).toFixed(1);
    const maxH = 100;
    const singleH = maxH;
    const batchH = Math.max(4, (batchMs / singleMs) * maxH);
    el.innerHTML = `<h3>⚡ Latency Comparison (${count} items)</h3>
            <div class="comparison-bars">
                <div class="comp-item"><h4>Individual Requests</h4><div class="comp-bar-wrap"><div class="comp-bar single" style="height:${singleH}px"></div></div><div class="comp-value" style="color:var(--red)">${singleMs.toFixed(0)} ms</div></div>
                <div class="comp-item"><h4>Batch Request</h4><div class="comp-bar-wrap"><div class="comp-bar batch" style="height:${batchH}px"></div></div><div class="comp-value" style="color:var(--green)">${batchMs.toFixed(0)} ms</div></div>
            </div>
            <div class="comp-speedup">Batch is <strong>${speedup}x faster</strong> than individual requests!</div>
            <div style="margin-top:1rem;padding:1rem;background:var(--bg);border-radius:var(--radius-sm);font-size:.85rem;color:var(--text-dim)">💡 <strong>Why?</strong> Each HTTP request has overhead (connection, serialization). Batching amortizes this cost. In real ML, GPUs also process batches in parallel via vectorized computation.</div>`;
  } catch (e) {
    showError(el, e.message);
  }
}
