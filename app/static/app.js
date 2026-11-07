(() => {
  const LABELS = [
    ["toxic", "Toxic"],
    ["severe_toxic", "Severe toxic"],
    ["obscene", "Obscene"],
    ["threat", "Threat"],
    ["insult", "Insult"],
    ["identity_hate", "Identity hate"],
  ];

  const $ = (id) => document.getElementById(id);
  const form = $("composer");
  const textarea = $("comment");
  const count = $("count");
  const analyzeBtn = $("analyze");
  const emptyEl = $("empty");
  const resultEl = $("result");
  const errorEl = $("error");
  const verdictEl = $("verdict");
  const metersEl = $("meters");
  const markedEl = $("marked");
  const termsEl = $("terms");
  const markedTitle = $("marked-title");
  const explainNote = $("explain-note");
  const meterNote = $("meter-note");

  let last = null; // { text, data }
  let selected = 0;

  const modelValue = () => form.querySelector('input[name="model"]:checked').value;

  function updateCount() {
    count.textContent = `${textarea.value.length} / 5000`;
  }

  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.hidden = false;
  }

  function pct(p) {
    return Math.round(p * 100);
  }

  function renderMeters(data) {
    metersEl.textContent = "";
    LABELS.forEach(([key, name], i) => {
      const p = data.scores[key];
      const thr = data.thresholds[key];
      const flagged = p >= thr;
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "meter" + (flagged ? " is-flagged" : "");
      btn.setAttribute("aria-pressed", String(i === selected));
      btn.setAttribute("aria-label", `${name}: ${pct(p)} percent. Show words behind this score.`);

      const label = document.createElement("span");
      label.className = "meter-name";
      label.textContent = name;

      const track = document.createElement("span");
      track.className = "track";
      const fill = document.createElement("span");
      fill.className = "fill";
      fill.style.width = "0%";
      const tick = document.createElement("span");
      tick.className = "tick";
      tick.style.left = `${thr * 100}%`;
      track.append(fill, tick);

      const val = document.createElement("span");
      val.className = "meter-pct";
      val.textContent = `${pct(p)}%`;

      btn.append(label, track, val);
      btn.addEventListener("click", () => {
        selected = i;
        metersEl.querySelectorAll(".meter").forEach((m, j) =>
          m.setAttribute("aria-pressed", String(j === selected))
        );
        renderMarked();
      });
      li.append(btn);
      metersEl.append(li);
      requestAnimationFrame(() => requestAnimationFrame(() => (fill.style.width = `${pct(p)}%`)));
    });
  }

  function renderMarked() {
    const { text, data } = last;
    const [key, name] = LABELS[selected];
    markedTitle.textContent = `Words behind the ${name.toLowerCase()} score`;

    const weights = data.words.map((w) => w.w[selected]);
    const max = Math.max(0, ...weights);
    markedEl.textContent = "";

    let cursor = 0;
    data.words.forEach((w, k) => {
      const weight = weights[k];
      if (max <= 0 || weight <= max * 0.04) return;
      if (w.start > cursor) markedEl.append(document.createTextNode(text.slice(cursor, w.start)));
      const mark = document.createElement("mark");
      mark.style.setProperty("--a", (0.22 + 0.7 * (weight / max)).toFixed(2));
      mark.textContent = text.slice(w.start, w.end);
      markedEl.append(mark);
      cursor = w.end;
    });
    if (cursor < text.length) markedEl.append(document.createTextNode(text.slice(cursor)));

    const terms = data.top_terms[key] || [];
    termsEl.textContent = "";
    if (terms.length) {
      const b = document.createElement("b");
      b.textContent = "Strongest terms: ";
      termsEl.append(b, document.createTextNode(terms.map((t) => t.term).join(", ")));
    } else {
      termsEl.textContent = `No words in this comment pushed the ${name.toLowerCase()} score up.`;
    }
  }

  function render(text, data) {
    last = { text, data };
    const top = LABELS.map(([k], i) => [data.scores[k], i]).sort((a, b) => b[0] - a[0])[0][1];
    selected = top;

    if (data.flagged.length) {
      const names = data.flagged.map((k) => LABELS.find(([key]) => key === k)[1].toLowerCase());
      verdictEl.textContent = `Flagged as ${names.join(", ")}.`;
      verdictEl.className = "verdict is-flagged";
    } else {
      verdictEl.textContent = "No category crosses its threshold.";
      verdictEl.className = "verdict is-clear";
    }

    explainNote.hidden = data.model === "lr";
    const same = LABELS.every(([k]) => data.thresholds[k] === 0.5);
    meterNote.textContent = (same
      ? "The tick on each bar is the 50% line."
      : "The tick on each bar is that category's decision threshold, tuned on validation data.")
      + " Select a category to see which words drove it.";
    renderMeters(data);
    renderMarked();
    emptyEl.hidden = true;
    errorEl.hidden = true;
    resultEl.hidden = false;
  }

  async function analyze() {
    const text = textarea.value.trim();
    if (!text) {
      showError("Enter a comment to analyze.");
      resultEl.hidden = true;
      emptyEl.hidden = true;
      textarea.focus();
      return;
    }
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing";
    errorEl.hidden = true;

    // Free hosting sleeps when idle, so the first request can be slow.
    const slowTimer = setTimeout(() => {
      showError("The server is waking up. This can take up to a minute on the first request.");
    }, 3000);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: modelValue() }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(typeof body.detail === "string" ? body.detail : "The server could not score this comment.");
      }
      render(text, await res.json());
    } catch (err) {
      resultEl.hidden = true;
      emptyEl.hidden = true;
      showError(err.message === "Failed to fetch" ? "Could not reach the server. Check your connection and try again." : err.message);
    } finally {
      clearTimeout(slowTimer);
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze comment";
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    analyze();
  });

  textarea.addEventListener("input", updateCount);
  textarea.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      analyze();
    }
  });

  document.querySelectorAll(".chip").forEach((chip) =>
    chip.addEventListener("click", () => {
      textarea.value = chip.dataset.text;
      updateCount();
      analyze();
    })
  );

  form.querySelectorAll('input[name="model"]').forEach((r) =>
    r.addEventListener("change", () => {
      if (textarea.value.trim() && last) analyze();
    })
  );

  $("clear").addEventListener("click", () => {
    textarea.value = "";
    updateCount();
    last = null;
    resultEl.hidden = true;
    errorEl.hidden = true;
    emptyEl.hidden = false;
    textarea.focus();
  });

  updateCount();

  // Show the BiLSTM option only when the server has its files.
  fetch("/api/health")
    .then((r) => r.json())
    .then((h) => {
      if (h.models && h.models.includes("bilstm")) $("seg-bilstm").hidden = false;
    })
    .catch(() => {});
})();
