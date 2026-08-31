import React, { useEffect, useState } from "react";

const RAW_API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
const API_BASE_URL = (RAW_API_BASE_URL || "").replace(/\/+$/, "");

const MOCK_RESPONSES = [
  {
    post: "This is a harmless status update about my weekend plans.",
    predicted_skills: ["neutral", "contextual"],
    output: [],
    harmful: false,
    risk_level: "SAFE",
    timestamp: new Date().toISOString(),
  },
  {
    post: "I will find you and make you pay.",
    predicted_skills: ["threat_language", "targeted_aggression"],
    output: [{ label: "threat", confidence: 0.92 }],
    harmful: true,
    risk_level: "HIGH",
    timestamp: new Date().toISOString(),
  },
];

const VERDICT_BADGE_CLASSES = {
  safe: "border-emerald-200 bg-emerald-50 text-emerald-800",
  harmful: "border-rose-200 bg-rose-50 text-rose-800",
};

const VERDICT_PANEL_CLASSES = {
  safe: "border-emerald-200 bg-emerald-50/70",
  harmful: "border-rose-200 bg-rose-50/70",
};

const HISTORY_KEY = "serml.history.v1";

function getFallbackMock(text) {
  const seed = text.length % MOCK_RESPONSES.length;
  return { ...MOCK_RESPONSES[seed], post: text, timestamp: new Date().toISOString() };
}

function formatTime(value) {
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function truncateText(value, limit = 108) {
  if (!value) {
    return "";
  }
  return value.length > limit ? `${value.slice(0, limit)}...` : value;
}

export default function SERMLApp() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [errorToast, setErrorToast] = useState("");
  const [historyOpen, setHistoryOpen] = useState(true);
  const [history, setHistory] = useState([]);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          setHistory(parsed.slice(0, 5));
        }
      }
    } catch {
      setHistory([]);
    }
  }, []);

  useEffect(() => {
    const onKeyDown = (event) => {
      const isMetaSubmit = (event.metaKey || event.ctrlKey) && event.key === "Enter";
      if (isMetaSubmit) {
        event.preventDefault();
        if (!loading) {
          void analyze();
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [loading, text]);

  useEffect(() => {
    if (!errorToast) {
      return undefined;
    }

    const timeout = setTimeout(() => setErrorToast(""), 3200);
    return () => clearTimeout(timeout);
  }, [errorToast]);

  const charCount = text.length;
  const categories = [...(result?.output || [])].sort((a, b) => b.confidence - a.confidence);
  const skills = result?.predicted_skills || [];
  const verdictTone = result?.harmful ? "harmful" : "safe";
  const topCategory = categories[0]?.label || "None";
  const summaryCards = result
    ? [
        {
          label: "Decision",
          value: result.harmful ? "Harmful" : "Safe",
          detail: "Final pipeline verdict",
        },
        {
          label: "Top Category",
          value: topCategory,
          detail: categories[0] ? formatPercent(categories[0].confidence) : "No harmful label",
        },
        {
          label: "Predicted Skills",
          value: String(skills.length),
          detail: skills.length ? "Routing signals retained" : "No expert skills triggered",
        },
        {
          label: "Expert Outputs",
          value: String(categories.length),
          detail: categories.length ? "Adapters returned labels" : "No harmful category output",
        },
      ]
    : [];

  const pushHistory = (res) => {
    setHistory((prev) => {
      const next = [
        {
          timestamp: res.timestamp,
          risk_level: res.risk_level,
          harmful: res.harmful,
          text: res.post,
        },
        ...prev,
      ].slice(0, 5);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
      return next;
    });
  };

  const analyze = async () => {
    const clean = text.trim();
    if (!clean) {
      setErrorToast("Enter text before analysis.");
      return;
    }

    setLoading(true);

    try {
      if (!API_BASE_URL) {
        throw new Error("VITE_API_BASE_URL is missing");
      }

      const endpoint = `${API_BASE_URL}/api/detect`;
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "1",
        },
        body: JSON.stringify({ text: clean }),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`API ${response.status}: ${detail.slice(0, 200)}`);
      }

      const data = await response.json();
      setResult(data);
      pushHistory(data);
    } catch (error) {
      const fallback = getFallbackMock(clean);
      const reason = error instanceof Error ? error.message : "unknown error";
      setErrorToast(`Remote API unreachable: ${reason}. Showing mock response.`);
      setResult(fallback);
      pushHistory(fallback);
    } finally {
      setLoading(false);
    }
  };

  const onPaste = async () => {
    try {
      const pasted = await navigator.clipboard.readText();
      setText((prev) => (prev ? `${prev}\n${pasted}` : pasted));
    } catch {
      setErrorToast("Clipboard access was denied.");
    }
  };

  const onClear = () => {
    setText("");
    setResult(null);
    setCopied(false);
  };

  const copyResult = async () => {
    if (!result) {
      return;
    }

    try {
      await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      setErrorToast("Unable to copy results.");
    }
  };

  return (
    <div className="min-h-screen bg-[#f5efe4] text-slate-900">
      {errorToast && (
        <div className="fixed right-4 top-4 z-50 rounded-2xl border border-amber-200 bg-[#fff7ea] px-4 py-3 text-sm text-amber-900 shadow-lg">
          {errorToast}
        </div>
      )}

      <header className="border-b border-[#ddd2bf] bg-[#fffaf0]/95 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 py-6 md:px-8">
          <div className="flex flex-col gap-4">
            <div className="max-w-4xl">
              <h1 className="text-2xl font-semibold tracking-tight text-[#1f2937] md:text-[2rem]">
                Skill-Based LLM Driven Expert Routing Pipeline
              </h1>
              <p className="mt-2 text-sm text-slate-600">
                Multilabel Harmful Speech Detection Pipeline for Social Media Post
              </p>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 py-6 md:px-8">
        <div className="grid gap-6 xl:grid-cols-[280px_minmax(0,1fr)]">
          <aside className="rounded-[28px] border border-[#ddd2bf] bg-[#fffaf3] shadow-[0_18px_45px_rgba(99,85,63,0.08)]">
            <div className="flex items-center justify-between border-b border-[#ebe1d0] px-5 py-4">
              {historyOpen && <h2 className="text-sm font-semibold tracking-wide text-slate-700">Recent Runs</h2>}
              <button
                onClick={() => setHistoryOpen((value) => !value)}
                className="rounded-full border border-[#d7cbb7] px-3 py-1 text-xs text-slate-600 transition hover:bg-[#f7efdf]"
              >
                {historyOpen ? "Hide" : "Show"}
              </button>
            </div>

            {historyOpen && (
              <div className="space-y-3 p-4">
                {history.length === 0 && <p className="text-xs text-slate-500">No analyses yet.</p>}
                {history.map((item, idx) => (
                  <div key={`${item.timestamp}-${idx}`} className="rounded-2xl border border-[#ebe1d0] bg-white p-3 shadow-sm">
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] ${
                          VERDICT_BADGE_CLASSES[item.harmful ? "harmful" : "safe"] || VERDICT_BADGE_CLASSES.safe
                        }`}
                      >
                        {item.harmful ? "Harmful" : "Safe"}
                      </span>
                      <span className="text-[10px] text-slate-500">{formatTime(item.timestamp)}</span>
                    </div>
                    <p className="mt-3 text-sm leading-6 text-slate-700">{truncateText(item.text)}</p>
                  </div>
                ))}
              </div>
            )}
          </aside>

          <main className="space-y-6">
            <section className="rounded-[30px] border border-[#ddd2bf] bg-[#fffaf3] p-6 shadow-[0_18px_45px_rgba(99,85,63,0.08)]">
              <div className="flex flex-col gap-6 xl:flex-row xl:items-start xl:justify-between">
                <div className="max-w-2xl">
                  <p className="text-sm font-semibold text-slate-700">Input Snippet</p>
                  <p className="mt-2 text-sm leading-6 text-slate-600">Compact input keeps the routed result in focus.</p>
                </div>
                <div className="rounded-2xl border border-[#e7dccb] bg-white px-4 py-3 text-xs leading-5 text-slate-500">
                  Shorter samples make the routing summary easier to scan.
                </div>
              </div>

              <textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                rows={4}
                placeholder="Paste content to analyze for hate, offense, bullying, or threats..."
                className="mt-5 min-h-[132px] w-full rounded-[24px] border border-[#d7cbb7] bg-white px-5 py-4 text-[15px] leading-7 text-slate-800 outline-none transition focus:border-[#8ba9a3] focus:ring-2 focus:ring-[#dbece8]"
              />

              <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500">
                  <span>{charCount.toLocaleString()} chars</span>
                  <span>Shortcut: Cmd/Ctrl + Enter</span>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={onPaste}
                    className="rounded-full border border-[#d7cbb7] bg-white px-4 py-2 text-sm text-slate-700 transition hover:bg-[#f7efdf]"
                  >
                    Paste
                  </button>
                  <button
                    onClick={onClear}
                    className="rounded-full border border-[#d7cbb7] bg-white px-4 py-2 text-sm text-slate-700 transition hover:bg-[#f7efdf]"
                  >
                    Clear
                  </button>
                  <button
                    onClick={analyze}
                    disabled={loading}
                    className={`rounded-full px-5 py-2 text-sm font-semibold text-white transition ${
                      loading ? "bg-slate-500" : "bg-[#2f6f73] hover:bg-[#275d61]"
                    }`}
                  >
                    {loading ? "Analyzing..." : "Analyze"}
                  </button>
                </div>
              </div>
            </section>

            <section className="rounded-[30px] border border-[#ddd2bf] bg-[#fffaf3] p-6 shadow-[0_18px_45px_rgba(99,85,63,0.08)]">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-slate-800">Pipeline Output</h2>
                  <p className="mt-1 text-sm text-slate-600">Routing details appear first, with raw JSON kept secondary.</p>
                </div>
                <button
                  onClick={copyResult}
                  className="rounded-full border border-[#d7cbb7] bg-white px-4 py-2 text-sm text-slate-700 transition hover:bg-[#f7efdf]"
                >
                  {copied ? "Copied" : "Copy JSON"}
                </button>
              </div>

              {loading ? (
                <div className="mt-5 grid gap-3 md:grid-cols-4">
                  {Array.from({ length: 4 }).map((_, idx) => (
                    <div key={idx} className="h-24 animate-pulse rounded-[24px] border border-[#ebe1d0] bg-white/80" />
                  ))}
                </div>
              ) : result ? (
                <div className="mt-6 space-y-6">
                  <div
                    className={`rounded-[28px] border px-5 py-5 ${
                      VERDICT_PANEL_CLASSES[verdictTone] || VERDICT_PANEL_CLASSES.safe
                    }`}
                  >
                    <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                      <div className="flex flex-wrap items-center gap-3">
                        <span
                          className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] ${
                            VERDICT_BADGE_CLASSES[verdictTone] || VERDICT_BADGE_CLASSES.safe
                          }`}
                        >
                          {result.harmful ? "Harmful" : "Safe"}
                        </span>
                        <span className="text-sm font-medium text-slate-700">{result.harmful ? "Harmful content detected" : "No harmful content detected"}</span>
                      </div>
                      <span className="text-xs text-slate-500">{formatTime(result.timestamp)}</span>
                    </div>

                    <div className="mt-4 grid gap-3 md:grid-cols-4">
                      {summaryCards.map((card) => (
                        <div key={card.label} className="rounded-[22px] border border-white/70 bg-white/80 p-4">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">{card.label}</p>
                          <p className="mt-2 text-lg font-semibold text-slate-800">{card.value}</p>
                          <p className="mt-1 text-xs text-slate-500">{card.detail}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="grid gap-6 xl:grid-cols-[1.08fr_0.92fr]">
                    <section className="rounded-[28px] border border-[#e6dccb] bg-white p-5">
                      <h3 className="text-base font-semibold text-slate-800">Routing Snapshot</h3>

                      <div className="mt-4 rounded-[22px] border border-[#ebe1d0] bg-[#fcf8f2] p-4">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Analyzed Snippet</p>
                        <p className="mt-3 border-l-4 border-[#cbbda5] pl-4 text-[15px] leading-7 text-slate-700">{result.post}</p>
                      </div>

                      <div className="mt-5 space-y-4">
                        <div className="flex gap-4">
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#edf5f3] text-sm font-semibold text-[#2f6f73]">
                            1
                          </div>
                          <div>
                            <p className="text-sm font-semibold text-slate-800">Skill prediction</p>
                            <div className="mt-2 flex flex-wrap gap-2">
                              {skills.length === 0 && <span className="text-sm text-slate-500">No routing skills retained.</span>}
                              {skills.map((skill) => (
                                <span
                                  key={skill}
                                  className="rounded-full border border-[#c7ddd7] bg-[#edf5f3] px-3 py-1 text-xs font-medium text-[#255c60]"
                                >
                                  {skill}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>

                        <div className="flex gap-4">
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#f8efe4] text-sm font-semibold text-[#9a6842]">
                            2
                          </div>
                          <div>
                            <p className="text-sm font-semibold text-slate-800">Expert responses</p>
                            <p className="mt-1 text-sm leading-6 text-slate-600">
                              {categories.length
                                ? `The activated experts produced ${categories.length} category prediction${categories.length > 1 ? "s" : ""}.`
                                : "No harmful-category expert output was returned for this sample."}
                            </p>
                          </div>
                        </div>

                        <div className="flex gap-4">
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#fdf0ed] text-sm font-semibold text-[#a95a54]">
                            3
                          </div>
                          <div>
                            <p className="text-sm font-semibold text-slate-800">Final decision</p>
                            <div className="mt-2 flex flex-wrap gap-2">
                              {categories.length === 0 && <span className="text-sm text-slate-500">No labels returned.</span>}
                              {categories.map((item) => (
                                <span
                                  key={`decision-${item.label}-${item.confidence}`}
                                  className="rounded-full border border-[#e3d2be] bg-[#fcf3e7] px-3 py-1 text-xs font-medium text-[#8a5b34]"
                                >
                                  {item.label}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    </section>

                    <section className="rounded-[28px] border border-[#e6dccb] bg-white p-5">
                      <h3 className="text-base font-semibold text-slate-800">Category Confidence</h3>
                      <p className="mt-1 text-sm text-slate-600">Horizontal bars make confidence easier to scan at a glance.</p>

                      <div className="mt-5 space-y-4">
                        {categories.length === 0 && (
                          <div className="rounded-[22px] border border-dashed border-[#d7cbb7] bg-[#fcf8f2] p-5 text-sm text-slate-500">
                            No harmful categories detected for this example.
                          </div>
                        )}

                        {categories.map((item) => (
                          <div key={`${item.label}-${item.confidence}`} className="rounded-[22px] border border-[#ebe1d0] bg-[#fcf8f2] p-4">
                            <div className="flex items-center justify-between gap-3">
                              <p className="text-sm font-semibold capitalize text-slate-800">{item.label}</p>
                              <span className="text-sm font-semibold text-[#2f6f73]">{formatPercent(item.confidence)}</span>
                            </div>
                            <div className="mt-3 h-2.5 rounded-full bg-[#e6e1d5]">
                              <div
                                className="h-2.5 rounded-full bg-[#2f6f73]"
                                style={{ width: `${Math.max(8, Math.round(item.confidence * 100))}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </section>
                  </div>

                  <details className="rounded-[24px] border border-[#e6dccb] bg-white p-5">
                    <summary className="cursor-pointer text-sm font-semibold text-slate-700">Raw JSON</summary>
                    <pre className="mt-4 max-h-80 overflow-auto rounded-[20px] border border-[#ebe1d0] bg-[#fcf8f2] p-4 text-xs leading-6 text-slate-700">
{JSON.stringify(result, null, 2)}
                    </pre>
                  </details>
                </div>
              ) : (
                <div className="mt-6 rounded-[28px] border border-dashed border-[#d7cbb7] bg-white p-8">
                  <p className="text-base font-semibold text-slate-800">Awaiting analysis</p>
                  <p className="mt-2 max-w-3xl text-sm leading-7 text-slate-600">Run an analysis to populate the routing summary and output labels.</p>
                </div>
              )}
            </section>
          </main>
        </div>

        <footer className="mt-8 rounded-[28px] border border-[#ddd2bf] bg-[#fffaf3] p-6 shadow-[0_18px_45px_rgba(99,85,63,0.08)]">
          <p className="text-sm font-semibold tracking-wide text-slate-800">Contact</p>
          <div className="mt-4 grid gap-3 text-sm text-slate-600 md:grid-cols-2">
            <p>
              <span className="font-semibold text-slate-800">Mudit Baid</span>
            </p>
            <p>
              <span className="font-semibold text-slate-800">Email:</span>{" "}
              <a className="text-[#2f6f73] hover:underline" href="mailto:muditb0712@gmail.com">
                muditb0712@gmail.com
              </a>
            </p>
            <p>
              <span className="font-semibold text-slate-800">Project Page:</span>{" "}
              <a
                className="text-[#2f6f73] hover:underline"
                href="https://muditbaid.github.io/Skill-Based-Expert-Routing-System-For-Multilabel-Hate-Speech-Detection/"
                target="_blank"
                rel="noreferrer"
              >
                View project page
              </a>
            </p>
            <p>
              <span className="font-semibold text-slate-800">GitHub:</span>{" "}
              <a
                className="text-[#2f6f73] hover:underline"
                href="https://github.com/muditbaid/Skill-Based-Expert-Routing-System-For-Multilabel-Hate-Speech-Detection"
                target="_blank"
                rel="noreferrer"
              >
                View repository
              </a>
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}