import React, { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

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
    predicted_skills: ["threat language", "targeted aggression"],
    output: [{ label: "threat", confidence: 0.92 }],
    harmful: true,
    risk_level: "HIGH",
    timestamp: new Date().toISOString(),
  },
];

const CATEGORY_HELP = {
  hate: "Detects hateful speech targeting protected groups.",
  offense: "Detects offensive and abusive language.",
  bully: "Detects bullying patterns and harassment cues.",
  threat: "Detects explicit or implied threats of harm.",
};

const RISK_BADGE_CLASSES = {
  SAFE: "bg-emerald-500/20 text-emerald-300 border-emerald-400/40",
  LOW: "bg-cyan-500/20 text-cyan-300 border-cyan-400/40",
  MEDIUM: "bg-amber-500/20 text-amber-300 border-amber-400/40",
  HIGH: "bg-rose-500/20 text-rose-300 border-rose-400/40",
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

export default function SERMLApp() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [errorToast, setErrorToast] = useState("");
  const [historyOpen, setHistoryOpen] = useState(true);
  const [history, setHistory] = useState([]);
  const [copied, setCopied] = useState(false);
  const [safeBurst, setSafeBurst] = useState(false);
  const [harmfulFlash, setHarmfulFlash] = useState(false);

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
      return;
    }

    const timeout = setTimeout(() => setErrorToast(""), 3200);
    return () => clearTimeout(timeout);
  }, [errorToast]);

  const charCount = text.length;

  const chartData = useMemo(() => {
    if (!result?.output?.length) {
      return [];
    }
    return result.output.map((item) => ({
      category: item.label,
      confidence: Math.round(item.confidence * 100),
    }));
  }, [result]);

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
    setSafeBurst(false);
    setHarmfulFlash(false);

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

      if (data.harmful) {
        setHarmfulFlash(true);
        setTimeout(() => setHarmfulFlash(false), 600);
      } else {
        setSafeBurst(true);
        setTimeout(() => setSafeBurst(false), 1100);
      }
    } catch (error) {
      const fallback = getFallbackMock(clean);
      const reason = error instanceof Error ? error.message : "unknown error";
      setErrorToast(`Remote API unreachable: ${reason}. Showing mock response.`);
      setResult(fallback);
      pushHistory(fallback);

      if (fallback.harmful) {
        setHarmfulFlash(true);
        setTimeout(() => setHarmfulFlash(false), 600);
      } else {
        setSafeBurst(true);
        setTimeout(() => setSafeBurst(false), 1100);
      }
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
    <div className="min-h-screen bg-[#0a0a0f] text-slate-100 relative overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(124,58,237,0.18),transparent_40%),radial-gradient(circle_at_80%_10%,rgba(6,182,212,0.15),transparent_42%)]" />
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.04)_1px,transparent_1px)] bg-[size:36px_36px] opacity-20" />

      <div className={`absolute inset-0 transition ${harmfulFlash ? "bg-rose-500/20" : "bg-transparent"}`} />
      {safeBurst && (
        <div className="pointer-events-none absolute inset-0">
          {Array.from({ length: 16 }).map((_, i) => (
            <span
              key={i}
              className="absolute h-2 w-2 rounded-full bg-emerald-300/80 animate-ping"
              style={{
                left: `${8 + ((i * 13) % 84)}%`,
                top: `${12 + ((i * 17) % 76)}%`,
                animationDelay: `${(i % 6) * 80}ms`,
              }}
            />
          ))}
        </div>
      )}

      {errorToast && (
        <div className="fixed top-4 right-4 z-50 rounded-xl border border-rose-400/50 bg-rose-900/70 backdrop-blur px-4 py-2 text-sm shadow-xl transition-all">
          {errorToast}
        </div>
      )}

      <div className="relative z-10 mx-auto flex max-w-7xl gap-4 px-4 py-6 md:px-8">
        <aside
          className={`transition-all duration-300 ${historyOpen ? "w-72" : "w-12"} shrink-0 rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl`}
        >
          <div className="flex items-center justify-between p-3 border-b border-white/10">
            {historyOpen && <h2 className="text-sm tracking-wide text-slate-300">History</h2>}
            <button
              onClick={() => setHistoryOpen((v) => !v)}
              className="rounded-md px-2 py-1 text-xs text-slate-300 hover:bg-white/10 transition"
            >
              {historyOpen ? "Hide" : "Show"}
            </button>
          </div>

          {historyOpen && (
            <div className="space-y-2 p-3">
              {history.length === 0 && <p className="text-xs text-slate-500">No analyses yet.</p>}
              {history.map((item, idx) => (
                <div key={`${item.timestamp}-${idx}`} className="rounded-xl border border-white/10 bg-black/25 p-2">
                  <div className="flex items-center justify-between gap-2">
                    <span
                      className={`rounded-full border px-2 py-0.5 text-[10px] ${RISK_BADGE_CLASSES[item.risk_level] || RISK_BADGE_CLASSES.LOW}`}
                    >
                      {item.risk_level}
                    </span>
                    <span className="text-[10px] text-slate-400">{formatTime(item.timestamp)}</span>
                  </div>
                  <p className="mt-1 text-xs text-slate-300">
                    {item.text.length > 95 ? `${item.text.slice(0, 95)}...` : item.text}
                  </p>
                </div>
              ))}
            </div>
          )}
        </aside>

        <main className="flex-1 space-y-4">
          <header className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl p-5">
            <div className="flex items-center gap-3">
              <span className="relative flex h-3 w-3">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-75" />
                <span className="relative inline-flex h-3 w-3 rounded-full bg-violet-500" />
              </span>
              <h1 className="text-2xl font-semibold tracking-wide">SERML</h1>
            </div>
            <p className="mt-1 text-sm text-slate-400">Real-time Harmful Content Detection</p>
            <p className="mt-2 text-xs text-slate-500 break-all">
              Backend: {API_BASE_URL || "MISSING (set VITE_API_BASE_URL in .env and restart Vite)"}
            </p>
            {!API_BASE_URL && (
              <p className="mt-1 text-[11px] text-slate-600 break-all">
                Debug raw env value: {String(RAW_API_BASE_URL)}
              </p>
            )}
          </header>

          <section className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl p-5">
            <label className="text-sm text-slate-300">Input Text</label>
            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              rows={8}
              placeholder="Paste content to analyze for hate, offense, bullying, or threats..."
              className="mt-2 w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-400/20"
            />

            <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
              <span className="text-xs text-slate-400">{charCount.toLocaleString()} chars</span>
              <div className="flex items-center gap-2">
                <button onClick={onPaste} className="rounded-lg border border-white/15 px-3 py-1.5 text-xs hover:bg-white/10 transition">
                  Paste
                </button>
                <button onClick={onClear} className="rounded-lg border border-white/15 px-3 py-1.5 text-xs hover:bg-white/10 transition">
                  Clear
                </button>
                <button
                  onClick={analyze}
                  disabled={loading}
                  className={`relative overflow-hidden rounded-lg px-5 py-2 text-sm font-medium transition ${
                    loading
                      ? "bg-violet-700/80 text-slate-200"
                      : "bg-gradient-to-r from-violet-600 to-cyan-500 text-white hover:brightness-110"
                  }`}
                >
                  {loading && <span className="absolute inset-0 animate-pulse bg-gradient-to-r from-transparent via-white/25 to-transparent" />}
                  <span className="relative">{loading ? "Analyzing..." : "Analyze"}</span>
                </button>
              </div>
            </div>
            <p className="mt-2 text-xs text-slate-500">Shortcut: Cmd/Ctrl + Enter</p>
          </section>

          <section className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl p-5">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm text-slate-300">Pipeline Output</h2>
              <button
                onClick={copyResult}
                className="rounded-md border border-white/15 px-3 py-1 text-xs hover:bg-white/10 transition"
              >
                {copied ? "Copied" : "Copy JSON"}
              </button>
            </div>

            {loading ? (
              <div className="space-y-2">
                {Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="h-12 rounded-xl bg-white/10 relative overflow-hidden">
                    <span className="absolute inset-0 animate-pulse bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                  </div>
                ))}
              </div>
            ) : result ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className={`rounded-full border px-3 py-1 text-xs ${RISK_BADGE_CLASSES[result.risk_level] || RISK_BADGE_CLASSES.LOW}`}>
                    {result.risk_level}
                  </span>
                  <span className="text-xs text-slate-400">{result.harmful ? "HARMFUL" : "SAFE"}</span>
                  <span className="text-xs text-slate-500">{formatTime(result.timestamp)}</span>
                </div>

                <div className="rounded-xl border border-white/10 bg-black/30 p-3">
                  <p className="text-xs text-slate-400">Predicted Skills</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {(result.predicted_skills || []).length === 0 && <span className="text-xs text-slate-500">None</span>}
                    {(result.predicted_skills || []).map((skill) => (
                      <span key={skill} className="rounded-full border border-cyan-400/30 bg-cyan-400/10 px-2 py-1 text-xs text-cyan-200">
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="rounded-xl border border-white/10 bg-black/30 p-3">
                  <p className="text-xs text-slate-400">Detected Categories</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {(result.output || []).length === 0 && <span className="text-xs text-slate-500">No harmful categories detected.</span>}
                    {(result.output || []).map((item) => (
                      <span
                        key={`${item.label}-${item.confidence}`}
                        title={CATEGORY_HELP[item.label] || "Detected by expert adapter."}
                        className="rounded-lg border border-violet-400/30 bg-violet-400/10 px-2.5 py-1 text-xs text-violet-200"
                      >
                        {item.label} ({Math.round(item.confidence * 100)}%)
                      </span>
                    ))}
                  </div>
                </div>

                <div className="h-56 rounded-xl border border-white/10 bg-black/20 p-2">
                  {chartData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} margin={{ top: 12, right: 8, left: -12, bottom: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                        <XAxis dataKey="category" tick={{ fill: "#cbd5e1", fontSize: 12 }} />
                        <YAxis tick={{ fill: "#cbd5e1", fontSize: 12 }} domain={[0, 100]} />
                        <Tooltip
                          contentStyle={{
                            background: "rgba(2, 6, 23, 0.92)",
                            border: "1px solid rgba(148, 163, 184, 0.3)",
                            borderRadius: "10px",
                          }}
                        />
                        <Bar dataKey="confidence" fill="#7c3aed" radius={[8, 8, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex h-full items-center justify-center text-sm text-slate-500">No category confidences to visualize.</div>
                  )}
                </div>

                <pre className="max-h-72 overflow-auto rounded-xl border border-white/10 bg-black/40 p-3 text-xs text-slate-200">
{JSON.stringify(result, null, 2)}
                </pre>
              </div>
            ) : (
              <p className="text-sm text-slate-500">Run analysis to view results.</p>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
