"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";

import DebugPanel from "../components/DebugPanel";

const PdfViewer = dynamic(() => import("../components/PdfViewer"), {
  ssr: false,
});

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
const DEMO_PROCESSED_FILE_PATH =
  "backend/storage/pipelines/99SMART-Annual-Report-2024/processed_99SMART-Annual-Report-2024.json";

function buildReportUrl(processedFilePath) {
  const url = new URL(`${API_BASE_URL}/api/v1/documents/report`);
  url.searchParams.set("processed_file_path", processedFilePath);
  return url.toString();
}

function formatAnswer(answer) {
  return answer || "No answer returned.";
}

export default function HomePage() {
  const [report, setReport] = useState(null);
  const [sessionId, setSessionId] = useState("demo-session-001");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [activeCitationPage, setActiveCitationPage] = useState(1);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [debugOpen, setDebugOpen] = useState(true);

  useEffect(() => {
    async function loadReport() {
      try {
        const response = await fetch(buildReportUrl(DEMO_PROCESSED_FILE_PATH));
        if (!response.ok) {
          throw new Error("Unable to load demo report metadata.");
        }
        const payload = await response.json();
        setReport(payload);
      } catch (loadError) {
        setError(loadError.message || "Unable to load report.");
      }
    }

    loadReport();
  }, []);

  const latestAssistantMessage = useMemo(() => {
    return [...messages].reverse().find((message) => message.role === "assistant");
  }, [messages]);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!question.trim() || pending || !report) {
      return;
    }

    const currentQuestion = question.trim();
    setError("");
    setPending(true);
    setQuestion("");
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        text: currentQuestion,
      },
    ]);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/agent/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          processed_file_path: report.processed_file_path,
          question: currentQuestion,
          top_k: 8,
        }),
      });

      const payload = await response.json();
      if (!response.ok || payload.status !== "success") {
        throw new Error(payload.error || "Unable to get response from FinScout.");
      }

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: payload.answer,
          payload,
        },
      ]);

      const firstCitationPage = payload.citations?.[0]?.page_number;
      if (firstCitationPage) {
        setActiveCitationPage(firstCitationPage);
      }
    } catch (requestError) {
      const message =
        requestError.message || "Unable to reach the assistant right now.";
      setError(message);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: message,
          payload: null,
        },
      ]);
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="appShell">
      <section className="workspace">
        <header className="topBar">
          <div>
            <p className="eyebrow">FinScout</p>
            <h1>Annual Report QA</h1>
          </div>
          <div className="reportMeta">
            <div>
              <span>Company</span>
              <strong>{report?.company_name || "Loading…"}</strong>
            </div>
            <div>
              <span>Year</span>
              <strong>{report?.year || "—"}</strong>
            </div>
            <label className="sessionField">
              <span>Session</span>
              <input
                type="text"
                value={sessionId}
                onChange={(event) => setSessionId(event.target.value)}
              />
            </label>
          </div>
        </header>

        <section className="chatPanel">
          <div className="messageStream">
            {messages.length === 0 ? (
              <div className="emptyState">
                <p className="eyebrow">Demo Report</p>
                <h2>{report?.title || "99SMART Annual Report 2024"}</h2>
                <p>
                  Ask about performance, management discussion, audit opinion, or
                  financial statements. Click any citation to jump the PDF viewer
                  to that page.
                </p>
              </div>
            ) : null}

            {messages.map((message) => (
              <article
                key={message.id}
                className={`message ${message.role === "user" ? "user" : "assistant"}`}
              >
                <div className="messageLabel">
                  {message.role === "user" ? "You" : "FinScout"}
                </div>
                <div className="messageBody">
                  <p>{formatAnswer(message.text)}</p>

                  {message.role === "assistant" && message.payload ? (
                    <>
                      {message.payload.citations?.length ? (
                        <section className="citationsSection">
                          <h3>Citations</h3>
                          <div className="citationList">
                            {message.payload.citations.map((citation, index) => {
                              const isActive =
                                citation.page_number === activeCitationPage;
                              return (
                                <button
                                  key={`${message.id}-${index}-${citation.page_number}`}
                                  type="button"
                                  className={`citationCard ${isActive ? "active" : ""}`}
                                  onClick={() =>
                                    setActiveCitationPage(citation.page_number)
                                  }
                                >
                                  <div className="citationMeta">
                                    <strong>Page {citation.page_number}</strong>
                                    <span>{citation.section}</span>
                                  </div>
                                  <p>{citation.excerpt}</p>
                                </button>
                              );
                            })}
                          </div>
                        </section>
                      ) : null}

                      {message.payload.executed_steps?.length ? (
                        <section className="stepsSection">
                          <h3>Executed Steps</h3>
                          <div className="stepsList">
                            {message.payload.executed_steps.map((step) => (
                              <div key={`${message.id}-step-${step.step_index}`} className="stepRow">
                                <div>
                                  <strong>
                                    Step {step.step_index}: {step.goal || step.query}
                                  </strong>
                                  <p>
                                    {step.route_strategy || "—"} •{" "}
                                    {step.selected_sections?.length
                                      ? step.selected_sections.join(", ")
                                      : "all sections"}
                                  </p>
                                </div>
                                <div className="stepPages">
                                  {step.cited_pages?.map((page) => (
                                    <button
                                      key={`${message.id}-step-${step.step_index}-page-${page}`}
                                      type="button"
                                      className="pageChip"
                                      onClick={() => setActiveCitationPage(page)}
                                    >
                                      p.{page}
                                    </button>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                        </section>
                      ) : null}
                    </>
                  ) : null}
                </div>
              </article>
            ))}
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <label className="composerLabel" htmlFor="question">
              Ask FinScout about the annual report
            </label>
            <div className="composerRow">
              <textarea
                id="question"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="How did profitability change and what does management attribute it to?"
                rows={3}
                disabled={pending || !report}
              />
              <button type="submit" className="sendButton" disabled={pending || !question.trim() || !report}>
                {pending ? "Thinking…" : "Ask"}
              </button>
            </div>
            {error ? <p className="errorText">{error}</p> : null}
          </form>
        </section>
      </section>

      <section className="viewerColumn">
        <PdfViewer
          pdfUrl={report?.pdf_url}
          title={report?.title}
          activePage={activeCitationPage}
          initialPage={1}
        />
        <DebugPanel
          open={debugOpen}
          onToggle={() => setDebugOpen((current) => !current)}
          planner={latestAssistantMessage?.payload?.planner}
          answer={latestAssistantMessage?.payload}
        />
      </section>
    </main>
  );
}
