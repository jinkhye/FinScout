"use client";

export default function DebugPanel({ open, onToggle, planner, answer }) {
  return (
    <aside className={`debugPanel ${open ? "open" : "closed"}`}>
      <div className="debugHeader">
        <div>
          <p className="eyebrow">Debug</p>
          <h2>Planner & Retrieval</h2>
        </div>
        <button
          type="button"
          className="ghostButton"
          onClick={onToggle}
          aria-expanded={open}
        >
          {open ? "Hide" : "Show"}
        </button>
      </div>

      {open ? (
        <div className="debugBody">
          <section className="debugSection">
            <h3>Summary</h3>
            <dl className="debugGrid">
              <div>
                <dt>Intent</dt>
                <dd>{planner?.intent || "—"}</dd>
              </div>
              <div>
                <dt>Multi-step</dt>
                <dd>{planner?.is_multi_step ? "Yes" : "No"}</dd>
              </div>
              <div>
                <dt>Route</dt>
                <dd>{answer?.route_strategy || "—"}</dd>
              </div>
              <div>
                <dt>Reranked</dt>
                <dd>{answer?.reranked ? "Yes" : "No"}</dd>
              </div>
              <div>
                <dt>Retried</dt>
                <dd>{answer?.retried ? "Yes" : "No"}</dd>
              </div>
              <div>
                <dt>Final query</dt>
                <dd>{answer?.final_query || "—"}</dd>
              </div>
            </dl>
          </section>

          <section className="debugSection">
            <h3>Executed Steps</h3>
            {answer?.executed_steps?.length ? (
              <ul className="debugList">
                {answer.executed_steps.map((step) => (
                  <li key={step.step_index}>
                    <strong>
                      Step {step.step_index}: {step.goal || step.query}
                    </strong>
                    <span>{step.route_strategy || "—"}</span>
                    <span>
                      Sections:{" "}
                      {step.selected_sections?.length
                        ? step.selected_sections.join(", ")
                        : "all sections"}
                    </span>
                    <span>
                      Pages:{" "}
                      {step.cited_pages?.length
                        ? step.cited_pages.join(", ")
                        : "—"}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="debugEmpty">No executed steps.</p>
            )}
          </section>

          <section className="debugSection">
            <h3>Planner Output</h3>
            <pre>{JSON.stringify(planner, null, 2)}</pre>
          </section>
        </div>
      ) : null}
    </aside>
  );
}
