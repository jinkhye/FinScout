"use client";

import { useEffect, useMemo, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";

pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

export default function PdfViewer({
  pdfUrl,
  initialPage = 1,
  activePage,
  title,
}) {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(initialPage);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    if (typeof activePage === "number" && activePage > 0) {
      setPageNumber(activePage);
    }
  }, [activePage]);

  const boundedPage = useMemo(() => {
    if (!numPages) {
      return pageNumber;
    }
    return Math.min(Math.max(pageNumber, 1), numPages);
  }, [numPages, pageNumber]);

  const jumpToPage = (nextPage) => {
    if (!Number.isFinite(nextPage)) {
      return;
    }
    setPageNumber(nextPage);
  };

  return (
    <section className="viewerShell">
      <div className="viewerHeader">
        <div>
          <p className="eyebrow">Annual Report</p>
          <h2>{title || "Document Viewer"}</h2>
        </div>
        <div className="viewerControls">
          <button
            type="button"
            className="iconButton"
            onClick={() => jumpToPage(boundedPage - 1)}
            disabled={boundedPage <= 1}
            aria-label="Previous page"
            title="Previous page"
          >
            ←
          </button>
          <label className="pageIndicator">
            <span>Page</span>
            <input
              type="number"
              min={1}
              max={numPages || undefined}
              value={boundedPage}
              onChange={(event) => jumpToPage(Number(event.target.value))}
            />
            <span>/ {numPages || "—"}</span>
          </label>
          <button
            type="button"
            className="iconButton"
            onClick={() => jumpToPage(boundedPage + 1)}
            disabled={Boolean(numPages) && boundedPage >= numPages}
            aria-label="Next page"
            title="Next page"
          >
            →
          </button>
        </div>
      </div>

      <div className="viewerCanvas">
        {!pdfUrl ? (
          <div className="viewerEmpty">Loading report…</div>
        ) : (
          <Document
            file={pdfUrl}
            onLoadSuccess={({ numPages: nextNumPages }) => {
              setNumPages(nextNumPages);
              setLoadError("");
            }}
            onLoadError={(error) => {
              setLoadError(error.message || "Unable to load PDF.");
            }}
            loading={<div className="viewerEmpty">Opening PDF…</div>}
            error={<div className="viewerEmpty">{loadError || "Unable to load PDF."}</div>}
          >
            <Page
              key={boundedPage}
              pageNumber={boundedPage}
              renderAnnotationLayer
              renderTextLayer
              width={720}
            />
          </Document>
        )}
      </div>
    </section>
  );
}
