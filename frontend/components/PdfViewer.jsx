"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";

pdfjs.GlobalWorkerOptions.workerSrc =
  `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

export default function PdfViewer({
  pdfUrl,
  initialPage = 1,
  activePage,
  navigationToken = 0,
  title,
}) {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(initialPage);
  const [loadError, setLoadError] = useState("");
  const viewerCanvasRef = useRef(null);

  useEffect(() => {
    if (typeof activePage === "number" && activePage > 0) {
      setPageNumber(activePage);
    }
  }, [activePage, navigationToken]);

  const boundedPage = useMemo(() => {
    if (!numPages) {
      return pageNumber;
    }
    return Math.min(Math.max(pageNumber, 1), numPages);
  }, [numPages, pageNumber]);

  const jumpToPage = (nextPage) => {
    if (!Number.isFinite(nextPage) || nextPage < 1) {
      return;
    }
    const boundedNextPage = numPages
      ? Math.min(Math.max(nextPage, 1), numPages)
      : nextPage;
    setPageNumber(boundedNextPage);
  };

  useEffect(() => {
    const root = viewerCanvasRef.current;
    if (!root || !activePage) {
      return;
    }

    const target = root.querySelector(
      `[data-page-number="${activePage}"]`
    );
    target?.scrollIntoView({
      block: "start",
      behavior: "smooth",
    });
  }, [activePage]);

  useEffect(() => {
    const root = viewerCanvasRef.current;
    if (!root || !numPages) {
      return undefined;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const visibleEntries = entries
          .filter((entry) => entry.isIntersecting)
          .sort(
            (left, right) => right.intersectionRatio - left.intersectionRatio
          );

        if (visibleEntries.length === 0) {
          return;
        }

        const nextPage = Number(
          visibleEntries[0].target.getAttribute("data-page-number")
        );
        if (Number.isFinite(nextPage)) {
          setPageNumber(nextPage);
        }
      },
      {
        root,
        threshold: [0.4, 0.6, 0.8],
      }
    );

    root.querySelectorAll("[data-page-number]").forEach((element) => {
      if (element) {
        observer.observe(element);
      }
    });

    return () => observer.disconnect();
  }, [numPages]);

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
            {"<-"}
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
            <span>/ {numPages || "-"}</span>
          </label>
          <button
            type="button"
            className="iconButton"
            onClick={() => jumpToPage(boundedPage + 1)}
            disabled={Boolean(numPages) && boundedPage >= numPages}
            aria-label="Next page"
            title="Next page"
          >
            {"->"}
          </button>
        </div>
      </div>

      <div ref={viewerCanvasRef} className="viewerCanvas">
        {!pdfUrl ? (
          <div className="viewerEmpty">Loading report...</div>
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
            loading={<div className="viewerEmpty">Opening PDF...</div>}
            error={
              <div className="viewerEmpty">
                {loadError || "Unable to load PDF."}
              </div>
            }
          >
            {Array.from(new Array(numPages || 0), (_, index) => {
              const currentPage = index + 1;
              return (
                <div
                  key={currentPage}
                  data-page-number={currentPage}
                  className={`pdfPageFrame ${
                    currentPage === boundedPage ? "active" : ""
                  }`}
                >
                  <Page
                    pageNumber={currentPage}
                    renderAnnotationLayer={false}
                    renderTextLayer={false}
                    width={720}
                  />
                </div>
              );
            })}
          </Document>
        )}
      </div>
    </section>
  );
}
