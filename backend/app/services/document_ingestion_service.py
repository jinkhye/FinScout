from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any, Dict, List

from fastapi import UploadFile

from ..core.config import Settings
from ..schemas.document import DocumentProcessResponse, PageOutput
from .auditor_extraction import (
    build_auditor_output_payload,
    build_consolidated_auditor_markdown,
    call_auditor_extraction_model,
    derive_year,
    load_auditor_pages,
)
from .ingestion_pipeline import (
    PipelineArtifacts,
    build_page_to_section_map,
    build_pipeline_artifacts,
    chunk_pages,
    clean_markdown_for_embedding,
    get_gemini_client,
    pages_to_target_pages,
    write_json_payload,
)
from .page_classification import normalize_classified_pages
from .table_processing import (
    extract_table_blocks,
    inject_table_summaries,
    replace_table_blocks_with_placeholders,
    summarize_table_with_gemini,
)


class DocumentIngestionService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def process_upload(
        self,
        upload: UploadFile,
        classified_pages_raw: Any,
    ) -> DocumentProcessResponse:
        saved_path = await self._save_upload(upload)
        pdf_name = saved_path.name

        try:
            classified_pages = normalize_classified_pages(classified_pages_raw)
            if not classified_pages:
                raise ValueError("classified_pages cannot be empty")

            pages, auditor_payload = await asyncio.to_thread(
                self._run_pipeline, saved_path, classified_pages
            )
            metadata = self._extract_auditor_metadata(auditor_payload)

            return self._build_response(
                file_path=saved_path,
                pdf_name=pdf_name,
                classified_pages=classified_pages,
                pages=pages,
                metadata=metadata,
                status="success",
                error=None,
            )
        except Exception as exc:
            return DocumentProcessResponse(
                file_path=str(saved_path),
                pdf=pdf_name,
                status="error",
                error=str(exc),
                errors=[str(exc)],
            )

    async def _save_upload(self, upload: UploadFile) -> Path:
        self._settings.uploads_dir.mkdir(parents=True, exist_ok=True)

        filename = Path(upload.filename or "uploaded.pdf").name
        destination = self._settings.uploads_dir / filename
        destination.write_bytes(await upload.read())
        return destination

    def _run_pipeline(
        self,
        saved_path: Path,
        classified_pages: Dict[str, List[int]],
    ) -> tuple[List[PageOutput], Dict[str, Any]]:
        artifacts = build_pipeline_artifacts(self._settings, saved_path.name)
        artifacts.pipeline_dir.mkdir(parents=True, exist_ok=True)

        if saved_path.resolve() != artifacts.upload_path.resolve():
            shutil.copy2(saved_path, artifacts.upload_path)

        page_to_section = self._page_to_section(classified_pages)
        pages_to_parse = sorted(page_to_section)

        raw_pages = self._parse_pdf_pages(
            saved_path=saved_path,
            page_to_section=page_to_section,
            pages_to_parse=pages_to_parse,
        )
        self._write_pages(
            artifacts.raw_json,
            saved_path.name,
            classified_pages,
            raw_pages,
        )

        cleaned_pages = self._clean_pages(raw_pages)
        self._write_pages(
            artifacts.cleaned_json,
            saved_path.name,
            classified_pages,
            cleaned_pages,
        )

        auditor_payload = self._extract_auditor_payload(
            pdf_name=saved_path.name,
            classified_pages=classified_pages,
            cleaned_pages=cleaned_pages,
        )
        write_json_payload(artifacts.auditor_json, auditor_payload)

        pages = [PageOutput.model_validate(page) for page in cleaned_pages]
        return pages, auditor_payload

    def _page_to_section(
        self, classified_pages: Dict[str, List[int]]
    ) -> Dict[int, str]:
        page_to_section, duplicates = build_page_to_section_map(classified_pages)
        if duplicates:
            detail = ", ".join(
                f"{page}: {sections}" for page, sections in duplicates[:10]
            )
            raise ValueError(f"Pages appear in multiple sections: {detail}")
        if not page_to_section:
            raise ValueError("No pages to parse")
        return page_to_section

    def _parse_pdf_pages(
        self,
        *,
        saved_path: Path,
        page_to_section: Dict[int, str],
        pages_to_parse: List[int],
    ) -> List[Dict[str, Any]]:
        from llama_cloud import LlamaCloud  # type: ignore

        client = LlamaCloud()
        markdown_by_page: Dict[int, str] = {}

        for batch_pages in chunk_pages(
            pages_to_parse, self._settings.llamaparse_batch_size
        ):
            result = client.parsing.parse(
                upload_file=str(saved_path),
                tier=self._settings.llamaparse_tier,
                version=self._settings.llamaparse_version,
                page_ranges={"target_pages": pages_to_target_pages(batch_pages)},
                processing_options={
                    "ignore": {
                        "ignore_text_in_image": self._settings.llamaparse_ignore_text_in_image,
                    }
                },
                expand=["markdown"],
            )

            markdown_result = getattr(result, "markdown", None)
            for page in getattr(markdown_result, "pages", []) or []:
                page_number = getattr(page, "page_number", None)
                if isinstance(page_number, int) and not isinstance(page_number, bool):
                    markdown_by_page[page_number] = getattr(page, "markdown", "") or ""

        return [
            {
                "page_number": page_number,
                "section": page_to_section[page_number],
                "markdown_raw": markdown_by_page.get(page_number, ""),
            }
            for page_number in pages_to_parse
        ]

    def _clean_pages(self, raw_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        gemini_client = get_gemini_client()
        cleaned_pages: List[Dict[str, Any]] = []

        for raw_page in raw_pages:
            page_number = int(raw_page["page_number"])
            section = str(raw_page.get("section") or "unknown")
            raw_markdown = str(raw_page.get("markdown_raw") or "")
            table_blocks = extract_table_blocks(raw_markdown)

            markdown_with_slots = replace_table_blocks_with_placeholders(raw_markdown)
            markdown_clean = clean_markdown_for_embedding(markdown_with_slots)

            tables: List[Dict[str, Any]] = []
            summaries: List[str] = []
            for table_index, table_raw in enumerate(table_blocks, start=1):
                summary, status, error = summarize_table_with_gemini(
                    gemini_client,
                    self._settings,
                    section,
                    page_number,
                    table_index,
                    table_raw,
                )
                summaries.append(summary)
                tables.append(
                    {
                        "table_index": table_index,
                        "table_raw": table_raw,
                        "table_summary": summary,
                        "status": status,
                        "error": error,
                        "model": self._settings.gemini_model,
                    }
                )

            cleaned_pages.append(
                {
                    "page_number": page_number,
                    "section": section,
                    "has_table": bool(table_blocks),
                    "tables": tables,
                    "markdown_raw": raw_markdown,
                    "markdown_clean": inject_table_summaries(markdown_clean, summaries),
                }
            )

        return cleaned_pages

    def _extract_auditor_payload(
        self,
        *,
        pdf_name: str,
        classified_pages: Dict[str, List[int]],
        cleaned_pages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        source_payload = {
            "pdf": pdf_name,
            "total_pages_parsed": len(cleaned_pages),
            "classified_pages": classified_pages,
            "pages": cleaned_pages,
        }
        auditor_pages = load_auditor_pages(source_payload)
        consolidated_markdown = build_consolidated_auditor_markdown(auditor_pages)
        auditor_response = call_auditor_extraction_model(
            get_gemini_client(),
            self._settings.gemini_model,
            consolidated_markdown,
        )
        return build_auditor_output_payload(source_payload, auditor_response)

    def _write_pages(
        self,
        path: Path,
        pdf_name: str,
        classified_pages: Dict[str, List[int]],
        pages: List[Dict[str, Any]],
    ) -> None:
        write_json_payload(
            path,
            {
                "pdf": pdf_name,
                "total_pages_parsed": len(pages),
                "classified_pages": classified_pages,
                "pages": pages,
            },
        )

    def _extract_auditor_metadata(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, str]:
        metadata = {
            "company_name": "unknown",
            "year": "unknown",
            "auditor_opinion": "unknown",
            "auditor_firm": "unknown",
            "auditor_name": "unknown",
            "audit_period": "unknown",
        }

        for key in metadata:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                metadata[key] = value.strip()

        if metadata["year"] == "unknown" and metadata["audit_period"] != "unknown":
            metadata["year"] = derive_year(metadata["audit_period"])

        return metadata

    def _build_response(
        self,
        *,
        file_path: Path,
        pdf_name: str,
        classified_pages: Dict[str, List[int]],
        pages: List[PageOutput],
        metadata: Dict[str, str],
        status: str,
        error: str | None,
    ) -> DocumentProcessResponse:
        return DocumentProcessResponse(
            file_path=str(file_path),
            pdf=pdf_name,
            company_name=metadata["company_name"],
            year=metadata["year"],
            auditor_opinion=metadata["auditor_opinion"],
            auditor_firm=metadata["auditor_firm"],
            auditor_name=metadata["auditor_name"],
            audit_period=metadata["audit_period"],
            total_pages_parsed=len(pages),
            classified_pages=classified_pages,
            pages=pages,
            status=status,
            error=error,
            errors=[] if error is None else [error],
        )
