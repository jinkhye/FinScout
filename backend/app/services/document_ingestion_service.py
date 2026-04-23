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
    write_auditor_debug_input,
)
from .document_artifacts import normalize_classified_pages
from .ingestion_pipeline import (
    PipelineArtifacts,
    build_page_to_section_map,
    build_pipeline_artifacts,
    chunk_pages,
    clean_markdown_for_embedding,
    get_gemini_client,
    load_existing_pages,
    load_json_payload,
    pages_to_target_pages,
    write_debug_markdown_chunks,
    write_json_payload,
)
from .table_processing import (
    extract_table_blocks,
    inject_table_summaries,
    replace_table_blocks_with_placeholders,
    summarize_table_with_gemini,
)


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


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
        artifacts = build_pipeline_artifacts(self._settings, pdf_name)

        try:
            classified_pages = normalize_classified_pages(classified_pages_raw)
            if not classified_pages:
                raise ValueError("classified_pages cannot be empty")

            artifact_paths = await asyncio.to_thread(
                self._run_pipeline, saved_path, classified_pages, artifacts
            )

            cleaned_payload = load_json_payload(artifacts.cleaned_json)
            auditor_payload = load_json_payload(artifacts.auditor_json)
            pages = self._pages_from_payload(cleaned_payload, classified_pages)
            metadata = self._extract_auditor_metadata(auditor_payload)

            return self._build_response(
                file_path=saved_path,
                pdf_name=pdf_name,
                classified_pages=classified_pages,
                pages=pages,
                metadata=metadata,
                artifact_paths=artifact_paths,
                pipeline_stage="completed",
                status="success",
                error=None,
            )
        except Exception as exc:
            failed_stage = (
                exc.stage if isinstance(exc, PipelineStageError) else "unknown"
            )
            artifact_paths = self._collect_existing_artifact_paths(artifacts)
            latest_payload = self._load_latest_checkpoint(artifacts)
            pages = self._pages_from_checkpoint(latest_payload, classified_pages_raw)
            metadata = self._extract_auditor_metadata(
                self._try_load_json(artifacts.auditor_json)
            )

            return self._build_response(
                file_path=saved_path,
                pdf_name=pdf_name,
                classified_pages=(
                    classified_pages
                    if "classified_pages" in locals()
                    else (
                        normalize_classified_pages(classified_pages_raw)
                        if isinstance(classified_pages_raw, dict)
                        else {}
                    )
                ),
                pages=pages,
                metadata=metadata,
                artifact_paths=artifact_paths,
                pipeline_stage=failed_stage,
                status="error",
                error=str(exc),
            )

    async def _save_upload(self, upload: UploadFile) -> Path:
        self._settings.uploads_dir.mkdir(parents=True, exist_ok=True)

        filename = Path(upload.filename or "uploaded.pdf").name
        destination = self._settings.uploads_dir / filename
        content = await upload.read()
        destination.write_bytes(content)
        return destination

    def _run_pipeline(
        self,
        saved_path: Path,
        classified_pages: Dict[str, List[int]],
        artifacts: PipelineArtifacts,
    ) -> Dict[str, str]:
        artifacts.pipeline_dir.mkdir(parents=True, exist_ok=True)
        artifacts.debug_dir.mkdir(parents=True, exist_ok=True)

        if saved_path.resolve() != artifacts.upload_path.resolve():
            shutil.copy2(saved_path, artifacts.upload_path)

        page_to_section, duplicates = build_page_to_section_map(classified_pages)
        if duplicates:
            detail = ", ".join(
                f"{page}: {sections}" for page, sections in duplicates[:10]
            )
            raise PipelineStageError(
                "classification",
                f"Pages appear in multiple sections (first 10): {detail}",
            )

        pages_to_parse = sorted(page_to_section.keys())
        if not pages_to_parse:
            raise PipelineStageError("classification", "No pages to parse")

        raw_pages = self._run_raw_stage(
            saved_path=saved_path,
            classified_pages=classified_pages,
            page_to_section=page_to_section,
            pages_to_parse=pages_to_parse,
            artifacts=artifacts,
        )

        parsed_pages = self._run_postprocess_stage(
            raw_pages=raw_pages,
            classified_pages=classified_pages,
            artifacts=artifacts,
        )

        cleaned_pages = self._run_rebuild_clean_stage(
            raw_pages=raw_pages,
            parsed_pages=parsed_pages,
            classified_pages=classified_pages,
            artifacts=artifacts,
        )

        self._run_auditor_stage(cleaned_pages=cleaned_pages, artifacts=artifacts)
        return self._artifact_path_map(artifacts)

    def _run_raw_stage(
        self,
        *,
        saved_path: Path,
        classified_pages: Dict[str, List[int]],
        page_to_section: Dict[int, str],
        pages_to_parse: List[int],
        artifacts: PipelineArtifacts,
    ) -> Dict[int, Dict[str, Any]]:
        existing_pages = load_existing_pages(artifacts.raw_json, saved_path.name)
        remaining_pages = [
            page for page in pages_to_parse if page not in existing_pages
        ]

        if not remaining_pages:
            if existing_pages:
                self._write_checkpoint_payload(
                    existing_pages,
                    classified_pages,
                    artifacts.raw_json,
                    saved_path.name,
                )
            return existing_pages

        from llama_cloud import LlamaCloud  # type: ignore

        client = LlamaCloud()
        batches = chunk_pages(remaining_pages, self._settings.llamaparse_batch_size)

        for batch_index, batch_pages in enumerate(batches, start=1):
            target_pages = pages_to_target_pages(batch_pages)
            try:
                result = client.parsing.parse(
                    upload_file=str(saved_path),
                    tier=self._settings.llamaparse_tier,
                    version=self._settings.llamaparse_version,
                    page_ranges={"target_pages": target_pages},
                    processing_options={
                        "ignore": {
                            "ignore_text_in_image": self._settings.llamaparse_ignore_text_in_image,
                        }
                    },
                    expand=["markdown"],
                )
            except Exception as exc:
                raise PipelineStageError(
                    "raw", f"LlamaParse failed on batch {batch_index}: {exc}"
                ) from exc

            md_by_page: Dict[int, str] = {}
            markdown_result = getattr(result, "markdown", None)
            pages_result = (
                getattr(markdown_result, "pages", None)
                if markdown_result is not None
                else []
            )
            for page in pages_result or []:
                page_number = getattr(page, "page_number", None)
                if isinstance(page_number, bool) or not isinstance(page_number, int):
                    continue
                md_by_page[int(page_number)] = getattr(page, "markdown", "") or ""

            for page_number in batch_pages:
                existing_pages[page_number] = {
                    "page_number": page_number,
                    "section": page_to_section[page_number],
                    "markdown_raw": md_by_page.get(page_number, ""),
                }

            self._write_checkpoint_payload(
                existing_pages,
                classified_pages,
                artifacts.raw_json,
                saved_path.name,
            )

        return existing_pages

    def _run_postprocess_stage(
        self,
        *,
        raw_pages: Dict[int, Dict[str, Any]],
        classified_pages: Dict[str, List[int]],
        artifacts: PipelineArtifacts,
    ) -> Dict[int, Dict[str, Any]]:
        existing_pages = load_existing_pages(
            artifacts.parsed_json, artifacts.upload_path.name
        )
        raw_pages_list = [raw_pages[page_number] for page_number in sorted(raw_pages)]
        remaining_pages = [
            page
            for page in raw_pages_list
            if int(page["page_number"]) not in existing_pages
        ]

        if not remaining_pages:
            if existing_pages:
                self._write_checkpoint_payload(
                    existing_pages,
                    classified_pages,
                    artifacts.parsed_json,
                    artifacts.upload_path.name,
                )
            return existing_pages

        gemini_client = get_gemini_client()
        batches = chunk_pages(
            [int(page["page_number"]) for page in remaining_pages],
            self._settings.llamaparse_batch_size,
        )
        raw_pages_by_number = {
            int(page["page_number"]): page for page in raw_pages_list
        }

        for batch_index, batch_pages in enumerate(batches, start=1):
            batch_pages_out: List[Dict[str, Any]] = []
            for page_number in batch_pages:
                raw_page = raw_pages_by_number[page_number]
                raw_markdown = raw_page.get("markdown_raw", "")
                section = raw_page.get("section") or "unknown"
                table_blocks = extract_table_blocks(raw_markdown)
                has_table = bool(table_blocks)

                markdown_with_slots = replace_table_blocks_with_placeholders(
                    raw_markdown
                )
                cleaned_markdown = clean_markdown_for_embedding(markdown_with_slots)

                table_entries: List[Dict[str, Any]] = []
                table_summaries_for_injection: List[str] = []
                for table_idx, table_raw in enumerate(table_blocks, start=1):
                    summary, status, error = summarize_table_with_gemini(
                        gemini_client,
                        self._settings,
                        section,
                        page_number,
                        table_idx,
                        table_raw,
                    )
                    table_summaries_for_injection.append(summary)
                    table_entries.append(
                        {
                            "table_index": table_idx,
                            "table_raw": table_raw,
                            "table_summary": summary,
                            "status": status,
                            "error": error,
                            "model": self._settings.gemini_model,
                        }
                    )

                cleaned_markdown = inject_table_summaries(
                    cleaned_markdown,
                    table_summaries_for_injection,
                )

                batch_pages_out.append(
                    {
                        "page_number": page_number,
                        "section": section,
                        "has_table": has_table,
                        "tables": table_entries,
                        "markdown_raw": raw_markdown,
                        "markdown_clean": cleaned_markdown,
                    }
                )

            for page in batch_pages_out:
                existing_pages[int(page["page_number"])] = page

            self._write_checkpoint_payload(
                existing_pages,
                classified_pages,
                artifacts.parsed_json,
                artifacts.upload_path.name,
            )
            write_debug_markdown_chunks(
                artifacts.debug_dir,
                batch_pages_out,
                self._settings.debug_pages_per_file,
                raw_key="markdown_raw",
                clean_key="markdown_clean",
            )

        self._write_checkpoint_payload(
            existing_pages,
            classified_pages,
            artifacts.parsed_json,
            artifacts.upload_path.name,
        )
        return existing_pages

    def _run_rebuild_clean_stage(
        self,
        *,
        raw_pages: Dict[int, Dict[str, Any]],
        parsed_pages: Dict[int, Dict[str, Any]],
        classified_pages: Dict[str, List[int]],
        artifacts: PipelineArtifacts,
    ) -> Dict[int, Dict[str, Any]]:
        existing_pages = load_existing_pages(
            artifacts.cleaned_json, artifacts.upload_path.name
        )
        if set(raw_pages).issubset(existing_pages) and existing_pages:
            return existing_pages

        cleaned_pages_by_number: Dict[int, Dict[str, Any]] = dict(existing_pages)
        for page_number in sorted(raw_pages):
            if page_number in cleaned_pages_by_number:
                continue

            raw_page = raw_pages[page_number]
            parsed_page = parsed_pages.get(page_number)
            if parsed_page is None:
                raise PipelineStageError(
                    "clean", f"Missing parsed page for page {page_number}"
                )

            raw_markdown = raw_page.get("markdown_raw", "")
            section = raw_page.get("section") or parsed_page.get("section") or "unknown"
            has_table = bool(extract_table_blocks(raw_markdown))

            tables = parsed_page.get("tables", [])
            if not isinstance(tables, list):
                raise PipelineStageError(
                    "clean", f"Page {page_number} tables field must be a list"
                )

            table_summaries: List[str] = []
            for table in tables:
                if not isinstance(table, dict):
                    raise PipelineStageError(
                        "clean", f"Page {page_number} has a non-object table entry"
                    )
                table_summaries.append(table.get("table_summary", ""))

            markdown_with_slots = replace_table_blocks_with_placeholders(raw_markdown)
            cleaned_markdown = clean_markdown_for_embedding(markdown_with_slots)
            cleaned_markdown = inject_table_summaries(cleaned_markdown, table_summaries)

            cleaned_page = dict(parsed_page)
            cleaned_page.update(
                {
                    "page_number": page_number,
                    "section": section,
                    "has_table": has_table,
                    "tables": tables,
                    "markdown_raw": raw_markdown,
                    "markdown_clean": cleaned_markdown,
                }
            )
            cleaned_pages_by_number[page_number] = cleaned_page

            self._write_checkpoint_payload(
                cleaned_pages_by_number,
                classified_pages,
                artifacts.cleaned_json,
                artifacts.upload_path.name,
            )

        self._write_checkpoint_payload(
            cleaned_pages_by_number,
            classified_pages,
            artifacts.cleaned_json,
            artifacts.upload_path.name,
        )
        return cleaned_pages_by_number

    def _run_auditor_stage(
        self,
        *,
        cleaned_pages: Dict[int, Dict[str, Any]],
        artifacts: PipelineArtifacts,
    ) -> Dict[str, Any]:
        if artifacts.auditor_json.exists():
            existing_auditor = load_json_payload(artifacts.auditor_json)
            if existing_auditor:
                return existing_auditor

        source_payload = load_json_payload(artifacts.cleaned_json)
        auditor_pages = load_auditor_pages(source_payload)
        consolidated_markdown = build_consolidated_auditor_markdown(auditor_pages)
        write_auditor_debug_input(consolidated_markdown, artifacts.debug_input_md)

        gemini_client = get_gemini_client()
        if gemini_client is None:
            raise PipelineStageError("auditor", "Gemini client unavailable")

        try:
            auditor_response = call_auditor_extraction_model(
                gemini_client,
                self._settings.gemini_model,
                consolidated_markdown,
            )
        except Exception as exc:
            raise PipelineStageError(
                "auditor", f"Auditor extraction failed: {exc}"
            ) from exc

        output_payload = build_auditor_output_payload(source_payload, auditor_response)
        write_json_payload(artifacts.auditor_json, output_payload)
        return output_payload

    def _write_checkpoint_payload(
        self,
        pages_by_number: Dict[int, Dict[str, Any]],
        classified: Dict[str, List[int]],
        path: Path,
        pdf_name: str,
    ) -> None:
        pages_out = [
            pages_by_number[page_number] for page_number in sorted(pages_by_number)
        ]
        payload: Dict[str, Any] = {
            "pdf": pdf_name,
            "total_pages_parsed": len(pages_out),
            "classified_pages": classified,
            "pages": pages_out,
        }
        write_json_payload(path, payload)

    def _collect_existing_artifact_paths(
        self, artifacts: PipelineArtifacts
    ) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        for key, path in self._artifact_path_map_paths(artifacts).items():
            if path.exists():
                paths[key] = str(path)
        return paths

    def _artifact_path_map_paths(self, artifacts: PipelineArtifacts) -> Dict[str, Path]:
        return {
            "upload_path": artifacts.upload_path,
            "pipeline_dir": artifacts.pipeline_dir,
            "raw_json": artifacts.raw_json,
            "parsed_json": artifacts.parsed_json,
            "cleaned_json": artifacts.cleaned_json,
            "auditor_json": artifacts.auditor_json,
            "debug_dir": artifacts.debug_dir,
            "debug_input_md": artifacts.debug_input_md,
        }

    def _artifact_path_map(self, artifacts: PipelineArtifacts) -> Dict[str, str]:
        return {
            key: str(path)
            for key, path in self._artifact_path_map_paths(artifacts).items()
        }

    def _load_latest_checkpoint(
        self, artifacts: PipelineArtifacts
    ) -> Dict[str, Any] | None:
        for path in (artifacts.cleaned_json, artifacts.parsed_json, artifacts.raw_json):
            if path.exists():
                return load_json_payload(path)
        return None

    def _try_load_json(self, path: Path) -> Dict[str, Any] | None:
        if not path.exists():
            return None
        return load_json_payload(path)

    def _pages_from_checkpoint(
        self,
        payload: Dict[str, Any] | None,
        classified_pages_raw: Any,
    ) -> List[PageOutput]:
        if not payload:
            return []

        pages_raw = payload.get("pages", [])
        if not isinstance(pages_raw, list):
            return []

        try:
            classified_pages = normalize_classified_pages(classified_pages_raw)
        except Exception:
            classified_pages = {}

        selected = self._select_pages(pages_raw, classified_pages)
        return [PageOutput.model_validate(page) for page in selected]

    def _pages_from_payload(
        self,
        payload: Dict[str, Any],
        classified_pages: Dict[str, List[int]],
    ) -> List[PageOutput]:
        pages_raw = payload.get("pages", [])
        if not isinstance(pages_raw, list):
            raise ValueError("pages field must be a list")

        selected = self._select_pages(pages_raw, classified_pages)
        if not selected:
            raise ValueError("No pages matched the supplied classified_pages input")
        return [PageOutput.model_validate(page) for page in selected]

    def _select_pages(
        self,
        pages: List[Dict[str, Any]],
        classified_pages: Dict[str, List[int]],
    ) -> List[Dict[str, Any]]:
        allowed_pages = set()
        for page_numbers in classified_pages.values():
            allowed_pages.update(page_numbers)

        selected: List[Dict[str, Any]] = []
        for page in pages:
            page_number = page.get("page_number")
            if isinstance(page_number, bool) or not isinstance(page_number, int):
                continue
            if page_number in allowed_pages:
                selected.append(page)

        selected.sort(key=lambda page: int(page["page_number"]))
        return selected

    def _extract_auditor_metadata(
        self,
        payload: Dict[str, Any] | None,
    ) -> Dict[str, str]:
        metadata = {
            "company_name": "unknown",
            "year": "unknown",
            "auditor_opinion": "unknown",
            "auditor_firm": "unknown",
            "auditor_name": "unknown",
            "audit_period": "unknown",
        }

        if not payload:
            return metadata

        for key in (
            "company_name",
            "auditor_opinion",
            "auditor_firm",
            "auditor_name",
            "audit_period",
            "year",
        ):
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
        artifact_paths: Dict[str, str],
        pipeline_stage: str,
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
            artifact_paths=artifact_paths,
            pipeline_stage=pipeline_stage,
            status=status,
            error=error,
            errors=[] if error is None else [error],
        )
