"""Recursive DICOM dataset scanner and audit writer."""

from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from app.dicom.ct_series_loader import CTSeriesSummary, summarize_ct_series
from app.dicom.dicom_metadata import DicomMetadata, format_sequence, read_dicom_metadata


DICOM_LIKE_SUFFIXES = {".dcm", ".dicom", ".ima"}


@dataclass(frozen=True)
class DicomAuditResult:
    """Output paths and summary counts for one DICOM audit run."""

    output_dir: Path
    inventory_csv: Path
    ct_series_summary_csv: Path
    seg_inventory_csv: Path
    summary_md: Path
    summary: dict[str, int]


def scan_dicom_dataset(
    data_root: Path,
    output_root: Path,
    workers: int | None = None,
    use_processes: bool = False,
) -> DicomAuditResult:
    """Scan a data root recursively and write reproducible DICOM audit files."""

    data_root = data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    audit_dir = output_root.expanduser().resolve() / "dicom_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    files = _discover_dicom_like_files(data_root)
    metadata = _read_metadata_parallel(files, workers=workers, use_processes=use_processes)
    ct_files = [item for item in metadata if item.classification == "CT"]
    seg_files = [item for item in metadata if item.classification == "SEG"]
    summaries = summarize_ct_series(ct_files)

    inventory_csv = audit_dir / "dicom_inventory.csv"
    ct_series_summary_csv = audit_dir / "ct_series_summary.csv"
    seg_inventory_csv = audit_dir / "seg_inventory.csv"
    summary_md = audit_dir / "dicom_audit_summary.md"

    _write_inventory_csv(inventory_csv, metadata)
    _write_ct_series_summary_csv(ct_series_summary_csv, summaries)
    _write_seg_inventory_csv(seg_inventory_csv, seg_files)


    summary = {
        "files_scanned": len(files),
        "readable_dicom_files": sum(1 for item in metadata if item.classification != "INVALID"),
        "invalid_files": sum(1 for item in metadata if item.classification == "INVALID"),
        "ct_slices": len(ct_files),
        "ct_series": len(summaries),
        "seg_objects": len(seg_files),
        "other_dicom_files": sum(1 for item in metadata if item.classification == "OTHER"),
    }
    _write_summary_markdown(summary_md, data_root, summary)

    return DicomAuditResult(
        output_dir=audit_dir,
        inventory_csv=inventory_csv,
        ct_series_summary_csv=ct_series_summary_csv,
        seg_inventory_csv=seg_inventory_csv,
        summary_md=summary_md,
        summary=summary,
    )


def _write_inventory_csv(path: Path, rows: list[DicomMetadata]) -> None:
    fieldnames = [
        "classification",
        "file_path",
        "patient_id",
        "study_instance_uid",
        "series_instance_uid",
        "sop_instance_uid",
        "modality",
        "instance_number",
        "image_position_patient",
        "rows",
        "columns",
        "referenced_sop_instance_uids",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            writer.writerow(_metadata_row(item))


def _write_ct_series_summary_csv(path: Path, rows: list[CTSeriesSummary]) -> None:
    fieldnames = [
        "study_instance_uid",
        "series_instance_uid",
        "patient_id",
        "slice_count",
        "first_file_path",
        "last_file_path",
        "sort_method",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            writer.writerow(item.__dict__)


def _write_seg_inventory_csv(path: Path, rows: list[DicomMetadata]) -> None:
    fieldnames = [
        "file_path",
        "patient_id",
        "study_instance_uid",
        "series_instance_uid",
        "sop_instance_uid",
        "modality",
        "referenced_sop_instance_uids",
        "rows",
        "columns",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            row = _metadata_row(item)
            writer.writerow({key: row[key] for key in fieldnames})


def _write_summary_markdown(path: Path, data_root: Path, summary: dict[str, int]) -> None:
    lines = [
        "# DICOM Audit Summary",
        "",
        f"Data root: `{data_root}`",
        "",
        f"- number of files scanned: {summary['files_scanned']}",
        f"- number of readable DICOM files: {summary['readable_dicom_files']}",
        f"- number of invalid files: {summary['invalid_files']}",
        f"- number of CT slices: {summary['ct_slices']}",
        f"- number of CT series: {summary['ct_series']}",
        f"- number of SEG objects: {summary['seg_objects']}",
        f"- number of OTHER DICOM files: {summary['other_dicom_files']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _metadata_row(item: DicomMetadata) -> dict[str, object]:
    return {
        "classification": item.classification,
        "file_path": str(item.file_path),
        "patient_id": item.patient_id or "",
        "study_instance_uid": item.study_instance_uid or "",
        "series_instance_uid": item.series_instance_uid or "",
        "sop_instance_uid": item.sop_instance_uid or "",
        "modality": item.modality or "",
        "instance_number": item.instance_number if item.instance_number is not None else "",
        "image_position_patient": format_sequence(item.image_position_patient),
        "rows": item.rows if item.rows is not None else "",
        "columns": item.columns if item.columns is not None else "",
        "referenced_sop_instance_uids": format_sequence(item.referenced_sop_instance_uids),
        "error": item.error or "",
    }


def _read_metadata_parallel(files: list[Path], workers: int | None = None, use_processes: bool = False) -> list[DicomMetadata]:
    if not files:
        return []
    max_workers = workers or min(32, (os.cpu_count() or 1) + 4)
    max_workers = max(1, max_workers)
    if max_workers == 1:
        return [read_dicom_metadata(path) for path in files]
    if use_processes:
        chunks = _chunk_paths(files, chunk_size=1000)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return [item for chunk in executor.map(_read_metadata_chunk, chunks) for item in chunk]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(read_dicom_metadata, files))


def _read_metadata_chunk(files: list[Path]) -> list[DicomMetadata]:
    return [read_dicom_metadata(path) for path in files]


def _chunk_paths(files: list[Path], chunk_size: int) -> list[list[Path]]:
    return [files[index : index + chunk_size] for index in range(0, len(files), chunk_size)]


def _discover_dicom_like_files(data_root: Path) -> list[Path]:
    return sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and (path.suffix.lower() in DICOM_LIKE_SUFFIXES or path.suffix == "" or path.name.upper() == "DICOMDIR")
    )
