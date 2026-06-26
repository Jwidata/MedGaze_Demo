"""Strict CT/SEG matching from DICOM audit CSV outputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ROIMatchResult:
    """Output paths and summary counts for CT/SEG matching."""

    output_dir: Path
    match_table_csv: Path
    summary_md: Path
    summary: dict[str, int]


def match_ct_seg_from_audit(output_root: Path) -> ROIMatchResult:
    """Match SEG objects to CT series using audit CSV files."""

    output_root = output_root.expanduser().resolve()
    audit_dir = output_root / "dicom_audit"
    inventory_csv = audit_dir / "dicom_inventory.csv"
    ct_series_summary_csv = audit_dir / "ct_series_summary.csv"
    seg_inventory_csv = audit_dir / "seg_inventory.csv"

    _require_file(inventory_csv)
    _require_file(ct_series_summary_csv)
    _require_file(seg_inventory_csv)

    ct_index = _build_ct_index(inventory_csv)
    # Read the series summary to validate the expected Step 2 input exists and is parseable.
    _read_csv(ct_series_summary_csv)
    seg_rows = _read_csv(seg_inventory_csv)

    output_dir = output_root / "roi_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)
    match_table_csv = output_dir / "ct_seg_match_table.csv"
    summary_md = output_dir / "ct_seg_match_summary.md"

    match_rows = [_match_seg_row(seg_row, ct_index) for seg_row in seg_rows]
    _write_match_table(match_table_csv, match_rows)

    summary = {
        "seg_objects": len(seg_rows),
        "strict_matches": sum(1 for row in match_rows if row["match_status"] == "matched_strict"),
        "partial_matches": sum(1 for row in match_rows if row["match_status"] == "matched_partial"),
        "unmatched_missing_ct": sum(1 for row in match_rows if row["match_status"] == "unmatched_missing_ct"),
        "invalid_no_references": sum(1 for row in match_rows if row["match_status"] == "invalid_no_references"),
        "multiple_ct_series_reference_count": sum(1 for row in match_rows if row["multiple_ct_series_reference"] == "true"),
    }
    _write_summary(summary_md, summary)

    return ROIMatchResult(
        output_dir=output_dir,
        match_table_csv=match_table_csv,
        summary_md=summary_md,
        summary=summary,
    )


def _build_ct_index(inventory_csv: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(inventory_csv)
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("classification") != "CT":
            continue
        sop_uid = row.get("sop_instance_uid", "").strip()
        if sop_uid:
            index[sop_uid] = row
    return index


def _match_seg_row(seg_row: dict[str, str], ct_index: dict[str, dict[str, str]]) -> dict[str, str]:
    references = _split_references(seg_row.get("referenced_sop_instance_uids", ""))
    matched_ct_rows = [ct_index[reference] for reference in references if reference in ct_index]
    matched_count = len(matched_ct_rows)
    referenced_count = len(references)

    if referenced_count == 0:
        match_status = "invalid_no_references"
    elif matched_count == referenced_count:
        match_status = "matched_strict"
    elif matched_count > 0:
        match_status = "matched_partial"
    else:
        match_status = "unmatched_missing_ct"

    study_uids = sorted({row.get("study_instance_uid", "") for row in matched_ct_rows if row.get("study_instance_uid", "")})
    series_uids = sorted({row.get("series_instance_uid", "") for row in matched_ct_rows if row.get("series_instance_uid", "")})

    return {
        "seg_file_path": seg_row.get("file_path", ""),
        "seg_sop_instance_uid": seg_row.get("sop_instance_uid", ""),
        "seg_patient_id": seg_row.get("patient_id", ""),
        "seg_study_instance_uid": seg_row.get("study_instance_uid", ""),
        "seg_series_instance_uid": seg_row.get("series_instance_uid", ""),
        "referenced_sop_instance_uids": "|".join(references),
        "referenced_sop_count": str(referenced_count),
        "matched_sop_count": str(matched_count),
        "missing_sop_count": str(referenced_count - matched_count),
        "match_status": match_status,
        "matched_ct_study_instance_uids": "|".join(study_uids),
        "matched_ct_series_instance_uids": "|".join(series_uids),
        "multiple_ct_series_reference": "true" if len(series_uids) > 1 else "false",
    }


def _split_references(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_match_table(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "seg_file_path",
        "seg_sop_instance_uid",
        "seg_patient_id",
        "seg_study_instance_uid",
        "seg_series_instance_uid",
        "referenced_sop_instance_uids",
        "referenced_sop_count",
        "matched_sop_count",
        "missing_sop_count",
        "match_status",
        "matched_ct_study_instance_uids",
        "matched_ct_series_instance_uids",
        "multiple_ct_series_reference",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, summary: dict[str, int]) -> None:
    lines = [
        "# CT/SEG Match Summary",
        "",
        f"- number of SEG objects: {summary['seg_objects']}",
        f"- strict matches: {summary['strict_matches']}",
        f"- partial matches: {summary['partial_matches']}",
        f"- unmatched missing CT: {summary['unmatched_missing_ct']}",
        f"- invalid/no references: {summary['invalid_no_references']}",
        f"- multiple CT series reference count: {summary['multiple_ct_series_reference_count']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required audit input not found: {path}")
