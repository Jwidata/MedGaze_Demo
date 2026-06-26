"""Standalone HTML report generation for guided gaze review."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd


def write_guided_narrations(path: Path, narrations: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(narrations, indent=2) + "\n", encoding="utf-8")


def write_interactive_report(path: Path, selected_cases: pd.DataFrame, narrations: list[dict[str, object]], output_dir: Path) -> None:
    narration_by_session = {str(item["session_id"]): item for item in narrations}
    sections = []
    for label, group in selected_cases.groupby("hidden_behavior_label", sort=False):
        cards = []
        for _, row in group.iterrows():
            session_id = str(row["session_id"])
            case_dir = output_dir / str(label)
            images = {
                "canvas": case_dir / f"{session_id}_canvas.png",
                "roi": case_dir / f"{session_id}_roi_overlay.png",
                "gaze": case_dir / f"{session_id}_gaze_points.png",
                "heatmap": case_dir / f"{session_id}_heatmap_overlay.png",
                "scanpath": case_dir / f"{session_id}_scanpath_overlay.png",
                "combined": case_dir / f"{session_id}_combined_overlay.png",
            }
            cards.append(_case_card(row, narration_by_session.get(session_id, {}), images))
        sections.append(f"<section><h2>{html.escape(str(label))}</h2>{''.join(cards)}</section>")
    document = _html_document("\n".join(sections))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def _case_card(row: pd.Series, narration: dict[str, object], images: dict[str, Path]) -> str:
    session_id = html.escape(str(row["session_id"]))
    card_id = "case_" + "".join(ch if ch.isalnum() else "_" for ch in session_id)
    image_tags = "".join(
        f'<img class="thumb" src="{_data_uri(path)}" alt="{html.escape(name)} overlay">'
        for name, path in images.items()
    )
    layered = "".join(
        [
            f'<img class="layer base" src="{_data_uri(images["canvas"])}" alt="placeholder canvas">',
            f'<img class="layer layer-roi" src="{_data_uri(images["roi"])}" alt="ROI overlay">',
            f'<img class="layer layer-heatmap" src="{_data_uri(images["heatmap"])}" alt="heatmap overlay">',
            f'<img class="layer layer-gaze" src="{_data_uri(images["gaze"])}" alt="gaze points overlay">',
            f'<img class="layer layer-scanpath" src="{_data_uri(images["scanpath"])}" alt="scanpath overlay">',
        ]
    )
    return f"""
    <article class="case-card" id="{card_id}">
      <div class="case-header">
        <h3>{session_id}</h3>
        <p>Case: {html.escape(str(row.get('case_id', '')))} | ROI: {html.escape(str(row.get('roi_id', '')))}</p>
      </div>
      <div class="controls">
        <button onclick="toggleLayer('{card_id}', 'layer-roi')">Show ROI</button>
        <button onclick="toggleLayer('{card_id}', 'layer-heatmap')">Show heatmap</button>
        <button onclick="toggleLayer('{card_id}', 'layer-gaze')">Show gaze points</button>
        <button onclick="toggleLayer('{card_id}', 'layer-scanpath')">Show scanpath</button>
      </div>
      <div class="case-body">
        <div class="viewer">{layered}</div>
        <aside class="narration">
          <h4>Guided Narration</h4>
          <p><strong>Behavior label:</strong> {html.escape(str(narration.get('behavior_label', row.get('hidden_behavior_label', ''))))}</p>
          <p><strong>Rule attention status:</strong> {html.escape(str(narration.get('rule_attention_status', row.get('rule_attention_status', 'unknown'))))}</p>
          <p><strong>Cognitive-load proxy:</strong> {html.escape(str(narration.get('cognitive_load_proxy', row.get('cognitive_load_proxy', 'unknown'))))}</p>
          <p><strong>What the heatmap suggests:</strong> {html.escape(str(narration.get('heatmap_suggestion', 'The heatmap summarizes gaze density in image coordinates.')))}</p>
          <p><strong>Why the behavior prediction makes sense:</strong> {html.escape(str(narration.get('behavior_rationale', 'The rendered overlays provide behavior-level evidence.')))}</p>
          <p><strong>Limitation note:</strong> {html.escape(str(narration.get('limitation_note', 'Real Tobii validation remains future work.')))}</p>
        </aside>
      </div>
      <div class="thumbs">{image_tags}</div>
    </article>
    """


def _html_document(body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MedGazeAR Interactive Gaze Review</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #101319; color: #eef2f7; }}
    header {{ padding: 24px 32px; background: #1a2332; border-bottom: 1px solid #31415c; }}
    main {{ padding: 24px 32px; }}
    h1, h2, h3 {{ margin-top: 0; }}
    section {{ margin-bottom: 36px; }}
    .case-card {{ background: #171d28; border: 1px solid #2d3a50; border-radius: 12px; padding: 18px; margin: 18px 0; }}
    .case-header p {{ color: #aeb9ca; word-break: break-all; }}
    .controls button {{ margin: 0 8px 12px 0; padding: 8px 12px; background: #28476f; color: #fff; border: 0; border-radius: 6px; cursor: pointer; }}
    .controls button:hover {{ background: #386294; }}
    .case-body {{ display: flex; gap: 20px; flex-wrap: wrap; }}
    .viewer {{ position: relative; width: min(512px, 90vw); aspect-ratio: 1 / 1; background: #222; border: 1px solid #45536a; }}
    .layer {{ position: absolute; left: 0; top: 0; width: 100%; height: 100%; object-fit: contain; }}
    .narration {{ max-width: 520px; line-height: 1.45; color: #dce5f2; }}
    .thumbs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-top: 14px; }}
    .thumb {{ width: 100%; border: 1px solid #3b4a64; border-radius: 8px; background: #222; }}
    .hidden-layer {{ display: none; }}
    code {{ color: #a9e6ff; }}
  </style>
</head>
<body>
  <header>
    <h1>MedGazeAR Interactive Gaze Review</h1>
    <p>This standalone report uses a source-agnostic visualization schema. Synthetic gaze and future Tobii gaze use the same coordinate and rendering pipeline.</p>
    <p>Heatmaps are generated in image coordinates. Exact real Tobii validation is future work; synthetic and real gaze should be compared by behavior-level features and heatmap similarity, not exact point matching.</p>
  </header>
  <main>{body}</main>
  <script>
    function toggleLayer(cardId, className) {{
      const card = document.getElementById(cardId);
      card.querySelectorAll('.' + className).forEach((el) => el.classList.toggle('hidden-layer'));
    }}
  </script>
</body>
</html>
"""


def _data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"
