from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def generate_markdown_report(
    ranked_payload: List[Dict[str, Any]],
    normalized_jobs_by_id: Dict[str, Dict[str, Any]],
    output_path: Path,
    top_n: int = 15,
) -> None:
    """
    ranked_payload: list of dicts from RankedJob (already JSON-safe)
    normalized_jobs_by_id: {job_id: normalized job dict including url/location/remote}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"# Job Ranking Report\n")
    lines.append(f"- Generated: **{now}**\n")
    lines.append(f"- Showing top **{top_n}** results\n")

    lines.append("\n---\n")
    lines.append("## Top Matches\n")

    lines.append("| Rank | Score | Title | Company | Location | Remote | Reasons | Link |")
    lines.append("|---:|---:|---|---|---|:---:|---|---|")

    for idx, item in enumerate(ranked_payload[:top_n], start=1):
        job_id = item.get("job_id", "")
        score = item.get("score", 0.0)
        title = item.get("title", "").replace("|", "\\|")
        company = item.get("company", "").replace("|", "\\|")

        job = normalized_jobs_by_id.get(job_id, {})
        location = str(job.get("location", "Unknown")).replace("|", "\\|")
        remote = job.get("remote", None)
        remote_str = "Y" if remote is True else ("N" if remote is False else "?")

        reasons = item.get("reasons", [])
        if isinstance(reasons, list):
            reasons_str = "; ".join(reasons[:2]).replace("|", "\\|")  # top 2 reasons
        else:
            reasons_str = str(reasons).replace("|", "\\|")

        url = job.get("url", "")
        link_str = url if url else ""

        lines.append(
            f"| {idx} | {score:.3f} | {title} | {company} | {location} | {remote_str} | {reasons_str} | {link_str} |"
        )

    lines.append("\n---\n")
    lines.append("### Notes\n")
    lines.append("- This report is generated from local ranking output.\n")
    lines.append("- Scores are relative and depend on scoring weights in `config/settings.yaml`.\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")