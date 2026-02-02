import json
from pathlib import Path
from typing import List, Dict, Any
from models.job import JobPosting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_JOBS_PATH = PROJECT_ROOT / "data" / "raw" / "job_sample.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NORMALIZED_JOBS_PATH = PROCESSED_DIR / "jobs_normalized.json"

def load_raw_jobs(path: Path = RAW_JOBS_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Raw jobs file not found: {path}")
    with path.open("r", encoding ="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("job_sample.json must contain a JSON list of job objects")
    return data

def normalize_job(raw: Dict[str, Any]) -> JobPosting:
    """
    Convert untrusted raw dict -> trusted JobPosting.
    Fill defaults and generate job_id.
    """
    title = (raw.get("title") or "").strip()
    company = (raw.get("company") or "").strip()
    url = (raw.get("url") or "").strip()

    if not title or not company:
        raise ValueError(f"Job missing required fields title/company: {raw}")

    job_id = JobPosting.make_job_id(company=company, title=title, url=url)

    # Build normalized object
    job = JobPosting(
        job_id=job_id,
        title=title,
        company=company,
        location=(raw.get("location") or "Unknown").strip(),
        remote=raw.get("remote", None),
        description=(raw.get("description") or "").strip(),
        url=url,
        source=(raw.get("source") or "manual").strip(),
    )

    if not job.description:
        raise ValueError(f"Job missing description: {title} @ {company}")

    return job


def normalize_jobs(raw_jobs: List[Dict[str, Any]]) -> List[JobPosting]:
    normalized: List[JobPosting] = []
    for raw in raw_jobs:
        try:
            normalized.append(normalize_job(raw))
        except Exception as e:
            print(f"[ingestion] Skipping invalid job row: {e}")
    if not normalized:
        raise ValueError("No valid jobs found after normalization.")
    return normalized


def save_normalized_jobs(jobs: List[JobPosting], path: Path = NORMALIZED_JOBS_PATH) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    payload = [j.model_dump() for j in jobs]  # pydantic v2
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[ingestion] Saved {len(jobs)} normalized jobs -> {path}")


def ingestion() -> List[JobPosting]:
    raw = load_raw_jobs()
    jobs = normalize_jobs(raw)
    save_normalized_jobs(jobs)
    return jobs