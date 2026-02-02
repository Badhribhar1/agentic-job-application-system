import json
from pathlib import Path
from config.loader import load_settings
from typing import List, Tuple
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.job import JobPosting
from models.candidate import CandidateProfile
from models.results import RankedJob


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NORMALIZED_JOBS_PATH = PROCESSED_DIR / "jobs_normalized.json"
RANKED_JSON_PATH = PROCESSED_DIR / "ranked_jobs.json"
RANKED_CSV_PATH = PROCESSED_DIR / "ranked_jobs.csv"

RESUME_PATH = PROJECT_ROOT / "data" / "raw" / "resume.txt"


def load_resume_text(path: Path = RESUME_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_normalized_jobs(path: Path = NORMALIZED_JOBS_PATH) -> List[JobPosting]:
    if not path.exists():
        raise FileNotFoundError(f"Normalized jobs not found: {path}. Run ingestion first.")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [JobPosting(**row) for row in data]


# USE FOR TESTING PURPOSES
# def default_candidate_profile(resume_text: str) -> CandidateProfile:
#  
#     return CandidateProfile(
#         resume_text=resume_text,
#         target_roles=[
#             "Data Engineer",
#             "Applied AI Engineer",
#             "Forward Deployed Engineer",
#             "Solutions Engineer",
#             "Technical Consultant",
#             "Analytics Engineer",
#         ],
#         must_haves=["python", "sql", "cloud", "pipelines", "data"],
#         dealbreakers=["phd required", "publish papers", "research-only", "c++ only"],
#         preferred_locations=["remote", "chicago", "boston", "new york"],
#     )


def tfidf_similarity(query: str, docs: List[str]) -> List[float]:
    """
    Returns cosine similarity(query, each doc) using TF-IDF.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=4000)
    matrix = vectorizer.fit_transform([query] + docs)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return sims.tolist()


def score_job(profile: CandidateProfile, job: JobPosting, base_sim: float, weights: dict, penalties: dict) -> Tuple[float, List[str]]:
    
    reasons: List[str] = []

    # Weighted base similarity
    semantic_w = float(weights.get("semantic", 1.0))
    score = base_sim * semantic_w

    title_lower = job.title.lower()
    desc_lower = job.description.lower()

    # Title match boost
    title_boost = float(weights.get("title_match", 0.0))
    for role in profile.target_roles:
        if role.lower() in title_lower:
            score += title_boost
            reasons.append(f"title matches target role: {role}")
            break

    # Must-have boost (scaled by hits)
    must_have_w = float(weights.get("must_have", 0.0))
    hits = 0
    matched = []
    for kw in profile.must_haves:
        if kw.lower() in desc_lower:
            hits += 1
            matched.append(kw)

    if hits:
        # Normalize: 1.0 means you hit all must-haves (cap at 1.0)
        frac = min(1.0, hits / max(1, len(profile.must_haves)))
        score += must_have_w * frac
        reasons.append(f"must-have keywords hit: {hits} ({', '.join(matched[:4])})")

    # Dealbreaker penalty
    dealbreaker_pen = float(penalties.get("dealbreaker", 0.0))
    for bad in profile.dealbreakers:
        if bad.lower() in desc_lower:
            score -= dealbreaker_pen
            reasons.append(f"dealbreaker detected: {bad}")
            break

    # Optional: location/remote preference boost
    loc_w = float(weights.get("location", 0.0))
    if loc_w and profile.preferred_locations:
        loc_text = f"{job.location} {'remote' if job.remote else ''}".lower()
        if any(loc.lower() in loc_text for loc in profile.preferred_locations):
            score += loc_w
            reasons.append("matches location/remote preference")

    # Clamp
    score = max(0.0, min(1.0, score))
    return score, reasons


def save_ranked_outputs(ranked: List[RankedJob]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    payload = [r.model_dump() for r in ranked]  # pydantic v2
    with RANKED_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # CSV
    df = pd.DataFrame(payload)
    df.to_csv(RANKED_CSV_PATH, index=False)

    print(f"[ranking] Saved ranked jobs -> {RANKED_JSON_PATH}")
    print(f"[ranking] Saved ranked jobs CSV -> {RANKED_CSV_PATH}")
    print("[ranking] Top 5:")
    for r in ranked[:5]:
        print(f"  - {r.score:.3f} | {r.title} @ {r.company}")



def rank_jobs(
    profile: CandidateProfile,
    jobs: List[JobPosting],
    weights: dict,
    penalties: dict
) -> List[RankedJob]:
    docs = [j.description for j in jobs]
    sims = tfidf_similarity(profile.resume_text, docs)

    ranked: List[RankedJob] = []
    for job, sim in zip(jobs, sims):
        score, reasons = score_job(profile, job, sim, weights, penalties)
        ranked.append(
            RankedJob(
                job_id=job.job_id,
                title=job.title,
                company=job.company,
                score=round(score, 4),
                reasons=reasons,
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def build_candidate_profile_from_settings(resume_text: str, settings: dict) -> CandidateProfile:
    return CandidateProfile(
        resume_text=resume_text,
        target_roles=settings["candidate"]["target_roles"],
        must_haves=settings["candidate"].get("must_haves", []),
        dealbreakers=settings["candidate"].get("dealbreakers", []),
        preferred_locations=settings.get("preferences", {}).get("preferred_locations", []),
    )



def run_ranking() -> List[RankedJob]:
    settings = load_settings()

    weights = settings.get("scoring", {}).get("weights", {})
    penalties = settings.get("scoring", {}).get("penalties", {})    

    resume_text = load_resume_text()
    profile = build_candidate_profile_from_settings(resume_text, settings)

    jobs = load_normalized_jobs()

    ranked = rank_jobs(profile, jobs, weights, penalties)
    save_ranked_outputs(ranked)
    return ranked

