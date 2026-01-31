from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
import hashlib

class JobPosting(BaseModel):
    job_id: str
    title: str
    company: str
    location: str = "unknown"
    remote: Optional[bool] = None
    description: str
    url: str = ""
    source: str = "manual"

    @staticmethod
    def make_job_id(company: str, title: str, url: str) -> str:
        base = f"{company.strip().lower()}|{title.strip().lower()}|{url.strip().lower()}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]