from pydantic import BaseModel
from typing import List, Dict, Optional

class RankedJob(BaseModel):
    job_id: str
    title: str
    company: str
    score: float
    reasons: List[str] = []
    subscores: Optional[Dict[str, float]] = None