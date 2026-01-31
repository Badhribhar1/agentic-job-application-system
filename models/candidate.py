from pydantic import BaseModel
from typing import List

class CandidateProfile(BaseModel):
    resume_text: str
    target_roles: List[str]
    must_haves: List[str] = []
    dealbreakers = List[str] = []
    preferred_locations: List[str] = []