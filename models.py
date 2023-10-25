from typing import Optional, Union
from pydantic import BaseModel


class PANSS(BaseModel):
    p1: int = 1
    p2: int = 1
    p3: int = 1
    n4: int = 1
    n6: int = 1
    g5: int = 1
    g9: int = 1


class PQB(BaseModel):
    id: int
    total_score: int = 0
    distress_score: int = 0


class DASS(BaseModel):
    anxiety: int = 0
    depression: int = 0
    stress: int = 0


class STAI(BaseModel):
    state: int = 0
    trait: int = 0


class Interview(BaseModel):
    text: str
    topic: Union[int, None]


class Session(BaseModel):
    pqb: list[PQB]
    interview: list[Interview]
    dass: Optional[DASS] = None
    stai: Optional[STAI] = None
    panss: Optional[PANSS] = None
