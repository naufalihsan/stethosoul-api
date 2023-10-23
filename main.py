from fastapi import FastAPI

from core import PsychosisModel
from models import Session
from utils import (
    get_prodromal_label,
    get_interview_label,
    get_dass_label,
    get_stai_label,
    calculate_risk,
)

app = FastAPI()

model = PsychosisModel("naufalihsan/psychosis_multi_class")

@app.post("/")
def predict(session: Session):
    response = {}
    
    prodromal_label = get_prodromal_label(session.pqb)
    response['pqb'] = prodromal_label

    interview_label = get_interview_label(session.interview, model=model)
    response['interview'] = interview_label

    risk = calculate_risk(prodromal_label, interview_label)
    response['risk'] = risk

    if session.dass:
        dass_label = get_dass_label(session.dass)
        response['dass'] = dass_label
    
    if session.stai:
        stai_label = get_stai_label(session.stai)
        response['stai'] = stai_label

    return response