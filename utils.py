from constants import DASS, STAI, PsychosisRisk

import pandas as pd

def calculate_risk(prodromal_label, interview_label):
    df = pd.DataFrame(interview_label)
    speech_risk = (df['predicted_class'] == 0).mean() < 0.51

    if speech_risk and prodromal_label:
        return PsychosisRisk.HIGH.name
    elif speech_risk or prodromal_label:
        return PsychosisRisk.MEDIUM.name
    else:
        return PsychosisRisk.LOW.name

def get_prodromal_label(pqb):
    df = pd.DataFrame([item.dict() for item in pqb])
    total_score = df['total_score'].sum()
    distress_score = df['distress_score'].sum()
    prodromal_risk = total_score > 7 and distress_score > 26
    
    return 1 if prodromal_risk else 0

def get_interview_label(interview, model=None):
    if not model:
        raise ValueError("model required")
    
    result = []
    for data in interview:
        result.append(model.predict(data))

    return result

def get_dass_label(dass) -> dict:
    return {
        'anxiety': get_dass_anxiety_label(dass.anxiety),
        'depression': get_dass_depression_label(dass.depression),
        'stress': get_dass_stress_label(dass.stress)
    }

def get_stai_label(stai) -> dict:
    return {
        'state': get_stai_item_label(stai.state),
        'trait': get_stai_item_label(stai.trait),
    }

def get_dass_depression_label(depression_score: int) -> DASS:
    if depression_score <= 4:
        return DASS.NORMAL
    elif depression_score >= 5 and depression_score <= 6:
        return DASS.MILD
    elif depression_score >= 7 and depression_score <= 10:
        return DASS.MODERATE
    elif depression_score >= 11 and depression_score <= 13:
        return DASS.SEVERE
    else:
        return DASS.EXTREMELY_SEVERE
    
def get_dass_anxiety_label(anxiety_score: int) -> DASS:
    if anxiety_score <= 3:
        return DASS.NORMAL
    elif anxiety_score >= 4 and anxiety_score <= 5:
        return DASS.MILD
    elif anxiety_score >= 6 and anxiety_score <= 7:
        return DASS.MODERATE
    elif anxiety_score >= 8 and anxiety_score <= 9:
        return DASS.SEVERE
    else:
        return DASS.EXTREMELY_SEVERE
    
def get_dass_stress_label(stress_score: int) -> DASS:
    if stress_score <= 3:
        return DASS.NORMAL
    elif stress_score >= 4 and stress_score <= 5:
        return DASS.MILD
    elif stress_score >= 6 and stress_score <= 7:
        return DASS.MODERATE
    elif stress_score >= 8 and stress_score <= 9:
        return DASS.SEVERE
    else:
        return DASS.EXTREMELY_SEVERE
    
def get_stai_item_label(stai_score: int) -> STAI:
    if stai_score <= 37:
        return STAI.LOW
    elif stai_score >= 38 and stai_score <= 44:
        return STAI.MODERATE
    else:
        return STAI.HIGH