import enum

class PsychosisRisk(enum.Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class DASS(enum.Enum):
    NORMAL = 1
    MILD = 2
    MODERATE = 3
    SEVERE = 4
    EXTREMELY_SEVERE = 5

class STAI(enum.Enum):
    LOW = 1
    MODERATE = 2
    HIGH = 3