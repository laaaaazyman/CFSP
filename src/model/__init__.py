from .WordSplitting.model import WSModel
from .ArgumentIdentification.model import AIModel
from .FrameIdentification.model import FIModel
from .RoleIdentification.model import RIModel




MODEL_MAP = {
    "WS": WSModel,
    "FI": FIModel,
    "AI": AIModel,
    "RI": RIModel 
}


