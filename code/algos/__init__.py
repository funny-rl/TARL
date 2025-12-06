from .DQN import DQN

from .TempoRL import TempoRL
from .UTE import UTE

ALGO_REGISTRY = {
    "DQN": DQN,
}

MODEL_REGISTRY = {
    "TempoRL": TempoRL,
    "UTE": UTE,
}