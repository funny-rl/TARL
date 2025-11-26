from .DQN import DQN

from .TempoRL import TempoRL

ALGO_REGISTRY = {
    "DQN": DQN,
}

MODEL_REGISTRY = {
    "TempoRL": TempoRL,
}