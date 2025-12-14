from .DQN import DQN
from .DDPG import DDPG

from .TempoRL import TempoRL
from .UTE import UTE
from .EQL import EQL

ALGO_REGISTRY = {
    "DQN": DQN,
    "DDPG": DDPG,
}

MODEL_REGISTRY = {
    "TempoRL": TempoRL,
    "UTE": UTE,
    "EQL": EQL,
}