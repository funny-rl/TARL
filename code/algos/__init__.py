from .DQN import DQN
from .DDPG import DDPG
from .Random import Random

from .TempoRL import TempoRL
from .UTE import UTE
from .EQL import EQL


ALGO_REGISTRY = {
    "DQN": DQN,
    "DDPG": DDPG,
    "Random": Random,
}

MODEL_REGISTRY = {
    "TempoRL": TempoRL,
    "UTE": UTE,
    "EQL": EQL,
}