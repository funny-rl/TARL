from .DQN import DQN
from .DDPG import DDPG

from .TempoRL import TempoRL
from .UTE import UTE
from .GTA import GTA

ALGOS_REGISTRY = {
    "DQN": DQN,
    "DDPG": DDPG,
}
MODEL_REGISTRY = {
    "TempoRL": TempoRL,
    "UTE": UTE,
    "GTA": GTA,
}