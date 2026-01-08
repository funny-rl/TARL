

from .ZigZag import ZigZag
from .Bridge import Bridge
from .CliffWalking import CliffWalking
from .ChainMDP import ChainMDP
from .Spiral import Spiral

ENVS_REGISTRY = {
    "CliffWalking": CliffWalking,
    "ZigZag": ZigZag,
    "Bridge": Bridge,
    "ChainMDP": ChainMDP,
    "Spiral": Spiral,
}