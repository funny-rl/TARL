

from .ZigZag import ZigZag
from .Bridge import Bridge
from .CliffWalking import CliffWalking

ENVS_REGISTRY = {
    "CliffWalking": CliffWalking,
    "ZigZag": ZigZag,
    "Bridge": Bridge,
}