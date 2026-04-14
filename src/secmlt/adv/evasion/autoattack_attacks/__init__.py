"""AutoAttack-based evasion wrappers."""

import importlib.util

from .autoattack_base import BaseAutoAttack

if importlib.util.find_spec("autoattack", None) is not None:
    from .autoattack_apgd import *  # noqa: F403
    from .autoattack_standard import *  # noqa: F403
