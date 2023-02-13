from .adv_loss import adv
from .coral import CORAL
from .cos import cosine
from .kl_js import kl_div, js
from .mmd import MMD_loss
from .mutual_info import Mine
from .pair_dist import pairwise_dist
from .multi_task_loss import mttl

__all__ = [
    'adv',
    'CORAL',
    'cosine',
    'kl_div',
    'js'
    'MMD_loss',
    'Mine',
    'pairwise_dist'
]