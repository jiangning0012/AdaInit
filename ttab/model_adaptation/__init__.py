# -*- coding: utf-8 -*-
from .cotta import CoTTA
from .no_adaptation import NoAdaptation
from .sar import SAR
from .tent import TENT
from .come import COME
from .adadem import AdaDEM
from .sar2 import SAR2
from .nctta import NCTTA
from .adainit import AdaInit


def get_model_adaptation_method(adaptation_name):
    return {
        "no_adaptation": NoAdaptation,
        "tent": TENT,
        "sar": SAR,
        "cotta": CoTTA,
        "come": COME,
        "adadem": AdaDEM,
        "sar2": SAR2,
        "nctta": NCTTA,
        "adainit": AdaInit,
    }[adaptation_name]
