# -*- coding: utf-8 -*-

from .last_iterate import LastIterate


def get_model_selection_method(selection_name):
    return {
        "last_iterate": LastIterate,
    }[selection_name]
