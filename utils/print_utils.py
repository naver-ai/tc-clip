"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import sys
import json
from omegaconf import OmegaConf


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def print_configs(logger, configs):
    selected_options = {k: v for k, v in configs.selected_option.items() if 'hydra' not in k}
    overriden_values = configs.overriden_values['task']
    logger.info(colorstr("Selected Options"))
    for key, val in selected_options.items():
        logger.info(f"\t-{key}: {val}")
    logger.info(colorstr("Overrides"))
    for override in overriden_values:
        key, value = override.split("=")
        logger.info(f"\t-{key}: {value}")
    logger.info("")


def print_configs_all(logger, configs):
    cfg_dict = OmegaConf.to_container(configs, resolve=True)
    logger.info(json.dumps(cfg_dict, indent=4))


def disable_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__
