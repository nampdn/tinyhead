# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import random
import subprocess
import sys

from mmengine.logging import print_log

import tinyhead

from tinyhead.tools import (train)

# Define valid modes
MODES = ('train')

CLI_HELP_MSG = \
    f"""
    Arguments received: {str(['tinyhead'] + sys.argv[1:])}. tinyhead commands use the following syntax:

        tinyhead MODE MODE_ARGS ARGS

        Where   MODE (required) is one of {MODES}
                MODE_ARG (optional) is the argument for specific mode
                ARGS (optional) are the arguments for specific command

    Some usages for xtuner commands: (See more by using -h for specific command!)

        1. Pretrain SLMs:
            tinyhead train $CONFIG
        
    Run special commands:

        tinyhead help
        tinyhead version
    """  # noqa: E501

special = {
    'help': lambda: print_log(CLI_HELP_MSG, 'current'),
    'version': lambda: print_log(tinyhead.__version__, 'current')
}

special = {
    **special,
    **{f'-{k[0]}': v
       for k, v in special.items()},
    **{f'--{k}': v
       for k, v in special.items()}
}

modes = {
    'train': train.__file__
}

print(modes)

def cli():
    args = sys.argv[1:]
    print(args)
    if not args:  # no arguments passed
        print_log(CLI_HELP_MSG, 'current')
        return
    if args[0].lower() in special:
        special[args[0].lower()]()
        return
    elif args[0].lower() in modes:
        print("ABC XYZ")
    else:
        print_log('WARNING: command error!', 'current', logging.WARNING)
        print_log(CLI_HELP_MSG, 'current', logging.WARNING)
        return