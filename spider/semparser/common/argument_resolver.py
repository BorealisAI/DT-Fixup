# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import traceback
from copy import deepcopy

from semparser.common import registry

REGISTERED_NAME = "REGISTERED_NAME"
REGISTERED_KIND = "REGISTERED_KIND"
CALL_REQUIRED = "CALL_REQUIRED"
GIT_COMMIT = "GIT_COMMIT"


def resolve_argument(argument_dict, caller=None):
    argument_dict = deepcopy(argument_dict)
    logger = logging.getLogger()

    if caller is None:
        caller_kind = argument_dict[REGISTERED_KIND]
        caller_name = argument_dict[REGISTERED_NAME]
        call_required = bool(argument_dict[CALL_REQUIRED]) if CALL_REQUIRED in argument_dict else False

        caller = registry.lookup(caller_kind, caller_name)
        argument_dict.pop(REGISTERED_NAME)
        argument_dict.pop(REGISTERED_KIND)
        if CALL_REQUIRED in argument_dict:
            argument_dict.pop(CALL_REQUIRED)

        if call_required:
            return resolve_argument(argument_dict, caller)
        else:
            return caller

    resolved_arguments = dict()
    for argument_name, argument_value in argument_dict.items():
        if argument_value is not None and isinstance(argument_value, dict):
            if REGISTERED_NAME in argument_value and REGISTERED_KIND in argument_value:
                try:
                    resolved_arguments[argument_name] = resolve_argument(argument_value)
                except Exception as ex:
                    logger.error("Failed to resolve caller: %s" % str(argument_value))
                    logger.error(traceback.format_stack())
                    logger.error(traceback.format_exc())
                    raise ex
            else:
                resolved_arguments[argument_name] = argument_value
        else:
            resolved_arguments[argument_name] = argument_value

    return caller(**resolved_arguments)


def update_argument(argument_dict, command_list):
    """
    Each str in command_list should be in the form of argument/path:argument_val.
    @param argument_dict: dict
    @param command_list: list(str)
    """
    logger = logging.getLogger()

    for command in command_list:
        try:
            argument_path, argument_val = command.split(':')

            argument_segments = argument_path.split('/')
            sub_arg_dict = argument_dict

            if len(argument_segments) == 0:
                continue

            while len(argument_segments) > 1:
                argument_segment = argument_segments[0]
                sub_arg_dict = sub_arg_dict[argument_segment]
                argument_segments.pop(0)

            old_val = sub_arg_dict[argument_segments[0]]
            val_type = type(old_val)
            if isinstance(old_val, bool) and argument_val.lower() == 'false':
                argument_val = False
            sub_arg_dict[argument_segments[0]] = val_type(argument_val)

            logger.info("Argument update: %s is changed from %s to %s" % (argument_path, str(old_val), argument_val))
        except Exception as ex:
            logger.error("Failed to resolve command input: %s" % str(command))
            logger.error(traceback.format_stack())
            logger.error(traceback.format_exc())
            continue


def build_dependency_graph(arguments, edges, prefix_path=()):

    if isinstance(arguments, list):
        for idx, arg in enumerate(arguments):
            prefix_path_ = prefix_path + (idx,)
            build_dependency_graph(arg, edges, prefix_path=prefix_path_)
        return

    if isinstance(arguments, dict):
        for key_, value_ in arguments.items():
            prefix_path_ = prefix_path + (key_,)
            build_dependency_graph(value_, edges, prefix_path=prefix_path_)

    if not isinstance(arguments, str):
        return

    # str
    if arguments.startswith('{') and arguments.endswith('}'):
        value_path = arguments[1:-1]
        chain_list = value_path.split('/')

        edges[prefix_path] = tuple(chain_list)


def resolve_dependencies(argument_dict):

    edges = dict()
    build_dependency_graph(argument_dict, edges)

    # find replacement value
    dependency = dict()
    for src, tgt in edges.items():
        visited = set()
        visited.add(src)

        while tgt in edges:
            if tgt in visited:
                raise ValueError("Unable to resolve cyclic argument dependencies!")
            visited.add(tgt)
            tgt = edges[tgt]

        dependency[src] = tgt

    # replace value
    for src, tgt in dependency.items():

        src_dict = argument_dict
        src_idx = 0
        src_cur = src[src_idx]
        while src_idx < len(src) - 1:
            src_dict = src_dict[src_cur]
            src_idx += 1
            src_cur = src[src_idx]

        tgt_val = argument_dict
        for tgt_cur in tgt:
            tgt_val = tgt_val[tgt_cur]

        src_dict[src_cur] = tgt_val


def insert_git_commit_hash(argument_dict, git_hash):
    argument_dict[GIT_COMMIT] = git_hash
