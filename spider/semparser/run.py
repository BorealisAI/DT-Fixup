# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from semparser.common import argument_resolver, log_wrapper, utils
import traceback
import os
import torch
import yaml
from yaml import Loader
import time
import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES'] = 0


def create_logger(exp_dir):
    log_path = os.path.join(exp_dir, 'logs')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    logger = log_wrapper.create_logger(to_disk=True, log_file=log_path)
    return logger


def create_parser():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--config_path', required=True, type=str,
                        help='path to the experiment config file.')
    parser.add_argument('--commit', required=True, type=str,
                        help='Git commit hash, which guarantees the reproducibility of the experiment. ')

    # Components
    parser.add_argument('--do_preprocess', default=False, action='store_true',
                        help='enable the preprocessor')
    parser.add_argument('--do_training', default=False, action='store_true',
                        help='enable the trainer')
    parser.add_argument('--do_evaluation', default=False, action='store_true',
                        help='enable the evaluator')

    # Others
    parser.add_argument(
        '-p', '--auto_path', default=False, action='store_true',
        help='generate experiment directory automatically, otherwise use the EXP_DIR specified in config file. '
             'The default experiment path is ./data/tmp/{timestamp}.'
    )
    parser.add_argument(
        '-d', '--description', required=False, type=str,
        help='Add a description in experiment config file.'
    )

    return parser


if __name__ == "__main__":
    # register all the components
    utils.import_module_and_submodules('semparser')

    # parse arguments
    parser = create_parser()
    args, unknown = parser.parse_known_args()
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

    # load config file
    config_path = args.config_path
    with open(config_path, 'r') as fp:
        config = yaml.load(fp, Loader=Loader)
    # update config file according to command input
    argument_resolver.update_argument(config, unknown)

    # insert commit hash
    argument_resolver.insert_git_commit_hash(config, args.commit)

    # auto path
    if args.auto_path:
        config["EXP_DIR"] = os.path.join(
            'data',
            '/'.join(['tmp', time.strftime("%Y%m%d-%H%M%S")])
        )

    # experiment description
    if hasattr(args, 'description') and args.description is not None:
        config['DESCRIPTION'] = args.description

    if not os.path.exists(config["EXP_DIR"]):
        os.makedirs(config["EXP_DIR"])

    # logger
    logger = create_logger(config["EXP_DIR"])

    # save config file to experiment directory
    with open(os.path.join(config["EXP_DIR"], 'config.yml'), 'w') as fp:
        yaml.dump(config, fp)

    # resolve variables
    argument_resolver.resolve_dependencies(config)

    try:
        if args.do_preprocess:
            argument_resolver.resolve_argument(config['PREPROCESSOR'])

        if args.do_training or args.do_evaluation:
            # create trainer
            exp_runner = argument_resolver.resolve_argument(config["EXP_RUNNER"])

            # create model
            model = argument_resolver.resolve_argument(config["MODEL"])

            if args.do_training:
                exp_runner.run(model)

            if args.do_evaluation:
                model.load(os.path.join(config["EXP_DIR"], "model.bin"), map_location=torch.device(exp_runner.device))
                if exp_runner.device == 'cuda':
                    model = model.cuda()

                exp_runner.evaluate(model, exp_runner.dev_data, save_failed_samples=True)
    except Exception as ex:
        logger.error(traceback.format_stack())
        logger.error(traceback.format_exc())
        raise ex
