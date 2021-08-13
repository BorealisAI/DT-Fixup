# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) Microsoft.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#################################################################################################
# Original code is based on https://github.com/namisan/mt-dnn/blob/master/data_utils/log_wrapper.py by Microsoft.
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#################################################################################################

import logging
import sys
from time import gmtime, strftime


def create_logger(name=None, silent=False, to_disk=False, log_file=None):
    """Logger wrapper
    """
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
    )
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file1 = (
            log_file
            if log_file is not None
            else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        )
        fh = logging.FileHandler(log_file1)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)

        log_file2 = (
            log_file + ".error"
            if log_file is not None
            else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        )
        fh = logging.FileHandler(log_file2)
        fh.setLevel(logging.ERROR)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log
