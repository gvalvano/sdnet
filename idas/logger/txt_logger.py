#  Copyright 2019 Gabriele Valvano
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging

DEBUG_LEVEL_NUM = 45  # custom level for log (s.t. logging.ERROR < 45 < logging.CRITICAL)
logging.addLevelName(DEBUG_LEVEL_NUM, "logger")


def parse_log_file(filename='log_file.txt'):
    """ utility to read log file """
    # Example to read the last line of a file and split it in half with separator sep
    sep = '='

    with open(filename, 'r') as f:
        last_line = f.readlines()[-1]
    s1, s2 = last_line.rsplit(sep)
    return s1, s2


def get_logger(log_file, lvl=DEBUG_LEVEL_NUM):
    """ utility for logger definition, with level lvl """
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(log_file[:-4])
    logger.setLevel(lvl)
    logger.addHandler(handler)

    return logger
