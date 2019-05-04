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
