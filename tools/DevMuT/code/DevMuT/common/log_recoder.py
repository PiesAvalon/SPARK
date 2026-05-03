import logging
import os
from logging.handlers import TimedRotatingFileHandler


def singleton(cls):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton


@singleton
class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self, level='info', log_file=None):
        self.logger = logging.getLogger()
        log_format = '%(levelname)s %(asctime)s - %(filename)s:%(lineno)s - %(message)s'
        data_format = '%Y-%m-%d %H:%M:%S'
        format_str = logging.Formatter(log_format, data_format)
        self.logger.setLevel(self.level_relations.get(level))

        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(format_str)
            self.logger.addHandler(sh)
            fh = TimedRotatingFileHandler(log_file, when='D', interval=1, backupCount=10)
            fh.setFormatter(format_str)
            self.logger.addHandler(fh)

        if log_file:
            if os.path.exists(log_file):
                os.remove(log_file)
        fh = TimedRotatingFileHandler(log_file, when='D', interval=1, backupCount=10)
        fh.setFormatter(format_str)
        self.logger.addHandler(fh)
