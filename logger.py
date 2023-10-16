import sys


class Logger(object):
    def __init__(self, log_path) -> None:
        self.termimal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = "w"
        self.file = open(file, mode)

    def write(self, message, terminal=True, file=True):
        if terminal is True:
            self.termimal.write(message)
            self.termimal.flush()
        if file is True:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        pass
