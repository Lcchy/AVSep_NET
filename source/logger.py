import logging
from parameters import LOG, DISPLAY

class Logger_custom():
    """Set-up a logger quickly. Prints nut logs only the relevant.
    Didn't use inheritence form logging.Logger because of logging.setLoggerClass() doc"""
    def __init__(self, name, file="", level=logging.INFO):
        super().__init__()
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if len(file) > 0:
            handler = logging.FileHandler(file)
            # handler.terminator = ""
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        self.logger = logger

    def print_log(self, msg, end="\n", level=logging.INFO):
        if DISPLAY and level != logging.ERROR:
            print(msg, end=end)
        if LOG and level != logging.ERROR:
            #if end[-1:] == "\n":    # Considered as one character
            # Using ANSI sequence to go up one line as logging automatically adds \n
            self.logger.log(level, msg.replace("\n", ""))     #TODO inline ANSI for non \n
        elif level == logging.ERROR:
            self.logger.log(level, msg)