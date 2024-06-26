import os
from datetime import datetime

class FileLogger:
    def __init__(self, logdir):
        self.logdir = logdir
        # Create the directory if it does not exist
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logdir, f"log_{timestamp}.txt")
        with open(log_file, "w") as f: # open for writing to clear the file
            pass
        self.log_file = log_file

    def log_info(self, text):
        with open(self.log_file, "a") as f:
            f.write(text + "\n")