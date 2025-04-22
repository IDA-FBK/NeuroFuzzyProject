import atexit
import sys

class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", buffering=1)
        # Ensure file closes on exit
        atexit.register(self.cleanup)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        if self.log and not self.log.closed:
            self.terminal.flush()
            self.log.flush()

    def cleanup(self):
        # Close file on program exit
        self.log.close()

def save_list_in_a_file(list_to_save, path_to_file):
    with open(path_to_file, "w") as f:
        for el_list in list_to_save:
            f.write(el_list + "\n")
            #print(el_list)
