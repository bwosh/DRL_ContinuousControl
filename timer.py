import time

class Timer:
    def __init__(self, name):
        self.start = time.time()
        self.name = name

    def finish(self):
        stop = time.time()
        diff = stop- self.start

        #print(f"{self.name}: {diff:.3f} seconds")
        # TODO remove class?