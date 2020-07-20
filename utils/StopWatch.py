import time
# useful tool for measuring runtime


class StopWatch:
    def __init__(self):
        self.beginning = 0

    def start(self):
        self.beginning = time.time()

    def stop(self):
        end = time.time()
        print(str(round(end - self.beginning, 1)) + "s")