class EpsilonGreedy(): # toDO remove ?
    def __init__(self, start, stop, decay):
        self.current = start
        self.stop = stop
        self.decay = decay

    def sample(self):
        eps = self.current

        new_value = self.current * self.decay
        if new_value >= self.stop:
            self.current = new_value
        else:
            self.current = self.stop
            
        return eps