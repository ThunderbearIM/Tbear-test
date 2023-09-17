import pandas as pd
import numpy as np

class Individual:


    def __init__(self):
        self.genes = genes()
        self.generation = generation_number()

    @staticmethod
    def genes():
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 10)
        genes = np.append(x, y)
        return genes

    @staticmethod
    def generation_number(x):
        if x:
            x += 1

        else:
            x = 1

        return x