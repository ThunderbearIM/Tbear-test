import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Individual:


    def __init__(self):
        self.brain = Brain()
        self.genes = genes()
        self.generation = generation_number()
        self.max_steps = 100
        self.time_left = time_left()
        self.velocity = velocity()
        self.accel = accel()

    def genes(self):
        brain = self.brain
        return brain.position

    @staticmethod
    def generation_number(x):
        if x > 0:
            x += 1

        else:
            x = 1

        return x

    def mutate(self):
        genes = self.genes
        """
        Sees if the genes in the individual will mutate 
        :return: 
        """
        mutation_chance = 0.01
        mutation_attempt = np.random.uniform(low=0, high=1)
        mutation_successful = mutation_attempt < mutation_chance
        if mutation_successful:
            """
            multiply the genes by somewhere between 1.1 and 0.9
            """
            mutate_x = np.random.uniform(low=0.9, high=1.1)
            mutate_y = np.random.uniform(low=0.9, high=1.1)
            genes = np.append(genes[0]*mutate_x, genes[1]*mutate_y)

        self.genes = genes

    def time_left(self, time = self.time_left):
        if time:




class Brain:
    """
    The brain of the dots individuals, we will
    """

    def __init__(self):
        self.size=400
        self.velocity = np.array([0, 0])
        self.acceleration = self.acceleration()
        self.position = np.array([0, 5])

    @staticmethod
    def acceleration():
        """
        Acceleration method for the direction of the dots
        :return:
        """
        x = np.random.uniform(low=-1, high=1)
        y = np.random.uniform(low=-1, high=1)
        return x, y

    def velocity(self):
        """
        Takes the acceleration and adds it to the velocity
        :return:
        """
        velocity = self.velocity
        x_accel, y_accel = self.acceleration
        velocity = np.array([velocity[0]+x_accel, velocity[1]+y_accel])

        if velocity[0] > 3:
            velocity[0] = 3

        if velocity[0] < -3:
            velocity[0] = -3

        if velocity[1] > 3:
            velocity[1] = 3

        if velocity[1] < -3:
            velocity[1] = -3

        self.velocity = velocity
        return self.velocity

    def position(self):
        """
        Adds the current velocity to the position
        :return:
        """
        pos = self.position
        velocity = self.velocity()
        pos[0] += self.velocity[0]



class Population:

    def __init__(self):
        self.individual = Individual()