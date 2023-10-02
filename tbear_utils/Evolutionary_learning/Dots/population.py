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
        self.goal = np.array([5, 95])

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

    def distance_from_goal(self):

    def fitness(self):
        """
        Evaluates the fitness of the dots based on distance from the goal in a 2d space with euclidian distance
        :return: goodness_of_fit
        """





class Brain:
    """
    The brain of the dots individuals, we will
    """

    def __init__(self):
        self.size=400
        self.velocity = np.zeros([400, 2])
        self.acceleration = self.acceleration()
        self.position = np.array([[5, 5]])

    def acceleration(self, size):
        """
        Acceleration method for the direction of the dots
        :return:
        """
        x = np.random.uniform(low=-1, high=1, size=size)
        y = np.random.uniform(low=-1, high=1, size=size)
        accel = np.append(x, y).reshape(size, 2)
        return accel

    def velocity(self):
        """
        Takes the acceleration and adds it to the velocity
        :return:
        """
        velocity = self.velocity
        accel = self.acceleration

        for i in range(1, len(velocity)):

            velocity[i] = accel[i]+velocity[i-1]

            if velocity[i][0] > 3:
                velocity[i][0] = 3

            if velocity[i][0] < -3:
                velocity[i][0] = -3

            if velocity[i][1] > 3:
                velocity[i][1] = 3

            if velocity[i][1] < -3:
                velocity[i][1] = -3

        return velocity

    def mutate(self):
        genes = self.acceleration
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

        self.acceleration = genes

    def position(self):
        """
        Adds the current velocity to the position
        :return:
        """
        velocity = self.velocity
        for i in range(len(velocity)):
            new_pos = pos[i]+velocity[i]
            pos = np.vstack((pos[i], new_pos))



class Population:

    def __init__(self):
        self.individual = Individual()
        self.brain = Brain()