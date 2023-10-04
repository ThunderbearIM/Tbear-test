import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import dist


class Individual:


    def __init__(self):
        self.brain = Brain()
        self.genes = self.genes()
        self.max_steps = 100
        self.pos = self.brain.position[-1]
        self.goal = np.array([5, 95])
        self.fitness = self.fitness()

    def genes(self):
        brain = self.brain
        return brain.acceleration()

    @staticmethod
    def generation_number(x):
        if x > 0:
            x += 1

        else:
            x = 1

        return x

    def distance_from_goal(self):
        pos = self.pos
        goal = self.goal
        distance = dist(pos, goal)
        return distance

    def fitness(self):
        """
        Evaluates the fitness of the dots based on distance from the goal in a 2d space with euclidian distance
        :return: goodness_of_fit: miminmized fitness value based on goal
        """
        distance = self.distance_from_goal()
        fit = distance^2
        if distance < 2:
            fit = fit/5
        return fit

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
        pos = self.position
        for i in range(len(velocity)):
            new_pos = pos[i]+velocity[i]
            pos = np.vstack((pos[i], new_pos))

        self.position = pos


class Population:

    """
    Class made up of multiple individuals
    """

    def __init__(self):
        self.pop_size = 400
        self.individual = Individual()
        self.generation = 0
        self.max_steps = 100
        self.steps_left = 100

    @staticmethod
    def create_population(pop_size):
        """
        Initialize a population of Class individuals based on the parameter pop_size
        :param pop_size:
        :return:
        """
        lst = list()
        for i in range(pop_size):
            lst.append(Individual())

        return lst

