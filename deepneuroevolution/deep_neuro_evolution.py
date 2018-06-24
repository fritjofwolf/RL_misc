import numpy as np
import keras

class DeepNeuroEvolution():

    def __init__(self, num_of_individuals):
        self._individuals = self._create_first_generation(num_of_individuals)

    def _create_first_generation(self, num_of_individuals):
        pass

    def evaluate_generation(self):
        pass

    def evaluate_policy_network(self, pol_net, env, iterations):
        pass

    