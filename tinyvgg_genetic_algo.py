"""
Written as part of CRUx inductions round 3. Objective is to implement a Tiny VGG architecture using tensorflow, trained
using a genetic algorithm. Architecture of network is as given in the task file. Model is evaluated against the CIFAR-10
database
"""
import tensorflow as tf
import numpy as np
import random
import time
import matplotlib.pyplot as plot

cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255, X_test / 255


class CNN_architecture:
    def __init__(self):
        self.inp_layer = tf.keras.layers.Input(shape=(32, 32, 3))
        self.conv_layer_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(
            self.inp_layer)
        self.max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')(self.conv_layer_1)
        self.conv_layer_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(
            self.max_pool_1)
        self.max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')(self.conv_layer_2)
        self.flattened_layer = tf.keras.layers.Flatten()(self.max_pool_2)
        self.dense_1 = tf.keras.layers.Dense(512, activation='relu')(self.flattened_layer)
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')(self.dense_1)
        self.model = tf.keras.Model(inputs=self.inp_layer, outputs=self.output_layer)
        # self.params = (self.model.get_weights())
        self.fitness = None

    def get_fitness(self):
        self.model.compile(optimizer='rmsprop', metrics=['accuracy'])
        test_loss, test_accuracy = self.model.evaluate(X_train, y_train, verbose=0)
        return test_accuracy

    def update_fitness(self):
        self.fitness = self.get_fitness()


def cross_individuals(m1, m2):
    c_w1, c_w2 = [], []
    p_w1, p_w2 = m1.model.get_weights(), m2.model.get_weights()
    for layer in range(0, len(p_w1)):
        p_w1_layer = np.array(p_w1[layer]).flatten()
        p_w2_layer = np.array(p_w2[layer]).flatten()
        c_w1_layer, c_w2_layer = np.zeros(len(p_w1_layer)), np.zeros(len(p_w2_layer))
        for n in range(0, len(p_w1_layer)):
            choice = random.random()
            if choice < 0.5:
                c_w1_layer[n] = p_w1_layer[n]
                c_w2_layer[n] = p_w2_layer[n]
            else:
                c_w1_layer[n] = p_w2_layer[n]
                c_w2_layer[n] = p_w1_layer[n]
        c_w1_layer = c_w1_layer.reshape(p_w1[layer].shape)
        c_w2_layer = c_w2_layer.reshape(p_w2[layer].shape)
        c_w1.append(c_w1_layer)
        c_w2.append(c_w2_layer)
    return c_w1, c_w2


def initialize_population(size):
    population = []
    for _ in range(0, size):
        model = CNN_architecture()
        population.append(model)
    return population


def select(population):
    for individual in population:
        individual.update_fitness()
    population = sorted(population, key=lambda c: c.fitness, reverse=True)
    elites = population[:int(len(population) / 2)]
    return elites


def cross_population(elites):
    new_population = []
    while len(elites) > 0:
        m1 = elites.pop(0)
        m2 = elites.pop(0)
        c1, c2 = cross_individuals(m1, m2)
        new_population.append(m1)
        new_population.append(m2)
        m3, m4 = CNN_architecture(), CNN_architecture()
        m3.model.set_weights(c1)
        m4.model.set_weights(c2)
        new_population.append(m3)
        new_population.append(m4)
    for individual in new_population:
        individual.update_fitness()
    return new_population


def print_population(number, population, average, best):
    print(f"GENERATION {number}")
    print(f"Average accuracy: {round(average, 3)} || Best Accuracy: {round(best,3)}")
    for individual in population:
        print(round(individual.fitness, 3), end=" ")
    print()


def evolve_populations(n_populations, population_size):
    originals = initialize_population(population_size)
    n = n_populations
    parents = originals
    averages = []
    bests = []
    s = 0
    for individual in originals:
        individual.update_fitness()
        s += individual.fitness
    parents = sorted(parents, key=lambda c: c.fitness, reverse=True)
    average = s / len(parents)
    averages.append(average)
    best = parents[0].fitness
    bests.append(best)
    print_population(0, parents, average, best)
    while n > 0:
        elites = select(parents)
        new_population = cross_population(elites)
        parents = new_population
        s = 0
        for individual in new_population:
            individual.update_fitness()
            s += individual.fitness
        parents = sorted(parents, key=lambda c: c.fitness, reverse=True)
        average = s / len(parents)
        averages.append(average)
        best = parents[0].fitness
        bests.append(best)
        print(f"Time: {time.ctime(time.time())}")
        print_population(n_populations - n+1, parents, average, best)
        n -= 1
    return parents, averages, bests


if __name__ == '__main__':
    '''model = CNN_architecture()
    print(model.model.summary())'''

    n_generations = 100
    population_size = 20
    generations = [n for n in range(1, n_generations+1)]
    print(f"Program started at {time.ctime(time.time())}")
    final_elites, averages, bests = evolve_populations(100, 20)
    plot.figure()
    plot.title("Accuracy over generations")
    plot.plot(generations, averages, label="Average accuracy")
    plot.plot(generations, bests, label="Best Accuracy")
    plot.xlabel("Generation")
    plot.ylabel("Accuracy")
    plot.show()

