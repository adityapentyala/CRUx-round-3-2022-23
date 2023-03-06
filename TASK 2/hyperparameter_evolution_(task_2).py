"""
Written as task 2 of crux inductions round 3. Details of hyperparameters tuned given in task 2.md file
"""

import tensorflow as tf
import random
import matplotlib.pyplot as plot
import time

cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255, X_test / 255
# print(X_train.shape)
# print(y_train.shape)
'''y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)'''

# convolutional layer hyperparams
n_conv_layers = (1, 2)  # originally upto 4 layers, cut down to circumvent time bottleneck (first run (1, 2))
n_filters = (32, 64, 128, 256)  # originally included 512, cut down to circumvent time bottleneck
kernel_sizes = (2, 3, 4, 5, 6)
strides = (1, 2, 3, 4, 5)
activation_functions = ('relu', 'sigmoid')

# dense layer hyperparams (incl activation functions)
n_dense_layers = (1, 2)
n_neurons = (256, 512, 1024)
# originally used 2048 and 4096 neurons as well, cut down in attempt to circumvent time bottleneck

# maxpool layer hyperparams
n_maxpool_layers = (1, 2)
pool_sizes = (2, 3, 4)

# misc hyperparams
n_epochs = [int(n) for n in
            range(2, 9)]  # originally ranged [5, 15], cut down in attempt to circumvent time bottleneck
optimizers = ('adam', 'rmsprop', 'sgd')


class CNN:
    def __init__(self, n_epochs, raw_conv_layers, raw_mp_layers, raw_dense_layers, optimizer):
        self.chromosome = [n_epochs, raw_conv_layers, raw_mp_layers, raw_dense_layers, optimizer]
        self.inp_layer = tf.keras.layers.Input(shape=(32, 32, 3))
        self.conv_layers = []
        self.dense_layers = []
        self.maxpool_layers = []
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        for c_layer in raw_conv_layers:
            layer = tf.keras.layers.Conv2D(filters=c_layer['filters'], kernel_size=c_layer['kernel_size'],
                                           strides=c_layer['strides'], activation=c_layer['activation'],
                                           padding='same')
            self.conv_layers.append(layer)
        for d_layer in raw_dense_layers:
            layer = tf.keras.layers.Dense(d_layer['n_nodes'], activation=d_layer['activation'])
            self.dense_layers.append(layer)
        for mp_layer in raw_mp_layers:
            layer = tf.keras.layers.MaxPool2D(pool_size=mp_layer['pool_size'], padding='same')
            self.maxpool_layers.append(layer)
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')
        self.model = tf.keras.Sequential()
        self.construct_model()
        self.fitness = None

    def construct_model(self):
        # add input layer
        self.model.add(self.inp_layer)
        # add conv & maxpool layers
        if len(self.maxpool_layers) == 1:
            for conv_layer in self.conv_layers:
                self.model.add(conv_layer)
            self.model.add(self.maxpool_layers[0])
        elif len(self.maxpool_layers) == 2:
            for n in range(len(self.conv_layers)):
                self.model.add(self.conv_layers[n])
                if n == 0:
                    self.model.add(self.maxpool_layers[0])
            self.model.add(self.maxpool_layers[1])
        # add flatten layer
        self.model.add(tf.keras.layers.Flatten())
        # add dense layers
        for dense_layer in self.dense_layers:
            self.model.add(dense_layer)
        # add output layer
        self.model.add(self.output_layer)
        self.model.compile(optimizer=self.optimizer, metrics=['accuracy'],
                           loss='sparse_categorical_crossentropy')
        self.model.fit(X_train, y_train, epochs=self.n_epochs, verbose=0)

    def check_fitness(self):
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        self.fitness = test_accuracy


def generate_conv_layers_gene():
    n_layers = random.choice(n_conv_layers)
    layers = []
    for _ in range(n_layers):
        gene = {}
        gene['filters'] = random.choice(n_filters)
        gene['kernel_size'] = random.choice(kernel_sizes)
        gene['strides'] = random.choice(strides)
        gene['activation'] = random.choice(activation_functions)
        layers.append(gene)
    return layers


def generate_dense_layers_gene():
    n_layers = random.choice(n_dense_layers)
    layers = []
    for _ in range(n_layers):
        gene = {}
        gene['n_nodes'] = random.choice(n_neurons)
        gene['activation'] = random.choice(activation_functions)
        layers.append(gene)
    return layers


def generate_maxpool_layers_gene():
    n_layers = random.choice(n_dense_layers)
    layers = []
    for _ in range(n_layers):
        gene = {}
        gene['pool_size'] = random.choice(pool_sizes)
        layers.append(gene)
    return layers


def initialize_population(size):
    population = []
    for _ in range(size):
        chromosome = []
        # epochs gene
        chromosome.append(random.choice(n_epochs))
        # convolutional layers gene
        chromosome.append(generate_conv_layers_gene())
        # maxpool layers gene
        chromosome.append(generate_maxpool_layers_gene())
        # dense layers gene
        chromosome.append(generate_dense_layers_gene())
        # optimizers gene
        chromosome.append(random.choice(optimizers))

        # create model with chromosome and add to population
        model = CNN(n_epochs=chromosome[0], raw_conv_layers=chromosome[1], raw_mp_layers=chromosome[2],
                    raw_dense_layers=chromosome[3], optimizer=chromosome[4])
        population.append(model)
        # print("model created!")
    for individual in population:
        individual.check_fitness()
    population = sorted(population, reverse=True, key=lambda c: c.fitness)
    return population


def print_population(gen_number, population, average, best):
    print(time.ctime(time.time()))
    print(f"GENERATION {gen_number}")
    print(f"Average accuracy: {round(average, 3)} || Best Accuracy: {round(best, 3)}")
    for individual in population:
        print(round(individual.fitness, 3), end=" ")
    print()


def select_elites(population):
    for individual in population:
        individual.check_fitness()
    sorted_population = sorted(population, reverse=True, key=lambda c: c.fitness)
    elites = sorted_population[:int(len(population) / 2)]
    return elites


def cross_individuals(p1, p2):
    c1_chromo, c2_chromo = [None, None, None, None, None], [None, None, None, None, None]
    p1_chromo, p2_chromo = p1.chromosome, p2.chromosome
    for i in range(0, len(p1.chromosome)):
        if random.random() < 0.5:
            c1_chromo[i] = p1_chromo[i]
            c2_chromo[i] = p2_chromo[i]
        else:
            c1_chromo[i] = p2_chromo[i]
            c2_chromo[i] = p1_chromo[i]
    c1 = CNN(c1_chromo[0], c1_chromo[1], c1_chromo[2], c1_chromo[3], c1_chromo[4])
    c2 = CNN(c2_chromo[0], c2_chromo[1], c2_chromo[2], c2_chromo[3], c2_chromo[4])
    return c1, c2


def find_avg(population):
    s = 0
    for individual in population:
        s += individual.fitness
    return round(s / len(population), 3)


def cross_population(elites):
    new_population = []
    while len(elites) > 0:
        p1 = elites.pop(random.randint(0, len(elites) - 1))
        p2 = elites.pop(random.randint(0, len(elites) - 1))
        c1, c2 = cross_individuals(p1, p2)
        new_population.append(p1)
        new_population.append(p2)
        new_population.append(c1)
        new_population.append(c2)
    for individual in new_population:
        individual.check_fitness()
    sorted_population = sorted(new_population, reverse=True, key=lambda c: c.fitness)
    return sorted_population


if __name__ == '__main__':
    starttime = time.ctime(time.time())
    print("Initialization started at", starttime)
    current_population = initialize_population(12)
    for individual in current_population:
        individual.check_fitness()
    current_population = sorted(current_population, reverse=True, key=lambda c: c.fitness)
    print_population(population=current_population, average=find_avg(current_population),
                     best=current_population[0].fitness, gen_number=0)
    n_generations = 10
    averages = []
    bests = []
    averages.append(find_avg(current_population))
    bests.append(current_population[0].fitness)
    generations = [int(i) for i in range(0, n_generations+1)]
    for n in range(n_generations):
        elites = select_elites(current_population)
        new_population = cross_population(elites)
        current_population = new_population
        av = find_avg(current_population)
        print_population(population=current_population, average=av,
                         best=current_population[0].fitness, gen_number=n+1)
        averages.append(av)
        bests.append(current_population[0].fitness)
        print()
    print("Best-performing architectures: ")
    print()
    for i in range(0, 4):
        print(i)
        print(current_population[i].model.summary())
        print("Hyperparameters tuned:", current_population[i].chromosome)
        print()
    print("Ended at:", time.ctime(time.time()))
    plot.figure()
    plot.title("Accuracy over generations")
    plot.plot(generations, averages, label="Average accuracy")
    plot.plot(generations, bests, label="Best Accuracy")
    plot.xlabel("Generation")
    plot.ylabel("Accuracy")
    plot.legend()
    plot.show()

    # Baseline model
    '''inp_layer = tf.keras.layers.Input(shape=(32, 32, 3))
    conv_layer_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(
        inp_layer)
    max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')(conv_layer_1)
    conv_layer_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(
        max_pool_1)
    max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')(conv_layer_2)
    flattened_layer = tf.keras.layers.Flatten()(max_pool_2)
    dense_1 = tf.keras.layers.Dense(1024, activation='relu')(flattened_layer)
    dense_2 = tf.keras.layers.Dense(1024, activation='relu')(dense_1)
    output_layer = tf.keras.layers.Dense(10, activation='softmax')(dense_2)
    model = tf.keras.Model(inputs=inp_layer, outputs=output_layer)
    model.compile(optimizer='sgd', metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    model.fit(X_train, y_train, epochs=10, verbose=1)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    model.summary()
    print("Accuracy of baseline model:", test_accuracy)'''
