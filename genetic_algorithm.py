import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class City:

    def __init__(self,x,y):
        self.x = float(x)
        self.y = float(y)

    def distance(self,city):
        return float(((self.x - city.x)**2 + (self.y - city.y)**2)**0.5)


class chromosome:

    def __init__(self,cities,nodes):

        self.cities = cities
        self.nodes = nodes
        self.route = self.generate_random_route()
        self.calculate_fitness(self.route,self.cities)


    def calculate_fitness(self,route,cities):
        '''
        Calculates the path distance
        '''
        dist = 0
        for i,j in zip(route,route[1:]):
            dist += cities[i].distance(cities[j])

        self.fitness = dist


    def mutate(self):
        '''
        Swaps two cities
        '''
        idx1 = np.random.randint(1,len(self.route)-1)
        idx2 = np.random.randint(1,len(self.route)-1)

        while idx2 == idx1:
            idx2 = np.random.randint(1,len(self.route)-1)

        mutated = self.route.copy()
        mutated[idx1] = self.route[idx2]
        mutated[idx2] = self.route[idx1]

        self.route = mutated
        self.fitness = self.calculate_fitness(self.route,self.cities)


    def generate_random_route(self):
        '''
        nodes = pd.DataFrame (columns = ["X","Y"], index=city)
        '''
        l = len(self.nodes)
        r = [i for i in range(1,l)]
        random.shuffle(r)
        r.insert(0,0)
        r.append(0)

        return r



def generate_cities(nodes):
    '''
    nodes = pd.DataFrame (columns = ["X","Y"], index=city)
    The city with index 0 is considered as the initial and final city
    '''
    cities = []
    for i in range(len(nodes)):
        city = City(nodes["X"][i],nodes["Y"][i])
        cities.append(city)

    return cities

def create_pop(npop,nodes):
    '''
    Creates a [npop] chromosomes in a list
    '''
    pop = []

    for i in range(npop):
        pop.append(chromosome(generate_cities(nodes),nodes))

    return pop

def sort_pop(population):
    '''
    Sort the chromosomes (ascending distance, shortest path first)
    '''
    pop = sorted(population, key=lambda x: x.fitness, reverse=False)

    return pop


def select_parent(pop):
    '''
    Select one parent from population.
    '''
    s = sum([1/i.fitness for i in pop])
    parent = random.choices(pop,weights=[(1/i.fitness)/s for i in pop])

    return parent[0]

def crossover(pop,cities,nodes):
    '''
    Returns all the childrens for the population (n_children = n_pop)
    Uses only one point
    '''
    pop_copy = pop.copy()
    children = []

    while len(pop_copy) >= 2:

        parent1 = select_parent(pop_copy)
        pop_copy.remove(parent1)

        parent2 = select_parent(pop_copy)
        pop_copy.remove(parent2)

        idx = np.random.randint(2,len(parent1.route)-2)

        child1 = chromosome(cities,nodes)
        child2 = chromosome(cities,nodes)

        temp_list = parent1.route[:idx] + [i for i in parent2.route[idx:] if i not in parent1.route[:idx]]
        child1.route = temp_list + [i for i in range(1,len(parent1.route)-1) if i not in temp_list]
        child1.route.append(0)

        temp_list = parent2.route[:idx] + [i for i in parent1.route[idx:] if i not in parent2.route[:idx]]
        child2.route = temp_list + [i for i in range(1,len(parent2.route)-1) if i not in temp_list]
        child2.route.append(0)

        children.append(child1)
        children.append(child2)

    return children

def genetic_algorithm(npop,nodes,generations,mutation_prob):
    '''
    Init population
    '''
    pop = create_pop(npop,nodes)
    cities = generate_cities(nodes)
    hist = []
    '''
    Start generations
    '''
    for i in range(generations):
        '''
        Crossover population
        '''
        children = crossover(pop,cities,nodes)
        '''
        Mutation operation
        '''
        for i in children:
            if np.random.uniform(0,1) < mutation_prob:
                i.mutate()
        '''
        Elitism
        '''
        pop += children

        for i in pop:
            i.calculate_fitness(i.route,cities)

        sorted_pop = sort_pop(pop)
        pop = sorted_pop[:npop]

        '''
        Append best route in this generation
        '''
        hist.append(pop[0].fitness)

    return hist, pop[0].route, pop[0].fitness, pop

def plot_results(hist, fitness, route, nodes):
    '''
    Plots the history of the distance and the best route generated
    '''
    f, (ax1,ax2) = plt.subplots(ncols=1,nrows=2,figsize=(15,15))

    ax1.plot(hist)
    ax1.set_title("Distance history",fontsize=20)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best distance")

    ax2.scatter(nodes["X"][0],nodes["Y"][0],s=300,label="Initial city")
    ax2.scatter(nodes["X"][1:],nodes["Y"][1:],s=300)
    for i,j in zip(route,route[1:]):
        ax2.plot([nodes["X"][i], nodes["X"][j]],[nodes["Y"][i],nodes["Y"][j]], "k")
    ax2.set_title("Route",fontsize=20)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    plt.legend(loc="best")
    plt.show()

    print("Total route distance = {}\n".format(fitness))
    print("Route = {}".format(route))
