import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

class City:
    '''
    x = float/int
    y = float/int
    '''

    def __init__(self,x,y):
        self.x = float(x)
        self.y = float(y)

    def distance(self,city):
        return float(((self.x - city.x)**2 + (self.y - city.y)**2)**0.5)


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


def generate_random_solution(nodes):
    '''
    nodes = pd.DataFrame (columns = ["X","Y"], index=city)
    '''
    l = len(nodes)
    r = [i for i in range(1,l)]
    random.shuffle(r)
    r.insert(0,0)
    r.append(0)

    return r

def route_distance(route,cities):
    '''
    route = list of cities index
    cities = list of cities
    '''
    dist = 0
    for i,j in zip(route,route[1:]):
        dist += cities[i].distance(cities[j])

    return dist


def search_neighbors_reverse(route,k):
    '''
    Example:
            k = 3
            A[BCD]EF --> A[DCB]EF

            if k=2 reverse two adjacent cities

    k = length of the subset to reverse in the route
    route = list of cities index
    '''

    random_idx = np.random.randint(1,len(route)-k)
    n = route.copy()
    n[random_idx:random_idx+k] = n[random_idx:random_idx+k][::-1]

    return n

def search_neighbors_random(route):
    '''
    route = list of cities index
    '''

    random_idx_1 = np.random.randint(1,len(route)-2)
    random_idx_2 = np.random.randint(1,len(route)-2)

    n = route.copy()
    n[random_idx_1] = route[random_idx_2]
    n[random_idx_2] = route[random_idx_1]

    return n

def simulated_annealing(nodes,it,k,j,T,alpha):
    '''
    nodes = pd.DataFrame (columns = ["X","Y"], index=city)
    it = number of iterations
    k = number of neighbors to create in each iteration
    j = number of subset in a route to reverse (neighbor search)
    T = Initial temperature
    alpha = Temperature factor
    '''

    '''
    Init random solution
    '''
    cities = generate_cities(nodes)
    current_sol = generate_random_solution(nodes)
    current_best_distance = route_distance(current_sol,cities)
    hist = []

    '''
    Start iterations
    '''
    for i in range(it):
        '''
        Neighbor generation
        '''
        for i in range(int(k/3)):
            '''Neighbors generation'''

            '''Reverse subset neighbor'''
            n1 = search_neighbors_reverse(current_sol,j)
            n1_dist = route_distance(n1,cities)
            '''Random cities swap'''
            n2 = search_neighbors_random(current_sol)
            n2_dist = route_distance(n2,cities)
            '''Adjacent cities permutation'''
            n3 = search_neighbors_reverse(current_sol,2)
            n3_dist = route_distance(n3,cities)

            '''Neighbors evaluation'''
            if n1_dist < current_best_distance:
                current_sol = n1
                current_best_distance = n1_dist
            elif np.random.uniform(0,1) < np.exp(-(abs(n1_dist - current_best_distance))/T):
                current_sol = n1
                current_best_distance = n1_dist

            if n2_dist < current_best_distance:
                current_sol = n2
                current_best_distance = n2_dist
            elif np.random.uniform(0,1) < np.exp(-(abs(n2_dist - current_best_distance))/T):
                current_sol = n2
                current_best_distance = n2_dist

            if n3_dist < current_best_distance:
                current_sol = n3
                current_best_distance = n3_dist
            elif np.random.uniform(0,1) < np.exp(-(abs(n3_dist - current_best_distance))/T):
                current_sol = n3
                current_best_distance = n3_dist

            '''Append iteration results'''
            hist.append(current_best_distance)
            '''Reduce temperature'''
            T = T*alpha

    return hist, current_best_distance, current_sol


def plot_results(hist, current_best_distance, current_sol, nodes):
    '''
    Plots the history of the distance and the best route generated
    '''
    f, (ax1,ax2) = plt.subplots(ncols=1,nrows=2,figsize=(15,15))

    ax1.plot(hist)
    ax1.set_title("Distance history",fontsize=20)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Distance")

    ax2.scatter(nodes["X"][0],nodes["Y"][0],s=300,label="Initial city")
    ax2.scatter(nodes["X"][1:],nodes["Y"][1:],s=300)
    for i,j in zip(current_sol,current_sol[1:]):
        ax2.plot([nodes["X"][i], nodes["X"][j]],[nodes["Y"][i],nodes["Y"][j]], "k")
    ax2.set_title("Route",fontsize=20)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    plt.legend(loc="best")
    plt.show()

    print("Total route distance = {}\n".format(current_best_distance))
    print("Route = {}".format(current_sol))
