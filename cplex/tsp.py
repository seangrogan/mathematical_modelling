import json
from collections import defaultdict
from random import randint, seed

import matplotlib.pyplot as plt
from docplex.mp.model import Model
from great_circle_calculator.great_circle_calculator import distance_between_points
from matplotlib.ticker import MaxNLocator


class Dist:
    """A collection of distance calculations
    """
    dist_matrix = defaultdict(dict)

    @staticmethod
    def euclidean(p1, p2):
        return pow(sum([(a - b) ** 2 for a, b in zip(p1, p2)]), 0.5)

    @staticmethod
    def manhattan(p1, p2):
        return sum(abs(a - b) for a, b in zip(p1, p2))

    @staticmethod
    def haversine(p1, p2):
        return distance_between_points(p1, p2, haversine=True)

    @staticmethod
    def law_of_cosines(p1, p2):
        return distance_between_points(p1, p2, haversine=False)

    @staticmethod
    def maximum_distance(p1, p2):
        return max(abs(a - b) for a, b in zip(p1, p2))

    def explicit(self, p1, p2, dist=None):
        if dist is None:
            return self.dist_matrix.get(p1, dict()).get(p2, None)
        self.dist_matrix[p1][p2] = dist
        return dist


def generate_cities(n=32, _min=0, _max=10):
    assert pow(abs(_max - _min), 2) >= n, "There aren't enough integer locations for the city" \
                                          "\nthis program places {n} cities at integer locations in the" \
                                          "\nx, y plane, there are {pow(abs(_max-_min), 2)} locations"
    cities = {(randint(_min, _max), randint(_min, _max)) for _ in range(n)}
    while len(cities) < n:
        cities.add((randint(_min, _max), randint(_min, _max)))
    return list(cities)


def calculate_matrix(cities, dist_function=Dist.euclidean):
    matrix = {(i, j): dist_function(i, j) for i in cities for j in cities if i != j}
    return matrix


def solve_tsp_cplex(cities, matrix, *, log=True):
    mdl = Model('TSP')
    x = mdl.binary_var_dict(matrix.keys(), name='x')
    d = mdl.continuous_var_dict(cities, name='d')
    # Objective Function
    mdl.minimize(mdl.sum(matrix[arc] * x[arc] for arc in matrix))
    # Flow out constraint
    c1 = [mdl.add_constraint(mdl.sum(x[(i, j)] for i, j in matrix if i == c) == 1, ctname=f'out_{c}') for c in cities]
    # Flow in constraint
    c2 = [mdl.add_constraint(mdl.sum(x[(i, j)] for i, j in matrix if j == c) == 1, ctname=f'in_{c}') for c in cities]
    # Logical constraint
    c3 = [mdl.add_indicator(x[(i, j)], d[i] + 1 == d[j], name=f'order_({i},_{j})') for i, j in matrix if j != cities[0]]
    # Choose one arc
    c4 = [mdl.add_constraint(x[(i, j)] + x[(j, i)] <= 1, ctname=f'identity_{i}_{j}') for i, j in matrix if i != j]
    mdl.export(f'./output/model_tsp_{len(cities)}.lp')
    try:
        mdl.solve(log_output=log)
        mdl.get_solve_status()
        arcs = {i: j for i, j in matrix if x[(i, j)].solution_value > 0.9}
        tour = []
        p1, p2 = arcs.popitem()
        while len(arcs) > 0:
            tour.append(p1)
            p1, p2 = p2, arcs.pop(p2)
        tour.append(p1)
        return tour
    except:
        print("Could not find solution!")


def plot_solution(tour, cities, *, title=None, **kwargs):
    if title is None:
        title = f"TSP with {len(cities)} cities"
    fig, ax = plt.subplots()
    fig.gca().set_aspect("equal", adjustable="box")
    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()
    ax.set_title(label=title)
    x, y = zip(*cities)
    ax.scatter(x, y, c="red", marker="o", label="Cities", zorder=100)
    x, y = zip(*tour + [tour[0]])
    ax.plot(x, y, c="blue", marker="", label="Tour", zorder=10)
    plt.savefig("./output/" + title.replace(" ", "_").lower() + ".png", bbox_inches=kwargs.get("bbox_inches", "tight"))
    plt.show()
    plt.close()


def write_data(cities, tour):
    n = len(cities)
    with open(f"./output/cities_{n}.json", "w") as json_file:
        json.dump(cities, json_file, indent=4)
    with open(f"./output/tour_{n}.json", "w") as json_file:
        json.dump({"tour": tour}, json_file, indent=4)


def tsp(n=10, _min=0, _max=10, rand_seed=None):
    if rand_seed is not None:
        seed(rand_seed)
    cities = generate_cities(n, _min, _max)
    matrix = calculate_matrix(cities)
    tour = solve_tsp_cplex(cities, matrix)
    plot_solution(tour, cities)
    write_data(cities, tour)


if __name__ == '__main__':
    tsp()
