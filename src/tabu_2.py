import pandas as pd
import time
import random
import numpy as np
from collections import defaultdict

df = pd.read_csv("data/connection_graph.csv").head(1000)

graph = defaultdict(list)
for index, row in df.iterrows():
    graph[row['start_stop']].append({
        'end_stop': row['end_stop'],
        'line': row['line'],
        'departure_time': row['departure_time'],
        'arrival_time': row['arrival_time']
    })

def calculate_changes(route):
    changes = 0
    for i in range(1, len(route)):
        if route[i-1]['line'] != route[i]['line']:
            changes += 1
    return changes

def calculate_time(route):
    total_time = 0
    for i in range(1, len(route)):
        dep_time = pd.to_datetime(route[i-1]['departure_time'])
        arr_time = pd.to_datetime(route[i]['arrival_time'])
        total_time += (arr_time - dep_time).seconds
    return total_time

def generate_neighbors(route):
    neighbors = []
    for i in range(1, len(route) - 1):
        new_route = route.copy()
        new_route[i], new_route[i+1] = new_route[i+1], new_route[i]
        neighbors.append(new_route)
    return neighbors

def tabu_search(initial_route, max_iterations, tabu_size, cost_function):
    current_route = initial_route
    best_route = initial_route
    best_cost = cost_function(initial_route)
    tabu_list = []

    for iteration in range(max_iterations):
        neighbors = generate_neighbors(current_route)
        neighbors = [neighbor for neighbor in neighbors if neighbor not in tabu_list]

        if not neighbors:
            break

        next_route = min(neighbors, key=cost_function)
        next_cost = cost_function(next_route)

        if next_cost < best_cost:
            best_route = next_route
            best_cost = next_cost

        
        tabu_list.append(next_route)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        current_route = next_route

    return best_route, best_cost

start_stop = "Chłodna"
visit_list = ["Wiejska", "FAT", "Paprotna"]
criteria = "t"  
initial_time = "5:00"


initial_route = [{'start_stop': start_stop, 'line': None, 'departure_time': initial_time, 'arrival_time': initial_time, 'end_stop': start_stop}]

for stop in visit_list:
    next_connection = random.choice(graph[initial_route[-1]['end_stop']])
    
    initial_route.append({
        'start_stop': initial_route[-1]['end_stop'],  
        'line': next_connection['line'],
        'departure_time': next_connection['departure_time'],
        'arrival_time': next_connection['arrival_time'],
        'end_stop': next_connection['end_stop']  
    })

initial_route.append({'start_stop': initial_route[-1]['end_stop'], 'line': None, 'departure_time': initial_time, 'arrival_time': initial_time, 'end_stop': start_stop})



if criteria == "t":
    cost_function = calculate_time
else:
    cost_function = calculate_changes

max_iterations = 1000
tabu_size = 50
start_time = time.time()
best_route, best_cost = tabu_search(initial_route, max_iterations, tabu_size, cost_function)
end_time = time.time()

print("Najlepsza trasa:")
for step in best_route:
    print(f"Linia: {step['line']}, Przystanek: {step['start_stop']} -> {step['end_stop']}")
print(f"Koszt: {best_cost}")
print(f"Czas obliczeń: {end_time - start_time:.4f} sekundy")


