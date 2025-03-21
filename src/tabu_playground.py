import random
import itertools
import networkx as nx
from datetime import datetime, timedelta
from process_csv import read_and_return_with_loc_line_and_time, read_with_loc_line_and_time, df_test

def convert_time(time_str):
    """Converts time from string 'HH:MM' to datetime object."""
    if type(time_str) == datetime:
        return time_str
    return datetime.strptime(time_str, "%H:%M:%S")

def time_to_minutes(time):
    """Converts datetime to minutes from midnight for easier calculation."""
    return time.hour * 60 + time.minute

def preprocess_stop_instances(graph, stop_name, arr_time):
    """
    Returns all valid stop instances (e.g., 'Chłodna@08:00_L1') for a given stop.
    """
    return [node for node in graph.nodes if node.startswith(stop_name) and convert_time(graph.nodes[node]["time"]) >= arr_time]

class TabuSearch:
    def __init__(self, graph, cost_type="transfers", tabu_tenure=5, max_iterations=100):
        """
        Tabu Search algorithm for finding an optimal path visiting required stops.
        
        :param graph: Graph of stops (NetworkX)
        :param cost_type: "weight" (travel time) or "transfers" (number of line changes)
        :param tabu_tenure: Number of iterations a move remains tabu
        :param max_iterations: Maximum number of iterations
        """
        self.graph = graph
        self.cost = self.cost_weight if cost_type == "weight" else self.line_cost
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations

    def cost_weight(self, path):
        """Calculates the total travel time for the path."""
        try:
            return sum(self.graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)) or float('inf')
        except (KeyError, TypeError):
            return float('inf')

    def line_cost(self, path):
        """Calculates the number of line transfers in the path."""
        try:
            return sum(1 for i in range(len(path)-2) if self.graph.nodes[path[i]]["line"] != self.graph.nodes[path[i+1]]["line"])
        except (KeyError, TypeError):
            return float('inf')
    # tu są jakieś krzaki - dzban nic nie znajduje
    def generate_random_solution(self, start, stops, departure_time):
        """Generates an initial random solution based on available stop instances and times."""
        initial_path = [start]
        current_time = convert_time(departure_time)

        for stop in stops:
            possible_instances = preprocess_stop_instances(self.graph, stop, current_time)
            valid_instances = []

            for instance in possible_instances:
                stop_time = convert_time(self.graph.nodes[instance]["time"])
                if stop_time >= current_time:
                    valid_instances.append(instance)

            if not valid_instances:
                return None  
            
            chosen_instance = random.choice(valid_instances)
            initial_path.append(chosen_instance)
            current_time = convert_time(self.graph.nodes[chosen_instance]["time"])

        initial_path.append(start)  
        return initial_path

    def generate_neighbors(self, path):
        """Generates neighboring solutions by swapping two random stops."""
        neighbors = []
        for i, j in itertools.combinations(range(1, len(path)-1), 2):
            new_path = path[:]
            new_path[i], new_path[j] = new_path[j], new_path[i]
            neighbors.append(new_path)
        return neighbors

    def tabu_search(self, start, stops, departure_time):
        """Main function to search for the best path."""
        start_instances = preprocess_stop_instances(self.graph, start, convert_time(departure_time))
        if not start_instances:
            raise ValueError(f"No valid instances found for start stop: {start}")
        start_instances.sort()
        start = start_instances[0]
        current_path = self.generate_random_solution(start, stops, departure_time)
        if not current_path:
            raise ValueError("No valid initial solution found with given time constraints.")

        best_path = current_path
        best_cost = self.cost(best_path)

        tabu_list = {}
        iteration = 0

        while iteration < self.max_iterations:
            neighbors = self.generate_neighbors(current_path)
            neighbors = [n for n in neighbors if tuple(n) not in tabu_list]

            if not neighbors:
                break

            best_neighbor = min(neighbors, key=self.cost)
            best_neighbor_cost = self.cost(best_neighbor)

            if best_neighbor_cost < best_cost:
                best_path = best_neighbor
                best_cost = best_neighbor_cost

            current_path = best_neighbor
            tabu_list[tuple(best_neighbor)] = iteration + self.tabu_tenure
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}

            iteration += 1

        return best_path, best_cost


if __name__ == '__main__':
    G = read_with_loc_line_and_time(df_test)
    ts = TabuSearch(G, cost_type="transfers", tabu_tenure=5, max_iterations=10000)

    start_stop = "Chłodna"
    stops_list = ["Wiejska", "FAT", "Paprotna", "Chłodna"]
    arrival_time_at_start = "07:30:00"  

    best_path, best_cost = ts.tabu_search(start_stop, stops_list, arrival_time_at_start)
    print("Optimal Path:", best_path)
    print("Total Cost (Transfers or Time):", best_cost)
