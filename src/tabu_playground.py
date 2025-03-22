import random
import itertools
import networkx as nx
from datetime import datetime, timedelta

from networkx import NetworkXNoPath
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

def normalize_name(node):
    """Extracts the part of the node name before '@'."""
    return str(node).split('@')[0] if isinstance(node, str) else str(node)

def get_nodes_starting_with(graph, prefix):
    """Returns all nodes from the graph that start with the given prefix."""
    return [node for node in graph.nodes if node.startswith(prefix)]

def find_path_with_nodes(G, required_nodes, source):
    normalized_required = {normalize_name(n) for n in required_nodes}
    potential_first_nodes = get_nodes_starting_with(G, source)
    potential_last_nodes = get_nodes_starting_with(G, source)
    # print(f"Potential first nodes: {potential_first_nodes}")

    for source in potential_first_nodes:
        for target in potential_last_nodes:
            if source != target and normalize_name(source) in normalized_required and normalize_name(target) in normalized_required:
                try:
                    path = get_random_path(G, source, target) or []
                    path_normalized = {normalize_name(n) for n in path}

                    if normalized_required.issubset(path_normalized):
                        # print(f"Found path: {path}")
                        return path  

                except nx.NetworkXNoPath:
                    continue
    return None  

def get_random_path(G, source, target, max_attempts=100):
    """
    Find a random path from source to target in a directed graph.
    
    Args:
        G: NetworkX DiGraph
        source: Source node
        target: Target node
        max_attempts: Maximum number of attempts to find a path
        
    Returns:
        A random path from source to target or None if no path is found
    """
    if not nx.has_path(G, source, target):
        return None
    
    def random_walk(current, visited=None):
        if visited is None:
            visited = [current]
        
        if current == target:
            return visited
        
        current_time = convert_time(G.nodes[current]["time"])
        
        neighbors = []
        for n in G.neighbors(current):
            if n not in visited:
                neighbor_time = convert_time(G.nodes[n]["time"])
                if neighbor_time > current_time:
                    neighbors.append(n)
        
        if not neighbors:
            return None
        
        next_node = random.choice(neighbors)
        visited.append(next_node)
        
        return random_walk(next_node, visited)
    
    for _ in range(max_attempts):
        path = random_walk(source)
        if path and path[-1] == target:
            return path
    
    try:
        return nx.shortest_path(G, source, target)
    except nx.NetworkXNoPath:
        return None

class TabuSearch:
    def __init__(self, graph, cost_type="transfers", tabu_tenure=None, max_iterations=100, departure_time=None):
        """
        Tabu Search algorithm for finding an optimal path visiting required stops.
        
        :param graph: Graph of stops (NetworkX)
        :param cost_type: "weight" (travel time) or "transfers" (number of line changes)
        :param tabu_tenure: Number of iterations a move remains tabu
        :param max_iterations: Maximum number of iterations
        :param departure_time: User's desired departure time
        """
        self.graph: nx.DiGraph = graph
        self.cost_type = cost_type
        self.cost = self.cost_weight if cost_type == "weight" else self.line_cost
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.departure_time = departure_time

    def calculate_initial_waiting_time(self, path):
        """Calculate waiting time from desired departure time to first bus/train departure."""
        if not path or not self.departure_time:
            return 0
        
        first_node_time = convert_time(self.graph.nodes[path[0]]["time"])
        
        if isinstance(self.departure_time, str):
            user_departure_time = convert_time(self.departure_time)
        else:
            user_departure_time = self.departure_time
            
        waiting_time = (first_node_time - user_departure_time).total_seconds() / 60
        
        return max(0, waiting_time)

    def cost_weight(self, path):
        """Calculates the total travel time for the path including initial waiting time."""
        if not path:
            return float('inf')
            
        initial_waiting_time = self.calculate_initial_waiting_time(path)
        
        # Calculate path travel time
        try:
            path_travel_time = nx.path_weight(self.graph, path, weight='weight')
        except (nx.NetworkXNoPath, ValueError, TypeError) as e:
            print(f"Error calculating path weight: {e}")
            return float('inf')
            
        total_cost = initial_waiting_time + path_travel_time
        # print(f'path: {path}, initial wait: {initial_waiting_time}, travel: {path_travel_time}, total: {total_cost}')
        return total_cost

    def line_cost(self, path):
        """Calculates the number of line transfers in the path plus initial waiting time penalty."""
        if not path:
            return float('inf')
            
        try:
            transfers = sum(1 for i in range(len(path)-2) 
                          if self.graph.nodes[path[i]]["line"] != self.graph.nodes[path[i+1]]["line"])
            
            # Add a scaled waiting time penalty
            waiting_time_penalty = self.calculate_initial_waiting_time(path) / 30  # Scale factor
            
            return transfers + waiting_time_penalty
        except (KeyError, TypeError) as e:
            print(f"Error calculating transfers: {e}")
            return float('inf')

    def generate_neighbors(self, path):
        """Generates neighboring solutions by swapping two random stops, ensuring the new path is valid."""
        neighbors = []
        
        for i, j in itertools.combinations(range(1, len(path) - 1), 2):
            new_path = path[:]
            
            new_path[i], new_path[j] = new_path[j], new_path[i]

            valid = True
            for k in range(len(new_path) - 1):
                if new_path[k+1] not in self.graph[new_path[k]]:
                    valid = False
                    break
            
            if valid:
                neighbors.append(new_path)  

        return neighbors


    def tabu_search(self, start, stops, departure_time):
        """Main function to search for the best path."""
        self.departure_time = departure_time
        
        start_instances = preprocess_stop_instances(self.graph, start, convert_time(departure_time))
        if not start_instances:
            raise ValueError(f"No valid instances found for start stop: {start}")
        start_instances.sort()
        current_path = find_path_with_nodes(self.graph, stops, start)
        if not current_path:
            raise ValueError("No valid initial solution found with given time constraints.")

        best_path = current_path
        best_cost = self.cost(best_path)

        tabu_list = {}
        iteration = 0
        no_improve_limit = 5
        no_improve_count = 0

        while iteration < self.max_iterations:
            neighbors = self.generate_neighbors(current_path)
            neighbors = [n for n in neighbors if tuple(n) not in tabu_list]

            if not neighbors:
                print("No valid neighbors found.")
                break

            best_neighbor = min(neighbors, key=self.cost)
            best_neighbor_cost = self.cost(best_neighbor)
            print(f"Iteration {iteration}: Best neighbor cost: {best_neighbor_cost}") 
            if best_neighbor_cost < best_cost:
                best_path = best_neighbor
                best_cost = best_neighbor_cost
                no_improve_count = 0  
            else:
                no_improve_count += 1  

            if no_improve_count >= no_improve_limit:
                print("Stopping early due to stagnation.")
                break  

            if best_neighbor_cost < best_cost:
                best_path = best_neighbor
                best_cost = best_neighbor_cost

            current_path = best_neighbor
            tabu_list[tuple(best_neighbor)] = iteration + (self.tabu_tenure or 0)
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}

            iteration += 1

        return best_path, best_cost

    def generate_random_solution(self, graph, start_stop, stops_list, arrival_time_at_start):
        """
        Generates a random solution path given the start stop, a list of stops to visit, 
        and an arrival time at the start stop.
        """
        arrival_time = convert_time(arrival_time_at_start)
        
        start_valid_nodes = preprocess_stop_instances(graph, start_stop, arrival_time)
        
        if not start_valid_nodes:
            raise ValueError(f"No valid start nodes found for stop {start_stop} at {arrival_time_at_start}")
        
        current_node = random.choice(start_valid_nodes)
        path = [current_node]
        current_time = graph.nodes[current_node]['time']

        for stop in stops_list:
            valid_nodes = preprocess_stop_instances(graph, stop, current_time)

            if not valid_nodes:
                raise ValueError(f"No valid nodes found for stop {stop} after {current_time}")
            
            next_node = random.choice(valid_nodes)
            path.append(next_node)

            current_time = graph.nodes[next_node]['time']

        valid_nodes = preprocess_stop_instances(graph, start_stop.split('@')[0], current_time)
        
        if not valid_nodes:
            raise ValueError(f"No valid nodes found for return to {start_stop} after {current_time}")
        
        next_node = random.choice(valid_nodes)
        path.append(next_node)

        return path


if __name__ == '__main__':
    G = read_with_loc_line_and_time(df_test)
    ts = TabuSearch(G, cost_type="weight", tabu_tenure=5, max_iterations=100)

    start_stop = "Chłodna"
    stops_list = [start_stop]+["Wiejska", "FAT", "Paprotna", "Chłodna"]
    arrival_time_at_start = "3:30:00"  

    best_path, best_cost = ts.tabu_search(start_stop, stops_list, arrival_time_at_start)
    print("Optimal Path:", best_path)
    print("Total Cost (Transfers or Time):", best_cost)
    # nodes_starting_with_chlodna = get_nodes_starting_with(G, "Chłodna")
    # print("Nodes starting with 'Chłodna':", nodes_starting_with_chlodna)
    
    # if len(nodes_starting_with_chlodna) >= 2:
    #     nodeA = nodes_starting_with_chlodna[0]
    #     nodeB = nodes_starting_with_chlodna[-1]
    #     random_path = get_random_path(G, nodeA, nodeB)
    #     if random_path:
    #         print(f"Random path from {nodeA} to {nodeB}:")
    #         print(random_path)
    #     else:
    #         print(f"No path found from {nodeA} to {nodeB}")
