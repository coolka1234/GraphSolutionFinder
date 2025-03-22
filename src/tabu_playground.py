from operator import ge
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

def get_time_from_node(node):
    """Extracts the time part from the node name."""
    return str(node).split('@')[1].split('_')[0]

def calculate_initial_waiting_time(path, departure_time):
    """Calculate waiting time from desired departure time to first bus/train departure."""
    if not path or not departure_time:
        return 0
    
    first_node_time = convert_time(get_time_from_node(path[0]))
    
    if isinstance(departure_time, str):
        user_departure_time = convert_time(departure_time)
    else:
        user_departure_time = departure_time
        
    waiting_time = (first_node_time - user_departure_time).total_seconds() / 60
    
    return max(0, waiting_time)

def find_path_with_nodes(G, required_nodes, source, departure_time=None, max_attempts=100):
    normalized_required = {normalize_name(n) for n in required_nodes}
    potential_first_nodes = get_nodes_starting_with(G, source)
    potential_last_nodes = get_nodes_starting_with(G, source)
    # print(f"Potential first nodes: {potential_first_nodes}")
    best_path = None
    best_cost = float('inf')
    iteration=0
    for source in potential_first_nodes:
        for target in potential_last_nodes:
            if source != target and normalize_name(source) in normalized_required and normalize_name(target) in normalized_required:
                try:
                    path = get_random_path(G, source, target) or []
                    path_normalized = {normalize_name(n) for n in path}

                    if normalized_required.issubset(path_normalized):
                        # print(f"Found path: {path}")
                        cost= nx.path_weight(G, path, weight="weight")+calculate_initial_waiting_time(path, departure_time)
                        if cost<best_cost:
                            best_path = path
                            best_cost = nx.path_weight(G, path, weight="weight")  
                     
                except nx.NetworkXNoPath:
                    continue
                iteration+=1
                if iteration>max_attempts:
                    break
    return best_path

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
    def __init__(self, graph, cost_type="transfers", tabu_tenure=None, max_iterations=100, departure_time=None, use_aspiration=True):
        """
        Tabu Search algorithm for finding an optimal path visiting required stops.
        
        :param graph: Graph of stops (NetworkX)
        :param cost_type: "weight" (travel time) or "transfers" (number of line changes)
        :param tabu_tenure: Number of iterations a move remains tabu
        :param max_iterations: Maximum number of iterations
        :param departure_time: User's desired departure time
        :param use_aspiration: Whether to use aspiration criteria to override tabu status
        """
        self.graph: nx.DiGraph = graph
        self.cost_type = cost_type
        self.cost = self.cost_weight if cost_type == "weight" else self.line_cost
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.departure_time = departure_time
        self.use_aspiration = use_aspiration

    def get_tabu_size_log(num_stops):
        """Tabu size scales logarithmically for large graphs."""
        import math
        return max(3, int(math.log2(num_stops) * 5))

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
            
            waiting_time_penalty = self.calculate_initial_waiting_time(path) / 30  
            
            return transfers + waiting_time_penalty
        except (KeyError, TypeError) as e:
            print(f"Error calculating transfers: {e}")
            return float('inf')

    def generate_neighbors(self, path, required_stops=None):
        """
        Generates neighboring solutions that:
        1. Include all required stops
        2. Maintain chronological order (later times come after earlier times)
        3. Are valid paths in the graph (edges exist between consecutive nodes)
        
        Args:
            path: Current path solution
            required_stops: List of required stop names (without time information)
            
        Returns:
            List of valid neighboring paths
        """
        if not required_stops:
            required_stops = list(set(normalize_name(node) for node in path))
        
        neighbors = []
        
        for i in range(len(path)):
            current_node = path[i]
            current_stop = normalize_name(current_node)
            
            alt_instances = [
                node for node in self.graph.nodes 
                if normalize_name(node) == current_stop and node != current_node
            ]
            
            for alt_node in alt_instances:
                new_path = path.copy()
                new_path[i] = alt_node
                
                if self.is_valid_path(new_path):
                    neighbors.append(new_path)
        
        required_stop_names = set(required_stops)
        for i in range(1, len(path) - 1):
            for j in range(i + 1, len(path) - 1):
                if (normalize_name(path[i]) in required_stop_names and 
                    normalize_name(path[j]) in required_stop_names):
                    continue
                    
                new_path = path.copy()
                new_path[i], new_path[j] = new_path[j], new_path[i]
                if self.is_valid_path(new_path):
                    neighbors.append(new_path)
        
        for i in range(len(path) - 1):
            if normalize_name(path[i]) in required_stop_names or normalize_name(path[i+1]) in required_stop_names:
                try:
                    shorter_path = nx.shortest_path(self.graph, path[i], path[i+1])
                    if len(shorter_path) < 3:  
                        continue
                        
                    new_path = path[:i] + shorter_path + path[i+2:]
                    if self.is_valid_path(new_path):
                        neighbors.append(new_path)
                except nx.NetworkXNoPath:
                    continue
        
        return neighbors
    
    def is_chronologically_ordered(self, path):
        """Check if a path maintains chronological ordering of nodes."""
        for i in range(len(path) - 1):
            time_i = convert_time(self.graph.nodes[path[i]]["time"])
            time_j = convert_time(self.graph.nodes[path[i+1]]["time"])
            if time_j <= time_i:
                return False
        return True
    
    def is_valid_path(self, path):
        """Check if a path is valid in the graph (edges exist between consecutive nodes)."""
        for i in range(len(path) - 1):
            if path[i+1] not in self.graph[path[i]]:
                return False
        return True

    def tabu_search(self, start, stops, departure_time):
        """Main function to search for the best path."""
        self.departure_time = departure_time
        self.tabu_tenure = self.get_tabu_size_log(len(stops)) if self.tabu_tenure is None else self.tabu_tenure
        
        start_instances = preprocess_stop_instances(self.graph, start, convert_time(departure_time))
        if not start_instances:
            raise ValueError(f"No valid instances found for start stop: {start}")
        start_instances.sort()
        current_path = find_path_with_nodes(self.graph, stops, start, departure_time)
        if not current_path:
            raise ValueError("No valid initial solution found with given time constraints.")

        best_path = current_path
        best_cost = self.cost(best_path)

        tabu_list = {}
        aspiration_values = {}  
        iteration = 0
        no_improve_limit = 5
        no_improve_count = 0
        
        print(f"Initial solution cost: {best_cost}")

        while iteration < self.max_iterations:
            all_neighbors = self.generate_neighbors(current_path, required_stops=stops)
            
            tabu_neighbors = []
            non_tabu_neighbors = []
            
            for neighbor in all_neighbors:
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple in tabu_list:
                    neighbor_cost = self.cost(neighbor)
                    
                    move_key = self.get_move_structure(current_path, neighbor)
                    if move_key not in aspiration_values or neighbor_cost < aspiration_values[move_key]:
                        aspiration_values[move_key] = neighbor_cost
                    
                    if self.use_aspiration and neighbor_cost < best_cost:
                        print(f"Aspiration applied! Tabu move accepted with cost: {neighbor_cost}")
                        non_tabu_neighbors.append(neighbor)
                    else:
                        tabu_neighbors.append(neighbor)
                else:
                    non_tabu_neighbors.append(neighbor)
            
            if non_tabu_neighbors:
                best_neighbor = min(non_tabu_neighbors, key=self.cost)
            elif tabu_neighbors and iteration % 10 == 0:  
                print("Using tabu move for diversification")
                best_neighbor = min(tabu_neighbors, key=self.cost)
            else:
                print("No valid neighbors found.")
                break

            best_neighbor_cost = self.cost(best_neighbor)
            print(f"Iteration {iteration}: Best neighbor cost: {best_neighbor_cost}") 
            
            if best_neighbor_cost < best_cost:
                best_path = best_neighbor
                best_cost = best_neighbor_cost
                no_improve_count = 0
                print(f"New best solution found! Cost: {best_cost}")
            else:
                no_improve_count += 1  

            if no_improve_count >= no_improve_limit:
                print("Stopping early due to stagnation.")
                break  

            current_path = best_neighbor
            
            move_key = self.get_move_structure(current_path, best_neighbor)
            tabu_list[tuple(best_neighbor)] = iteration + self.tabu_tenure
            
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}

            iteration += 1

        return best_path, best_cost
    
    def get_move_structure(self, current_path, new_path):
        """
        Extract a representation of the move structure (what changed between paths).
        This helps identify similar moves for aspiration criteria.
        """
        # proóbkowanie
        differences = []
        for i in range(min(len(current_path), len(new_path))):
            if current_path[i] != new_path[i]:
                differences.append((i, normalize_name(current_path[i]), normalize_name(new_path[i])))
        
        return tuple(differences)
        
    def diversify(self, current_path, tabu_list, required_stops, num_attempts=5):
        """
        Create a diversified solution when search stagnates.
        """
        for _ in range(num_attempts):
            new_path = find_path_with_nodes(self.graph, required_stops, normalize_name(current_path[0]), self.departure_time)
            
            if new_path and tuple(new_path) not in tabu_list:
                return new_path
                
        perturbed = self.perturb_solution(current_path, required_stops)
        return perturbed
    
    def perturb_solution(self, path, required_stops):
        """Apply a strong perturbation to the current solution to escape local optima."""
        required_stop_names = set(normalize_name(s) for s in required_stops)
        
        replaceable_indices = [i for i in range(1, len(path)-1) 
                             if normalize_name(path[i]) not in required_stop_names]
        
        if not replaceable_indices:
            return path  
            
        num_to_replace = max(1, int(len(replaceable_indices) * 0.3))
        indices_to_replace = random.sample(replaceable_indices, num_to_replace)
        
        new_path = path.copy()
        
        for idx in indices_to_replace:
            prev_time = convert_time(self.graph.nodes[new_path[idx-1]]["time"])
            next_time = convert_time(self.graph.nodes[new_path[idx+1]]["time"])
            
            potential_replacements = []
            for node in self.graph.nodes:
                node_time = convert_time(self.graph.nodes[node]["time"])
                if prev_time < node_time < next_time:
                    if new_path[idx-1] in self.graph.predecessors(node) and node in self.graph.predecessors(new_path[idx+1]):
                        potential_replacements.append(node)
            
            if potential_replacements:
                new_path[idx] = random.choice(potential_replacements)
        
        if self.is_valid_path(new_path):
            return new_path
        return path 

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
    ts = TabuSearch(G, cost_type="weight", tabu_tenure=5, max_iterations=100, use_aspiration=True)

    start_stop = "Chłodna"
    stops_list = [start_stop]+["Wiejska", "FAT", "Paprotna", "Chłodna"]
    arrival_time_at_start = "3:30:00"  

    best_path, best_cost = ts.tabu_search(start_stop, stops_list, arrival_time_at_start)
    print("Optimal Path:", best_path)
    print("Total Cost (Transfers or Time):", best_cost)
