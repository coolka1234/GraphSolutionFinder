import sys
from process_csv import df_test, convert_time,read_with_loc_line_and_time
import networkx as nx
import heapq
from math import radians, sin, cos, sqrt, atan2

def time_to_minutes(time):
    """Converts datetime to minutes from midnight for easier calculation."""
    return time.hour * 60 + time.minute
class A_Star():
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        # self.heuristic = self.heuristic_euclidean

    def heuristic_optimal(self, node, target):
        return nx.shortest_path_length(self.graph, source=node, target=target, weight='weight')
    

    def haversine(self, lat1, lon1, lat2, lon2):
        """Calculate the great-circle distance (in km) between two points using latitude & longitude."""
        R = 6371  
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def heuristic(self, node, end_nodes, avg_speed=30):
        """
        Estimate the remaining travel time (in minutes) from `node` to `end_nodes`
        based on straight-line distance (Haversine formula) and average speed.
        """
        stop, time_line = node.split("@")
        
        lat1, lon1 = self.graph.nodes[node]["pos"]

        min_time = float('inf')
        lat2, lon2 = self.graph.nodes[end_nodes[0]]["pos"]
        distance_km = self.haversine(lat1, lon1, lat2, lon2)

        estimated_time = (distance_km / avg_speed) * 60  
        min_time = min(min_time, estimated_time)

        return min_time  

    

    def a_star_with_time(self, graph, start, end, departure_time):
        """A* search algorithm optimizing for the shortest travel time."""
        pq = []
        start_time = convert_time(departure_time)
        heapq.heappush(pq, (0, start_time, start))  

        dist = {node: float("inf") for node in graph.nodes}
        dist[start] = 0
        pred = {}

        while pq:
            total_time, curr_time, node = heapq.heappop(pq)

            if node in end:
                return self.reconstruct_paths(pred, start, node)

            for neighbor in graph.neighbors(node):
                edge = graph[node][neighbor]
                travel_time = edge["weight"]
                neighbor_time = graph.nodes[neighbor]["time"]

                if neighbor_time < curr_time:
                    continue  

                new_time = neighbor_time
                new_total_time = total_time + (new_time - curr_time).seconds // 60

                if new_total_time < dist[neighbor]:
                    dist[neighbor] = new_total_time
                    pred[neighbor] = node
                    priority = new_total_time + self.heuristic(neighbor, end[0])  
                    heapq.heappush(pq, (priority, new_time, neighbor))

        return None

    def a_star_with_line(self, graph, start, end, departure_time):
        """
        A* search algorithm that optimizes for minimum line changes,
        and respects arrival time at the start node while preventing cycles.
        """
        pq = []
        start_time = convert_time(departure_time)
        heapq.heappush(pq, (0, 0, start_time, start, frozenset([start])))  

        best_costs = {}
        best_costs[(start, 0)] = 0
        
        pred = {}
        
        current_lines = {start: graph.nodes[start]["line"]}
        
        expanded_states = set()

        while pq:
            priority, line_count, curr_time, node, visited = heapq.heappop(pq)
            
            state_key = (node, line_count)
            if state_key in expanded_states:
                continue
                
            expanded_states.add(state_key)
            
            if node in end:
                print(f"Found path to {node} with {line_count} line changes")
                return self.reconstruct_paths(pred, start, node)

            current_line = current_lines.get(node, graph.nodes[node]["line"])

            for neighbor in graph.neighbors(node):
                if neighbor in visited:
                    continue
                    
                edge = graph[node][neighbor]
                neighbor_time = graph.nodes[neighbor]["time"]
                neighbor_line = graph.nodes[neighbor]["line"]

                if neighbor_time < curr_time:
                    continue  

                new_line_count = line_count
                if current_line != neighbor_line:
                    new_line_count += 1

                new_time = neighbor_time
                time_diff = (new_time - curr_time).total_seconds() / 60  # in minutes
                
                new_cost = time_diff
                neighbor_state = (neighbor, new_line_count)
                
                if neighbor_state not in best_costs or new_cost < best_costs[neighbor_state]:
                    best_costs[neighbor_state] = new_cost
                    
                    pred[neighbor] = node
                    current_lines[neighbor] = neighbor_line
                    
                    estimated_changes = self.estimate_line_changes(neighbor, end[0], neighbor_line)
                    priority = new_line_count + estimated_changes
                    
                    new_visited = frozenset(visited.union([neighbor]))
                    
                    heapq.heappush(pq, (priority, new_line_count, new_time, neighbor, new_visited))
            
            if len(pq) > 10000:
                print(f"Warning: Queue size limit reached, pruning to 5000 most promising states")
                pq = heapq.nsmallest(5000, pq)
                heapq.heapify(pq)

        return None
        
    def estimate_line_changes(self, node, end, current_line):
        """
        Estimate the minimum number of line changes needed to reach the end.
        This is a simple heuristic - returns 1 if the lines are different, 0 otherwise.
        A more sophisticated approach could use a precomputed line change database.
        """
        try:
            end_line = self.graph.nodes[end]["line"]
            if current_line != end_line:
                return 1
            return 0
        except (KeyError, IndexError):
            return 0
    
    def reconstruct_paths(self, pred, start, end):
        """
        Reconstructs the path from start to end node using the predecessor dictionary.
        
        Args:
            pred: Dictionary mapping each node to its predecessor
            start: Starting node
            end: Ending node
            
        Returns:
            The reconstructed path as a list of nodes, or an empty list if no path exists
        """
        if end not in pred and end != start:  
            print("No valid route found!")
            return []

        path = []
        current = end

        while current != start:
            path.append(current)
            print(f"current={current}, start={start}")
            
            current = pred.get(current, None)
            
            if current is None:
                print("Error: Path reconstruction failed - missing predecessor")
                return []
                
            if current in path:
                print("Error: Cycle detected in path reconstruction")
                return []

        path.append(start)
        path.reverse()  
        print("\nOptimal Route:")
        prev_line = None
        prev_time = None

        for node in path:
            stop, time_line = node.split("@")
            time, line = time_line.split("_")

            if prev_line is None or line != prev_line:
                print(f"\n🚏 Take {line} from {stop} at {time}", end=" ")

            if prev_time:
                print(f"→ {stop} at {time}", end=" ")

            if prev_line and line != prev_line:
                print(f"(Switch to {line})")

            prev_line = line
            prev_time = time

        print("\n")
        self.print_to_err_diff(path)
        return path


    def get_start_and_end_nodes(self, start, end, departure_time):
        start_nodes = [node for node in self.graph.nodes if node.startswith(f"{start}@")]
        start_nodes.sort(key=lambda x: self.graph.nodes[x]["time"]) 

        start_node = None
        for node in start_nodes:
            if self.graph.nodes[node]["time"] >= convert_time(departure_time):
                start_node = node
                break

        if start_node is None:
            print(f"No available departures from {start} at this time.")
        else:
            end_nodes = [node for node in self.graph.nodes if node.startswith(f"{end}@")]
        return start_node, end_nodes

    def print_to_err_diff(self, path):
        """Prints the time difference between first and last stop in the path.

        Args:
            path (list[str]): Final optimal path, as a
                list of stop names.
        """
        def extract_time(node):
            return convert_time(node.split("@")[1].split("_")[0])
        
        time_diff = None

        if (len(path) > 1):
            time_diff=extract_time(path[-1])-extract_time(path[0])
            print(f"Time difference between first and last stop: {time_diff}", file=sys.stderr)
        else:
            time_diff=convert_time(path[0].split("@")[1].split("_")[0])
            print(f"Weight cost: {time_diff}", file=sys.stderr)
    
def run_a_star_time(start, end, departure_time):
    G = read_with_loc_line_and_time(df_test)
    a_star = A_Star(G)
    start, end = a_star.get_start_and_end_nodes(start, end, departure_time)
    print(a_star.a_star_with_time(G, start, end, departure_time))

def run_a_star_line(start, end, arrival_time):
    G = read_with_loc_line_and_time(df_test)
    a_star = A_Star(G)
    start, end = a_star.get_start_and_end_nodes(start, end, arrival_time)
    print(a_star.a_star_with_line(G, start, end, arrival_time))
    
if __name__ == '__main__':
    G = read_with_loc_line_and_time(df_test)
    a_star = A_Star(G)
    # arg1, arg2, arg3 = 'Chłodna', 'Różanka', '05:29:00'
    arg1, arg2, arg3 = "Paprotna", "Broniewskiego", '20:00:00'
    start, end = a_star.get_start_and_end_nodes(arg1, arg2, arg3)
    (a_star.a_star_with_line(G, start, end, arg3))