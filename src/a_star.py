from process_csv import read_and_return, df_test, read_and_return_with_loc_and_line, convert_time, read_and_visualize_with_loc_and_line, read_with_loc_line_and_time
import networkx as nx
import heapq
from math import radians, sin, cos, sqrt, atan2
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
        lat2, lon2 = self.graph.nodes[end_nodes]["pos"]
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
                    # print(f'end={end}')
                    priority = new_total_time + self.heuristic(neighbor, end[0])  # f = g + h
                    heapq.heappush(pq, (priority, new_time, neighbor))

        return None
    def a_star_with_line(self, start, end):
        """A* search algorithm optimizing for minimum line changes."""
        pq = []
        heapq.heappush(pq, (0, start))  # Start with 0 line changes
        dist = {node: float("inf") for node in self.graph.nodes}
        dist[start] = 0
        pred = {}  # To reconstruct the path
        visited = set()

        while pq:
            line_changes, node = heapq.heappop(pq)

            if node == end:
                return self.reconstruct_path(pred, start, end)  # Reconstruct the path if we reached the end

            if node in visited:
                continue

            visited.add(node)

            for neighbor in self.graph.neighbors(node):
                edge = self.graph[node][neighbor]
                # If we're on a different line, consider it a line change (penalty = 1)
                line_change_penalty = 1 if self.graph.nodes[node]["line"] != self.graph.nodes[neighbor]["line"] else 0

                new_line_changes = line_changes + line_change_penalty
                if new_line_changes < dist[neighbor]:  # If this is a better (fewer line changes) path
                    dist[neighbor] = new_line_changes
                    pred[neighbor] = node
                    f_cost = new_line_changes + 0  # f = g + h
                    heapq.heappush(pq, (f_cost, neighbor))

        return None  # Return None if no path found
    
    def reconstruct_paths(self, pred, start, end):
        if end not in pred:
            print("No valid route found!")
            return []

        path = []
        current = end

        while current != start:
            path.append(current)
            current = pred.get(current, None)
            if current is None:
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
        return path

    # def reconstruct_paths(self, pred, start, end):
    #     """Reconstructs the shortest path from `start` to `end` using the predecessor dictionary."""
    #     path = []
    #     node = end

    #     while node is not None:
    #         path.append(node)
    #         node = pred.get(node)  # Move to the predecessor

    #     path.reverse()  # Reverse to get the correct order from start → end

    #     if path[0] != start:
    #         return None  # No valid path found

    #     return path


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
    
if __name__ == '__main__':
    G = read_with_loc_line_and_time(df_test)
    a_star = A_Star(G)
    arg1, arg2, arg3 = 'Chłodna', 'Różanka', '05:29:00'
    # arg1, arg2, arg3 = "Paprotna", "Broniewskiego", '20:00:00'
    start, end = a_star.get_start_and_end_nodes(arg1, arg2, arg3)
    print(start, end)
    print(a_star.a_star_with_line(G, start, end, arg3))