# Description: Djikstra's algorithm for finding the shortest path in a graph
from datetime import datetime
from itertools import count
import sys
import networkx as nx
import heapq
from collections import deque
from heapq import heappop, heappush
from process_csv import convert_time, read_with_loc_line_and_time, df_test
class Djikstra():
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.visited = set()

    def dijkstra_with_time(self, start: str, end, departure_time: str):
        G = self.graph
        start_time = convert_time(departure_time)

        pq = []
        heapq.heappush(pq, (0, start_time, start))

        dist = {node: float("inf") for node in G.nodes}
        dist[start] = 0
        pred = {}

        while pq:
            total_time, curr_time, node = heapq.heappop(pq)

            if node in end:
                print(f"Found end node: {node}")
                end= [node]
                break

            for neighbor in G.neighbors(node):
                edge = G[node][neighbor]
                travel_time = edge["weight"]
                neighbor_time = G.nodes[neighbor]["time"]

                if neighbor_time < curr_time:
                    continue

                new_time = neighbor_time
                new_total_time = total_time + (new_time - curr_time).seconds // 60

                if new_total_time < dist[neighbor]:
                    dist[neighbor] = new_total_time
                    pred[neighbor] = node
                    heapq.heappush(pq, (new_total_time, new_time, neighbor))

        return self.reconstruct_paths(pred, start, end)
    
    def reconstruct_paths(self, pred, start, end):
        # print(pred)
        print(f"printing end {end}")
        if [i for i in end if i in pred.values()]:
            print("No valid route found!")
            return []
        path = []

        current = end[0]

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
        self.print_to_err_diff(path)
        return path
    
    def get_start_and_end_nodes(self, start, end, departure_time):
        start_nodes = [node for node in self.graph.nodes if node.startswith(f"{start}@")]
        start_nodes.sort(key=lambda x: datetime.time(self.graph.nodes[x]["time"]))  

        print(f"Available departures from {start}: {', '.join([node.split('@')[1].split('_')[0] for node in start_nodes])}")
        print(f"Departure time: {datetime.time(convert_time(departure_time))}")
        start_node = None
        for node in start_nodes:
            if datetime.time(self.graph.nodes[node]["time"]) >= datetime.time(convert_time(departure_time)):
                print(f"Starting from {start} at {datetime.time(self.graph.nodes[node]['time'])}, because it's the first available departure from {start} after {datetime.time(convert_time(departure_time))}.")
                start_node = node
                break

        if start_node is None:
            print(f"No available departures from {start} at this time.")
        else:
            end_nodes = [node for node in self.graph.nodes if node.startswith(f"{end}@") and datetime.time((self.graph.nodes[node]["time"])) >= datetime.time(convert_time(departure_time))]
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

def run_djikstra(start, end, departure_time):
    G = read_with_loc_line_and_time(df_test)
    djikstra = Djikstra(G)
    start, end = djikstra.get_start_and_end_nodes(start, end, departure_time)
    djikstra.dijkstra_with_time(start, end, departure_time)

def test_run_djikstra():
    run_djikstra('Chłodna', 'Różanka', '5:29:00')
    # TESTCASE 1: (Chłodna, Różanka)

if __name__=='__main__':
    G = read_with_loc_line_and_time(df_test)
    djikstra = Djikstra(G)
    start, end= djikstra.get_start_and_end_nodes('Chłodna', 'Różanka', '10:29:00')
    print(start, end)
    ((djikstra.dijkstra_with_time(start, end, '10:29:00'), 'Chłodna', 'Różanka'))
    # TESTCASE 1: (Chłodna, Różanka)
