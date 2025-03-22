# Description: Djikstra's algorithm for finding the shortest path in a graph
from itertools import count
import networkx as nx
import heapq
from collections import deque
from heapq import heappop, heappush
from src.process_csv import convert_time, read_with_loc_line_and_time, df_test
class Djikstra():
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.visited = set()
    
    # def shortest_path(self, start: str, end: str):
    #     dist = {node: float('inf') for node in self.graph.nodes}
    #     dist[start] = 0
    #     while len(self.visited) < len(self.graph.nodes):
    #         node = self._min_distance(dist)
    #         self.visited.add(node)
    #         print(f"Visited: {node} at distance {dist[node]}")
    #         for neighbor in self.graph.neighbors(node):
    #             if neighbor not in self.visited:
    #                 new_dist = dist[node] + self.graph[node][neighbor]['weight']
    #                 if new_dist < dist[neighbor]:
    #                     dist[neighbor] = new_dist
    #     return dist[end]
    
    # def shortest_path_with_path(self, start: str, end: str):
    #     dist = {node: float('inf') for node in self.graph.nodes}
    #     dist[start] = 0
    #     pred = {node: [] for node in self.graph.nodes}
    #     self.visited = set()

    #     while len(self.visited) < len(self.graph.nodes):
    #         node = self._min_distance(dist)
    #         self.visited.add(node)
    #         line = self.graph.nodes[node].get('line', 'Unknown')

    #         print(f"Visited: {node} (Line: {line}), at distance {dist[node]}")

    #         for neighbor in self.graph.neighbors(node):
    #             if neighbor not in self.visited:
    #                 new_dist = dist[node] + self.graph[node][neighbor]['weight']
    #                 if new_dist < dist[neighbor]:
    #                     dist[neighbor] = new_dist
    #                     pred[neighbor] = [node]  
    #                 elif new_dist == dist[neighbor]:
    #                     pred[neighbor].append(node)  

    #     paths = self.reconstruct_paths(pred, start, end)
        
    #     return paths


    # def reconstruct_paths(self, pred, start, end):
    #     """Reconstruct the path from start to end, including line changes."""
    #     paths = []
    #     current = end

    #     while current != start:
    #         if current not in pred or not pred[current]:
    #             print("No path found!")
    #             return []

    #         previous = pred[current][0]  
    #         paths.append((previous, current))
    #         current = previous  

    #     paths.reverse()  

    #     print("\nOptimal Route:")
    #     prev_line = None
    #     for i, (start_stop, end_stop) in enumerate(paths):
    #         start_line = self.graph.nodes[start_stop].get("line", "Unknown")
    #         end_line = self.graph.nodes[end_stop].get("line", "Unknown")

    #         if start_line != prev_line:
    #             print(f"🚏 Take {start_line} from {start_stop}", end=" ")

    #         print(f"→ {end_stop}", end=" ")

    #         if end_line != start_line:
    #             print(f"(Switch to {end_line})")

    #         prev_line = end_line

    #     print("\n")
    #     return paths


    
    # def deconstruct_path(self, pred, start, end):
    #     path = deque([end])
    #     while path[0] != start:
    #         path.appendleft(pred[path[0]][0])
    #     return path
    
    # def _min_distance(self, dist):
        # min_dist = float('inf')
        # min_node = None
        # for node in dist:
        #     if node not in self.visited and dist[node] < min_dist:
        #         min_dist = dist[node]
        #         min_node = node
        # return min_node

    @staticmethod
    def multi_source_dijkstra(G, sources, pred=None, paths=None, target=None, cutoff=None, weight="weight"):
        G_succ = G._adj  
        push = heappush
        pop = heappop
        dist = {}  
        seen = {}
        c = count()
        fringe = []
        for source in sources:
            seen[source] = 0
            push(fringe, (0, next(c), source))
        while fringe:
            (d, _, v) = pop(fringe)
            if v in dist:
                continue  
            dist[v] = d
            if v == target:
                break
            for u, e in G_succ[v].items():
                cost = weight(v, u, e)
                if cost is None:
                    continue
                vu_dist = dist[v] + cost
                if cutoff is not None:
                    if vu_dist > cutoff:
                        continue
                if u in dist:
                    u_dist = dist[u]
                    if vu_dist < u_dist:
                        raise ValueError("Contradictory paths found:", "negative weights?")
                    elif pred is not None and vu_dist == u_dist:
                        pred[u].append(v)
                elif u not in seen or vu_dist < seen[u]:
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
                    if paths is not None:
                        paths[u] = paths[v] + [u]
                    if pred is not None:
                        pred[u] = [v]
                elif vu_dist == seen[u]:
                    if pred is not None:
                        pred[u].append(v)
        return dist
    

    def dijkstra_with_time(self, start: str, end: str, departure_time: str):
        G = self.graph
        start_time = convert_time(departure_time)

        pq = []
        heapq.heappush(pq, (0, start_time, start))

        dist = {node: float("inf") for node in G.nodes}
        dist[start] = 0
        pred = {}

        while pq:
            total_time, curr_time, node = heapq.heappop(pq)

            if node == end:
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
        return start_node, end_nodes[0]

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
    start, end= djikstra.get_start_and_end_nodes('Chłodna', 'Różanka', '5:29:00')
    print(start, end)
    ((djikstra.dijkstra_with_time(start, end, '5:29:00'), 'Chłodna', 'Różanka'))
    # TESTCASE 1: (Chłodna, Różanka)
