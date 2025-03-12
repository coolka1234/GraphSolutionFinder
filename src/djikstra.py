# Description: Djikstra's algorithm for finding the shortest path in a graph
from itertools import count
import networkx as nx

from collections import deque
from heapq import heappop, heappush
class Djikstra():
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.visited = set()
    
    def shortest_path(self, start: str, end: str):
        dist = {node: float('inf') for node in self.graph.nodes}
        dist[start] = 0
        while len(self.visited) < len(self.graph.nodes):
            node = self._min_distance(dist)
            self.visited.add(node)
            print(f"Visited: {node} at distance {dist[node]}")
            for neighbor in self.graph.neighbors(node):
                if neighbor not in self.visited:
                    new_dist = dist[node] + self.graph[node][neighbor]['weight']
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
        return dist[end]
    
    def shortest_path_with_path(self, start: str, end: str):
        dist = {node: float('inf') for node in self.graph.nodes}
        dist[start] = 0
        pred = {node: [] for node in self.graph.nodes}
        self.visited = set()

        while len(self.visited) < len(self.graph.nodes):
            node = self._min_distance(dist)
            self.visited.add(node)
            print(f"dists: {dist} at node {node}")
            print(f"Visited: {node} at distance {dist[node]}")

            for neighbor in self.graph.neighbors(node):
                if neighbor not in self.visited:
                    new_dist = dist[node] + self.graph[node][neighbor]['weight']
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        pred[neighbor] = [node]  
                    elif new_dist == dist[neighbor]:
                        pred[neighbor].append(node)  

        paths = self.reconstruct_paths(pred, start, end)
        
        return dist[end], paths

    
    def reconstruct_paths(self, pred, start, end, path=None, all_paths=None):
        if path is None:
            path = [end]
        if all_paths is None:
            all_paths = []

        if end == start:
            all_paths.append(path[::-1]) 
            return all_paths

        for predecessor in pred[end]:
            self.reconstruct_paths(pred, start, predecessor, path + [predecessor], all_paths)

        return all_paths

    
    def deconstruct_path(self, pred, start, end):
        path = deque([end])
        while path[0] != start:
            path.appendleft(pred[path[0]][0])
        return path
    
    def _min_distance(self, dist):
        min_dist = float('inf')
        min_node = None
        for node in dist:
            if node not in self.visited and dist[node] < min_dist:
                min_dist = dist[node]
                min_node = node
        return min_node

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

if __name__=='__main__':
    from process_csv import read_and_return, df_test, read_and_return_with_loc_and_line
    G = read_and_return_with_loc_and_line(df_test)
    djikstra = Djikstra(G)
    # print(djikstra.multi_source_dijkstra(G,'Czajkowskiego', target='Krucza'))
    # print(djikstra.shortest_path('Czajkowskiego', 'Krucza'))
    #TODO so the problem is that it doesnt see a combination between lines at all. Bałtycka_A and Bałtycka_B are not connected whatsoever
    print((djikstra.shortest_path_with_path('Bałtycka_A', 'Pola_A')[1], 'Bałtycka_A', 'Pola_A'))
