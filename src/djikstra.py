# Description: Djikstra's algorithm for finding the shortest path in a graph
import networkx as nx
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
    
    def _min_distance(self, dist):
        min_dist = float('inf')
        min_node = None
        for node in dist:
            if node not in self.visited and dist[node] < min_dist:
                min_dist = dist[node]
                min_node = node
        return min_node

if __name__=='__main__':
    from process_csv import read_and_return
    G = read_and_return()
    djikstra = Djikstra(G)
    print(djikstra.shortest_path('Kolumba', 'Tczewska'))