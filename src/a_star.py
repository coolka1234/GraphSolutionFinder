from process_csv import read_and_return, df_test, read_and_return_with_loc_and_line
import networkx as nx
import heapq
class A_Star():
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.heuristic = self.heuristic_euclidean

    def heuristic_optimal(self, node, target):
        return nx.shortest_path_length(self.graph, source=node, target=target, weight='weight')
    
    def heuristic_euclidean(self, node, target):
        pos = nx.get_node_attributes(self.graph, 'pos')
        x1, y1 = pos[node]
        x2, y2 = pos[target]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def heuristic_line_change(self, node, target):
        line = nx.get_node_attributes(self.graph, 'line') 
        line1=line[node]
        line2= line[target]
        return 0 if line1==line2 else 1
    
    def shortest_path_a_star_line_change(self, start, end, heuristic):
        self.heuristic = heuristic
        pq = []  
        heapq.heappush(pq, (0, start))  
        
        dist = {node: float('inf') for node in self.graph.nodes}
        dist[start] = 0
        
        pred = {node: [] for node in self.graph.nodes}
        
        while pq:
            _, node = heapq.heappop(pq)  
            
            if node == end:  
                break
            
            for neighbor in self.graph.neighbors(node):
                weight = self.graph[node][neighbor].get('weight', 1)  
                new_dist = dist[node] + weight
                
                if new_dist < dist[neighbor]:  
                    dist[neighbor] = new_dist
                    pred[neighbor] = [node]
                    f_score = new_dist + self.heuristic(neighbor, end)  
                    heapq.heappush(pq, (f_score, neighbor))
                elif new_dist == dist[neighbor]:  
                    pred[neighbor].append(node)
        
        paths = self.reconstruct_paths(pred, start, end)
        return len(paths[0]), paths

    def shortest_path_a_star(self, start, end, heuristic):
        """A* search to find the shortest weighted path."""
        self.heuristic = heuristic
        pq = []  
        heapq.heappush(pq, (0, start))  
        dist = {node: float('inf') for node in self.graph.nodes}
        dist[start] = 0
        pred = {node: [] for node in self.graph.nodes}
        while pq:
            _, node = heapq.heappop(pq)  
            
            if node == end:  
                break
            
            for neighbor in self.graph.neighbors(node):
                weight = self.graph[node][neighbor].get('weight', 1)  
                new_dist = dist[node] + weight
                
                if new_dist < dist[neighbor]:  
                    dist[neighbor] = new_dist
                    pred[neighbor] = [node]
                    f_score = new_dist + self.heuristic(neighbor, end)  
                    heapq.heappush(pq, (f_score, neighbor))
                elif new_dist == dist[neighbor]:  
                    pred[neighbor].append(node)
        
        paths = self.reconstruct_paths(pred, start, end)
        return dist[end], len(paths[1])

    def reconstruct_paths(self, pred, start, end, path=None, all_paths=None):
        """Recursively reconstructs paths from start to end."""
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
    
if __name__ == '__main__':
    G = read_and_return_with_loc_and_line(df_test)
    a_star = A_Star(G)
    # print(a_star.shortest_path_a_star('Czajkowskiego', 'Krucza', a_star.heuristic_optimal))
    print(a_star.shortest_path_a_star('Czajkowskiego', 'Gagarina', a_star.heuristic_euclidean))
    # print(a_star.shortest_path_a_star_line_change('Czajkowskiego', 'Gagarina', a_star.heuristic_line_change))