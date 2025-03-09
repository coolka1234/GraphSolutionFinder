import random
import itertools
import networkx as nx

class TabuSearch:
    def __init__(self, graph, cost_type="weight", tabu_tenure=5, max_iterations=100):
        """
        Algorytm Tabu Search dla problemu najkrótszej trasy przez przystanki.
        
        :param graph: Graf przystanków (NetworkX)
        :param cost_type: "time" (czas przejazdu) lub "transfers" (liczba przesiadek)
        :param tabu_tenure: Liczba iteracji, przez które ruchy pozostają zakazane
        :param max_iterations: Maksymalna liczba iteracji algorytmu
        """
        self.graph = graph
        self.cost = self.cost_weight if cost_type == "weight" else self.line_cost
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
    # nie zadziala dla time, bo nie ma takiego klucza w grafie, jest weight
    def cost_weight(self, path):
        """Oblicza koszt trasy w zależności od wybranego kryterium."""
        return sum(self.graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
    
    def line_cost(self, path):
        """Oblicza koszt trasy w zależności od liczby przesiadek."""
        return sum(1 for i in range(len(path)-1) if self.graph[path[i]][path[i+1]]['line'] != self.graph[path[i+1]][path[i+2]]['line'])

    def generate_neighbors(self, path):
        """Generuje sąsiednie rozwiązania przez zamianę dwóch losowych węzłów (swap)."""
        neighbors = []
        for i, j in itertools.combinations(range(1, len(path)-1), 2):  
            new_path = path[:]
            new_path[i], new_path[j] = new_path[j], new_path[i]
            neighbors.append(new_path)
        return neighbors

    def tabu_search(self, start, stops):
        """Główna funkcja wyszukująca najlepszą trasę."""
        best_path = [start] + stops + [start]  
        best_cost = self.cost(best_path)

        current_path = best_path
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
    
if __name__=='__main__':
    from process_csv import read_and_return_with_loc_and_line, df_test
    G = read_and_return_with_loc_and_line(df_test)
    ts = TabuSearch(G, cost_type="transfers", tabu_tenure=5, max_iterations=100)
    result=ts.tabu_search('Czajkowskiego', ['Stanki', 'Poprzeczna', 'Paprocka'])
    print(result)

