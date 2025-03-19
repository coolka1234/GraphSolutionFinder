# AI & Knowledge Engineering - Optimization Lab  


## Overview  
This project is part of an AI and Knowledge Engineering course, focusing on optimization problems and heuristic search methods. The aim is to implement various pathfinding algorithms, such as Dijkstra’s algorithm, A* search, and Tabu Search, to find the optimal routes in a public transportation network.  

## Objectives  
- Understand optimization problems and their challenges.  
- Implement heuristic and approximation techniques for pathfinding.  
- Compare A* and Dijkstra’s algorithms in terms of efficiency.  
- Use Tabu Search to solve the Traveling Salesman Problem (TSP).  

## Theoretical Background  
The project involves:  
- **Optimization problems**: Defining constraints and objective functions.  
- **A* Algorithm**: A heuristic search method combining path cost and estimated distance to the goal.  
- **Dijkstra’s Algorithm**: A shortest-path algorithm for graphs with non-negative weights.  
- **Tabu Search**: A metaheuristic technique for avoiding local optima in search problems.  

## Implementation Details  
The project uses a dataset containing public transportation routes and schedules in Wrocław. The task is to develop algorithms that compute the shortest or most efficient paths based on travel time or the number of transfers.  

### Features  
- Shortest path computation using **Dijkstra’s and A*** algorithms.  
- Route optimization considering **either travel time or the number of transfers**.  
- Implementation of **Tabu Search** to solve an extended version of the problem with multiple destinations.  
- Performance evaluation based on computational time and solution quality.  

## Dataset  
The project uses **connection_graph.csv**, which contains:  
- Stop and route details (stop names, coordinates).  
- Public transport schedules (departure and arrival times).  
- Operator and line information.  

## Requirements  
- Python 3.x  
- Libraries: `numpy`, `pandas`, `networkx`, `matplotlib`  

## Usage  
Run the script with the following parameters:  
```bash
python main.py --start "Stop_A" --end "Stop_B" --criteria "time" --departure "08:00"
```  
- `--criteria` can be **"time"** for shortest travel time or **"transfers"** for the fewest number of transfers.  

For solving the Traveling Salesman Problem:  
```bash
python tsp_solver.py --start "Stop_A" --stops "Stop_B;Stop_C;Stop_D" --criteria "time"
```  

## Evaluation  
- The efficiency of **A*** and **Dijkstra’s** algorithms will be compared in terms of execution time.  
- **Tabu Search** will be analyzed based on solution quality and convergence speed.  

## Report  
The final report will include:  
- A theoretical overview of the methods used.  
- Implementation details and modifications made.  
- Performance analysis with benchmarks.  
- Challenges encountered and solutions applied.  

## References  
- Original research papers on **A*** and **Tabu Search**.  
- Documentation on **heuristic search techniques**.  
- Open data from **Wrocław public transport**.  

This project provides a hands-on approach to solving real-world optimization problems using heuristic methods and graph algorithms.
