import networkx as nx
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pyparsing import line

df_connection_graph=pd.read_csv('data/connection_graph.csv')
df_test=df_connection_graph.head(40000)
G=nx.Graph()


def convert_time_from_weird(time_str):
    if time_str[0]=='2':
        return "0"+time_str[1:]
    elif time_str[0]=='3':
        return "1"+time_str[1:]
    elif time_str[0]=='4':
        return "2"+time_str[1:]
    elif time_str[0]=='5':
        return "3"+time_str[1:]
    else:
        return time_str

def read_and_visualize(data_frame):
    for _, row in data_frame.iterrows():
        G.add_edge(row['start_stop'], row['end_stop'], weight=((datetime.strptime(convert_time_from_weird(row['departure_time']), '%H:%M:%S')
            - datetime.strptime(convert_time_from_weird(row['arrival_time']), '%H:%M:%S')).seconds // 60))

    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(G, pos)
    plt.show()

def read_and_return(data_frame):
    for _, row in data_frame.iterrows():
        G.add_edge(row['start_stop'], row['end_stop'], weight=((datetime.strptime(convert_time_from_weird(row['departure_time']), '%H:%M:%S')
            - datetime.strptime(convert_time_from_weird(row['arrival_time']), '%H:%M:%S')).seconds // 60))
    return G

# TODO - add line to the graph
def read_and_return_with_loc_and_line(data_frame):
    for _, row in data_frame.iterrows():
        G.add_node(row['start_stop'], pos=(row['start_stop_lat'], row['start_stop_lon']), line=row['line'])
        G.add_node(row['end_stop'], pos=(row['end_stop_lat'], row['end_stop_lon']), line=row['line'])
        G.add_edge(row['start_stop'], row['end_stop'], weight=((datetime.strptime(convert_time_from_weird(row['departure_time']), '%H:%M:%S')
            - datetime.strptime(convert_time_from_weird(row['arrival_time']), '%H:%M:%S')).seconds // 60))
    return G
# TODO - this doesnt work, because it adds every line to evey node
def read_and_visualize_with_loc_and_line(data_frame):
    set_of_lines=set()
    for _, row in data_frame.iterrows():
        G.add_node(row['start_stop'], pos=(row['start_stop_lat'], row['start_stop_lon']), line=row['line'], start_time=row['departure_time'])
        G.add_node(row['end_stop'], pos=(row['end_stop_lat'], row['end_stop_lon']), line=row['line'], stop_time=row['arrival_time'])
        G.add_edge(row['start_stop'], row['end_stop'], weight=((datetime.strptime(convert_time_from_weird(row['departure_time']), '%H:%M:%S')
            - datetime.strptime(convert_time_from_weird(row['arrival_time']), '%H:%M:%S')).seconds // 60))
    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(G, pos)
    plt.show()


def convert_time(time_str):
    return datetime.strptime(convert_time_from_weird(time_str), "%H:%M:%S")

def read_with_loc_line_and_time(data_frame):
    G = nx.DiGraph()
    stop_times = {}

    for _, row in data_frame.iterrows():
        line = row['line']
        start_stop = row['start_stop']
        end_stop = row['end_stop']
        dep_time = convert_time(row['departure_time'])
        arr_time = convert_time(row['arrival_time'])
        travel_time = (arr_time - dep_time).seconds // 60

        start_node = f"{start_stop}@{row['departure_time']}_{line}"
        end_node = f"{end_stop}@{row['arrival_time']}_{line}"

        G.add_node(start_node, stop=start_stop, line=line, time=dep_time)
        G.add_node(end_node, stop=end_stop, line=line, time=arr_time)

        G.add_edge(start_node, end_node, weight=travel_time, travel_time=travel_time, type="travel")

        if start_stop not in stop_times:
            stop_times[start_stop] = []
        stop_times[start_stop].append((dep_time, start_node))

    for stop, times in stop_times.items():
        times.sort()
        for i in range(len(times) - 1):
            time1, node1 = times[i]
            time2, node2 = times[i + 1]
            wait_time = (time2 - time1).seconds // 60
            G.add_edge(node1, node2, weight=wait_time, type="transfer")

    return G

def visualize(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=200, font_size=10)
    nx.draw_networkx_edge_labels(G, pos)
    plt.show()


if __name__ == '__main__':
    (read_with_loc_line_and_time(df_test))
