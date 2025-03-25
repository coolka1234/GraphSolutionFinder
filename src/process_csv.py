from turtle import pos
import networkx as nx
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pyparsing import line

df_connection_graph=pd.read_csv('data/connection_graph.csv', low_memory=False)
df_test=df_connection_graph.head(800000)
G=nx.Graph()


def convert_time_from_weird(time_str):
    if time_str[0]=='2' and int(time_str[1])>3:
        return "0"+time_str[1:]
    elif time_str[0]=='3' and int(time_str[1])==0:
        return "06"+time_str[2:]
    elif time_str[0]=='3' and int(time_str[1])==1:
        return "07"+time_str[2:]
    elif time_str[0]=='3' and int(time_str[1])==2:
        return "08"+time_str[2:]
    elif time_str[0]=='3' and int(time_str[1])==3:
        return "09"+time_str[2:]
    elif time_str[0]=='4' and int(time_str[1])>4:
        return "2"+time_str[1:]
    elif time_str[0]=='5' and int(time_str[1])>4:
        return "3"+time_str[1:]
    elif time_str[0]=='6' and int(time_str[1])>4:
        return "4"+time_str[1:]
    elif time_str[0]=='7' and int(time_str[1])>4:
        return "5"+time_str[1:]
    elif time_str[0]=='8' and int(time_str[1])>4:
        return "6"+time_str[1:]
    elif time_str[0]=='9' and int(time_str[1])>4:
        return "7"+time_str[1:]
    else:
        return time_str



def convert_time(time_str):
    # print(time_str)
    if type(time_str) == datetime:
        return time_str
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
        pos_start_x=row['start_stop_lat']
        pos_start_y=row['start_stop_lon'] 
        pos_end_x=row['end_stop_lat']
        pos_end_y=row['end_stop_lon']

        start_node = f"{start_stop}@{row['departure_time']}_{line}"
        end_node = f"{end_stop}@{row['arrival_time']}_{line}" 

        G.add_node(start_node, stop=start_stop, line=line, time=dep_time, pos=(pos_start_x, pos_start_y))
        G.add_node(end_node, stop=end_stop, line=line, time=arr_time, pos=(pos_end_x, pos_end_y))

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
    
    # add_connections_between_the_same_stop(G)
    return G

def add_connections_between_the_same_stop(G: nx.DiGraph):
    already_added=set()
    for nodes in G.nodes():
        if nodes not in already_added:
            for nodes2 in G.nodes():
                if nodes!=nodes2 and G.nodes[nodes]['stop']==G.nodes[nodes2]['stop']:
                    G.add_edge(nodes, nodes2, weight=0, type="transfer")
            already_added.add(nodes)

def read_and_return_with_loc_line_and_time(data_frame):
    G = nx.DiGraph()
    for _, row in data_frame.iterrows():
        G.add_node(row['start_stop'], pos=(row['start_stop_lat'], row['start_stop_lon']), line=row['line'], start_time=row['departure_time'])
        G.add_edge(row['start_stop'], row['end_stop'], weight=((datetime.strptime(convert_time_from_weird(row['departure_time']), '%H:%M:%S')
            - datetime.strptime(convert_time_from_weird(row['arrival_time']), '%H:%M:%S')).seconds // 60), line=row['line'])
    return G
        

def visualize(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=200, font_size=10)
    nx.draw_networkx_edge_labels(G, pos)
    plt.show()


if __name__ == '__main__':
    visualize(read_with_loc_line_and_time(df_test))
