import networkx as nx
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

df_connection_graph=pd.read_csv('data/connection_graph.csv')
df_test=df_connection_graph.head(1000)
G=nx.Graph()


def convert_time(time_str):
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
        G.add_edge(row['start_stop'], row['end_stop'], weight=((datetime.strptime(convert_time(row['departure_time']), '%H:%M:%S')
            - datetime.strptime(convert_time(row['arrival_time']), '%H:%M:%S')).seconds // 60))

    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(G, pos)
    plt.show()

def read_and_return(data_frame):
    for _, row in data_frame.iterrows():
        G.add_edge(row['start_stop'], row['end_stop'], weight=((datetime.strptime(convert_time(row['departure_time']), '%H:%M:%S')
            - datetime.strptime(convert_time(row['arrival_time']), '%H:%M:%S')).seconds // 60))
    return G

def read_and_return_with_loc_and_line(data_frame):
    for _, row in data_frame.iterrows():
        G.add_edge(row['start_stop'], row['end_stop'], weight=((datetime.strptime(convert_time(row['departure_time']), '%H:%M:%S')
            - datetime.strptime(convert_time(row['arrival_time']), '%H:%M:%S')).seconds // 60))
        G.add_node(row['start_stop'], pos=(row['start_stop_lat'], row['start_stop_lon']))
        G.add_node(row['end_stop'], pos=(row['end_stop_lat'], row['end_stop_lon']))
        G.add_node(row['start_stop'], line=row['line'])
        G.add_node(row['end_stop'], line=row['line'])
    return G

if __name__ == '__main__':
    read_and_visualize(df_test)
    print(read_and_return(df_test))
