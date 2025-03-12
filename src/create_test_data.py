import pandas
import random
from process_csv import read_and_return_with_loc_and_line
generator=random.Random(42)
df_test_1 = pandas.read_csv('data/connection_graph.csv', skiprows=generator.randint(0, 1000), nrows=10)
df_test_2=pandas.read_csv('data/connection_graph.csv', skiprows=generator.randint(1000, 4000), nrows=10)
df_test_3=pandas.read_csv('data/connection_graph.csv', skiprows=generator.randint(4000, 12000), nrows=10)

def return_test_data():
    return read_and_return_with_loc_and_line(pandas.concat([df_test_1, df_test_2, df_test_3]))

def n_random_stops(n):
    together=pandas.concat([df_test_1, df_test_2, df_test_3])
    stops=together['start_stop'].unique()
    return generator.sample(list(stops), n)