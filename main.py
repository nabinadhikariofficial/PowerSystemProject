from generator import DataProcessor
import random

df = DataProcessor("data_full_trunc")

print(df.data['Total_MVA'].describe())


load_ran = random.randrange(
    df.data['Total_MVA'].min(), df.data['Total_MVA'].max(), 1)
print(load_ran)
print(df.data.index[0])

for data in df.data['Total_MVA']:
    pass
