import pandas as pd


class DataProcessor():
    def __init__(self, data_name):
        self.data = pd.read_csv('data/'+data_name+".csv", index_col=0)[:500]
        self.cols = self.data.columns.tolist()
        self.data_name = data_name

    def display(self):
        print(self.data)

    def cutter(self, col_name, point):
        self.data[col_name] = self.data[col_name].round(point)

    def cut(self):
        for col in self.cols:
            if 'MVA' in col:
                self.cutter(col, 0)
            else:
                self.cutter(col, 3)

    def save(self):
        self.data.to_csv('data/'+self.data_name+"_trunc.csv")
