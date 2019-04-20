import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_series(filename):

    csvfile=pd.read_csv(filename)
    # csvreader = csv.reader(csvfile)
    # print(csvfile)
    high = csvfile.iloc[:,1]
    normalized_high = (high - np.mean(high)) / np.std(high)
    # print(high)

    low = csvfile.iloc[:,2]
    normalized_low = (low - np.mean(low)) / np.std(low)

    op = csvfile.iloc[:,3]
    normalized_open = (op - np.mean(op)) / np.std(op)

    value = csvfile.iloc[:,5]
    normalized_value = (value - np.mean(value)) / np.std(value)

    close = csvfile.iloc[:,6]
    normalized_close = (close - np.mean(close)) / np.std(close)

    data=[]
    # print(len(high))
    for i in range(0,len(high)):
        temp=[normalized_high[i],normalized_low[i],normalized_open[i],normalized_value[i],normalized_close[i]]
        data.append(temp)
    # print(len(data))
    return data



def split_data(data, sample_date,percent_train=0.9 ):
    num_rows = len(data)
    train_data, test_data = [], []
    for idx, row in enumerate(data):
        if idx < num_rows * percent_train:
            train_data.append(row)
        else:
            test_data.append(row)

    sample=data[-sample_date:]

    return train_data, test_data, sample




if __name__=='__main__':
    timeseries = load_series('./data_test.txt')


    # print(timeseries[-30:])

    plt.figure()
    plt.plot(timeseries)

    plt.show()

