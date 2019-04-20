import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt

start = datetime.datetime(2005,12,1)
end = datetime.date.today()
stock = web.DataReader("600138.SS", "yahoo", start, end)
print(stock.all)
#High   Low  Open  Close      Volume  Adj Close
file=open('data_test.txt','w+')
for row in stock.iterrows():

    date=row[0]
    high=row[1][0]
    low=row[1][1]
    Open=row[1][2]
    Close=row[1][3]
    Volume=row[1][4]
    AdjClose=row[1][5]

    print(str(date)+','+str(high)+','+str(low)+','+str(Open)+','+str(Close)+','+str(Volume)+','+str(AdjClose))

    file.write(str(date)+','+str(high)+','+str(low)+','+str(Open)+','+str(Close)+','+str(Volume)+','+str(AdjClose)+'\n')
file.close()
# for i in stock['Close']:
#     print(i)
plt.figure()
plt.plot(stock['Close'], 'g')
plt.show()
