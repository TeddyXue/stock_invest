import tushare as ts
import pandas as pd

result=ts.get_k_data('600138','2005-01-01')
result['em']=None
cols = list(result)
print(cols)
cols.insert(1,cols.pop(cols.index('low')))
cols.insert(1,cols.pop(cols.index('high')))
cols.insert(4,cols.pop(cols.index('em')))
cols.insert(5,cols.pop(cols.index('volume')))
result = result.loc[:,cols]



print(result)

result.to_csv('Result.csv',index=False,header=None)
