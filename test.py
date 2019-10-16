import numpy as np
import pandas as pd

matrix = np.random.randn(6,4)
df2 = pd.DataFrame(matrix, columns=list('ABCD'))

print(df2.loc[1,'A'])

for i in range(0,df2[0].shape):
    if df2.loc[i,'A'] is None:
       print('None')
    else:
        print('is not none')

