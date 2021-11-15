from ipfn import ipfn
import numpy as np
import pandas as pd

m      = np.array([1., 2., 1., 3., 5., 5., 6., 2., 2., 1., 7., 2.,
               5., 4., 2., 5., 5., 5., 3., 8., 7., 2., 7., 6.], )
dma_l  = [501, 501, 501, 501, 501, 501, 501, 501, 501, 501, 501, 501,
          502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502]
size_l = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
          1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]

age_l  = ['20-25','30-35','40-45',
          '20-25','30-35','40-45',
          '20-25','30-35','40-45',
          '20-25','30-35','40-45',
          '20-25','30-35','40-45',
          '20-25','30-35','40-45',
          '20-25','30-35','40-45',
          '20-25','30-35','40-45']

df = pd.DataFrame()
df['dma'] = dma_l
df['size'] = size_l
df['age'] = age_l
df['total'] = m

xipp = df.groupby('dma')['total'].sum()     # get a copy of the reduced df
xpjp = df.groupby('size')['total'].sum()
xppk = df.groupby('age')['total'].sum()
xijp = df.groupby(['dma', 'size'])['total'].sum()
xpjk = df.groupby(['size', 'age'])['total'].sum()
# xppk = df.groupby('age')['total'].sum()
print(xipp)
xipp.loc[:] = [52,  49]
#xipp.loc[502] = 48
print(xipp)
xpjp.loc[1] = 20
xpjp.loc[2] = 30
xpjp.loc[3] = 35
xpjp.loc[4] = 15

xppk.loc['20-25'] = 35
xppk.loc['30-35'] = 40
xppk.loc['40-45'] = 25

xijp.loc[501] = [9, 17, 19, 7]
xijp.loc[502] = [11, 13, 16, 8]

xpjk.loc[1] = [7, 9, 4]
xpjk.loc[2] = [8, 12, 10]
xpjk.loc[3] = [15, 12, 8]
xpjk.loc[4] = [5, 7, 3]
print(xpjk)
print(xipp)
print(type(xppk))
aggregates = [xipp, xpjp, xppk, xijp, xpjk]
dimensions = [['dma'], ['size'], ['age'], ['dma', 'size'], ['size', 'age']]
print(df)
IPF = ipfn.ipfn(df, aggregates, dimensions)
df = IPF.iteration()

print(df)
