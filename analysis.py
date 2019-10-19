#データを解析する
#データの一部を表示し、nan列を個数を含めて表示

import pandas as pd
from load_data import load_data, load_train_data, load_test_data

#dfの表示を省略を防ぐために設定
pd.set_option( 'display.width', 100 )

#テストデータをロード
df_train_x, df_train_y = load_train_data( )
df_test = load_test_data( )

#訓練データのXとYのサイズと行を表示する
#テストデータはXのみ表示
print( 'train_column_x : {} \n {}'.format( df_train_x.shape, df_train_x.columns ) )
print( 'train_y_columns : {} \n {}'.format( df_train_y.shape, df_train_y.columns ) )
print( 'test columns : {} \n {}'.format( df_test.shape, df_test.columns ) )

#試しに1行だけ表示
print( df_train_x.loc[[ 1 ]] )

#NaNの判定
nan_train_x = df_train_x.isnull().sum()
nan_train_y = df_train_y.isnull().sum()
nan_test = df_test.isnull().sum()

#Trueのみを抽出するリストを作成
nan_sum_train_x = []
nan_sum_train_y = []
nan_sum_test = []

i = 0
print('nan_train_x:',df_train_x.shape)
for sum in nan_train_x :
    if sum > 0 :
        #print('shape:{}'.format(df_train_x.shape))
        print('nan column:{} sum:{} share:{:.2%}'.format(nan_train_x.index[i],nan_train_x[i],nan_train_x[i] / df_train_x.shape[0]))
    i += 1

i = 0
print('\nnan_train_y:',df_train_y.shape)
for sum in nan_train_y :
    if sum > 0 :
        print('nan column:{} sum:{} share:{:.2%}'.format(nan_train_y.index[i],nan_train_y[i],nan_train_y[i] / df_train_y.shape[0]))
    i += 1

i = 0
print('\nnan_test:',df_test.shape)
for sum in nan_test :
    if sum > 0 :
        print('nan column:{} sum:{} share:{:.2%}'.format(nan_test.index[i],nan_test[i],nan_test[i] / df_test.shape[0]))
    i += 1

print(df_train_x.dtypes)
print('---')
print(df_train_y.dtypes)

#Ticketはどんな意味を持つのか調査した
#print( df_train_x.loc[:]['Ticket'] )

#Cabinがどんな意味を持つのか調査　本当に必要？
#print( df_train_x[:]['Cabin'].unique() )
#print( len(df_train_x[:]['Cabin'].unique()) )
