#nanカラムを予想するクラスを定義する
#メソッドはnanが含まれている行と、nanが含まれない行を分けるload_data_nan
#nanカラムを学習するfit、nanカラムを予測するpredictから構成される
#このクラスはtrain.pyから呼び出されることを想定している

import pandas as pd
import numpy as np
from load_data import load_train_data
from logging import getLogger, StreamHandler, DEBUG, Formatter, FileHandler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression

logger = getLogger(__name__)
DIR = 'log/'

log_fmt = Formatter('%(asctime)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')

handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)

handler = FileHandler(DIR+__name__+'.log','a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)

nan_train_x=[]
non_nan_train_x=[]

def load_data_nan(train_x,  nan_column):
    #DataFrameで渡されることを想定
    logger.info('enter')
    nan_train_x = train_x[train_x[nan_column].isnull()]
    non_nan_train_x = train_x[train_x[nan_column].isnull()==False]
    logger.info('end')
    return nan_train_x,non_nan_train_x

def nan_data_predict(nan_train_x,non_nan_train_x,nan_column):
    logger.info('enter')
    
    train_y = non_nan_train_x[nan_column]
    train_x = non_nan_train_x.drop(nan_column,axis=1)

    logger.info('create training data from non_nan_train_x:{},{}'.format(train_x.shape,train_y.shape))
    
    test_x = nan_train_x.drop(nan_column,axis=1)
    logger.info('create test data from nan_train_x:{}'.format(test_x.shape))

    lr = LinearRegression().fit(train_x,train_y)
    logger.info('lr fitted')

    test_y = lr.predict(test_x)
    logger.info('lr predicted:{}'.format(test_y.shape))
    test_x['Age'] = test_y
    logger.info('test_x.shape:{}  test_y.shape:{}'.format(test_x.shape,test_y.shape))
    df_x = pd.concat([test_x,non_nan_train_x])
    logger.info('df_temp.shape:{}  non_nan_train_x.shape:{}'.format(test_x.shape,non_nan_train_x.shape))

    return df_x

if __name__ == '__main__':
    logger.info('enter')
    train_x,train_y = load_train_data()
    nan_train_x,non_nan_train_x = load_data_nan(train_x,'Age')
    logger.info('load_data_nan loaded')
    logger.info('nan_train_x.shape:{}'.format(nan_train_x.shape))
    logger.info('non_nan_train_x.shape:{}'.format(non_nan_train_x.shape))
    df_x = nan_data_predict(nan_train_x,non_nan_train_x,'Age')
    logger.info('result:{}'.format(df_x.shape))
    logger.info('end')
