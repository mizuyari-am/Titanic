import pandas as pd
import numpy as np
from load_data import load_train_data,load_test_data
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.ensemble import GradientBoostingClassifier

DIR = 'log/'
logger = getLogger( __name__ )

if __name__ == '__main__' :
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler( )
    handler.setLevel( 'INFO' )
    handler.setFormatter( log_fmt )
    logger.addHandler( handler )

    handler = FileHandler( DIR + 'train.py.log', 'a' )
    handler.setLevel( DEBUG )
    handler.setFormatter( log_fmt )
    logger.setLevel( DEBUG )
    logger.addHandler( handler )
    
    logger.info( 'start' )

    train_x, train_y = load_train_data( )
    train_x = train_x.values
    train_y = train_y.values
    logger.info( 'train data loaded:{}, {}'.format(train_x.shape, type(train_x) ) )

    gbrt = GradientBoostingClassifier( random_state=0 )
    gbrt.fit( train_x, train_y )
    logger.info( 'gbrt fitted')

    print( 'Accuracy on training set:{.3f}'.format(gbrt.score(train_x, train_y)))
