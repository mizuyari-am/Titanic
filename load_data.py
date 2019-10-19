import pandas as pd
import numpy as np

from logging import getLogger, StreamHandler, DEBUG, Formatter, FileHandler

TRAIN_DATA = 'input/train.csv'
TEST_DATA = 'input/test.csv'

logger = getLogger( __name__ )
DIR = 'log/'

log_fmt = Formatter( '%(asctime)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
handler = StreamHandler( )
handler.setLevel( 'INFO' )
handler.setFormatter( log_fmt )
logger.addHandler( handler )

handler = FileHandler( DIR + 'load_data.py.log', 'a' )
handler.setLevel( DEBUG )
handler.setFormatter( log_fmt )
logger.setLevel( DEBUG )
logger.addHandler( handler )

def load_data( path ) :
    logger.debug( 'enter' )
    df = pd.read_csv( path )
    logger.debug( 'exit' )
    return df

def load_train_data( ) :
    logger.debug( 'enter' )
    df = load_data( TRAIN_DATA )
    logger.info( 'train data loaded:{}'.format(df.shape) )
    
    #途中にSurviedがあるので分割して結合し、train_xとtrain_yを作成する
    #不要な列の削除も行う　具体的には名前、チケット番号
    df_y = df[ [ 'Survived' ] ]
    logger.info( 'train_y created:{}'.format(df_y.shape) )
    df_x = df.drop( ['Survived', 'Name', 'Ticket'], axis=1 )
    logger.info('train_x created:{}'.format(df_x.shape))
    df_x = pd.get_dummies(df_x, columns=['Sex', 'Cabin', 'Embarked', 'Pclass'] )
    logger.info('train_x one-hot encoded:{}'.format(df_x.shape))

    logger.debug( 'exit' )
    return df_x, df_y

def load_test_data( ) :
    logger.debug( 'enter' )
    df = load_data( TEST_DATA )
    logger.debug( 'exit' )
    return df

if __name__ == '__main__' :
    logger.info( 'start' )
    x, y = load_train_data( )
    print( x.head( ) ) 
    print( y.head( ) )
    print( load_test_data( ).head( ) )
    logger.info( 'end' )
