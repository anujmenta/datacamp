import pandas
import requests
from time import time
d= pd.read_csv('lbv.csv')
def dateclean( row):
    date,time =row['CHARTTIME'].split('')
    return pandas.Series([date,time)
d[['time_hour','time_min','time_sec']] = d.apply(dateclean,axis=1)
final_df = df[[' UREA_VALUE','Potassium_Level','WBC_VALUE','Sodium_Level','B_VALUE','DIE_IN_24HRS','time_hour','time_min','time_sec']]
net = tflearn.input_data(shape=[None,8])
net = tflearn.fully_connected(net,24)
net = tflearn.fully_connected(net,2,activation='softmax')
model = tflearn.DNN(net)
model.fit(data,labels,n_epoch=5,batch_size=100,show_metric=True)
model.save('initial100iter_1.tflearn')
