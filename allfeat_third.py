import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import datetime
def make_day_gap(x):
    rs = []
    for i in range(1, len(x)):
        if x[i]!=x[i-1]:
            g = x[i] - x[i - 1]
            rs.append(g)
    return rs


def make_day_gap_mean(x):
    if len(x) == 0:
        return np.nan
    return np.mean(x)


def make_day_gap_min(x):
    if len(x) == 0:
        return np.nan
    return np.min(x)


def make_day_gap_sum(x):
    if len(x) == 0:
        return np.nan
    return np.sum(x)


def make_day_gap_median(x):
    if len(x) == 0:
        return np.nan
    return np.median(x)


def make_day_gap_max(x):
    if len(x) == 0:
        return np.nan
    return np.max(x)


def make_day_gap_std(x):
    if len(x) == 0:
        return np.nan
    return np.std(x)

def convert_date(t):
    t=t/1000
    d = datetime.datetime.fromtimestamp(t)
    dd = d.strftime("%d")
    return int(dd)
bj = [1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3]
allfeat =  pd.read_csv('allfeat_second.csv')

train = pd.read_csv("all")

train['date'] = train['time'].apply(convert_date)

add = train.groupby('id')['date'].max().reset_index()
add.columns = ['id','date']
add['week'] = add['date'].apply(lambda x:bj[x-15])
allfeat = allfeat.merge(add, on='id', how='left')

add = train.groupby(['id'])['video_class'].value_counts().rename("video_class_count")
add = add.groupby('id').head(1).reset_index()
add.columns = ['id','video_class','video_class_count']
allfeat = allfeat.merge(add, on='id', how='left')


train['sex'] = train['sex'].apply(lambda x:'sexmis'if x=='-' else x)

add = train.groupby(['id','sex'])['time'].agg('count').reset_index()
add = add.pivot(index='id',columns='sex',values='time').reset_index()
add = add.fillna(0)
allfeat = allfeat.merge(add, on='id', how='left')

allfeat['repeal_author'] = allfeat['launch_count'] - allfeat['user_act_author_cnt']
allfeat['repeal_video'] = allfeat['launch_count'] - allfeat['user_act_video_cnt']
####################

add = train.groupby('id')['install'].nunique().reset_index()
add.columns = ['id', 'user_act_install_cnt']
allfeat = allfeat.merge(add, on='id', how='left')

allfeat['video_len_ava_avg'] = allfeat.apply(lambda x:0 if x['launch_ava_count']==0 else x['video_len_sum']/x['launch_ava_count'],axis=1)
allfeat['video_play_rt'] = allfeat.apply(lambda x:0 if x['video_play_len_ava_avg']==0 else x['video_len_ava_avg']/x['video_play_len_ava_avg'],axis=1)


launch_dates = train.groupby('id')['time'].agg(lambda x: sorted(list(x))).reset_index()
launch_dates.columns = ['id', 'time']

launch_dates['day_gaps'] = launch_dates['time'].apply(make_day_gap)
launch_dates['launch_new_count'] = launch_dates['day_gaps'].apply(lambda x:len(x))
launch_dates['day_before_last_act_avg_new'] = launch_dates['day_gaps'].apply(make_day_gap_mean)
launch_dates['day_before_last_act_min_new'] = launch_dates['day_gaps'].apply(make_day_gap_min)
launch_dates['day_before_last_act_max_new'] = launch_dates['day_gaps'].apply(make_day_gap_max)
launch_dates['day_before_last_act_std_new'] = launch_dates['day_gaps'].apply(make_day_gap_std)
launch_dates['day_before_last_act_median_new'] = launch_dates['day_gaps'].apply(make_day_gap_median)
launch_dates = launch_dates[['id','launch_new_count','day_before_last_act_avg_new','day_before_last_act_min_new','day_before_last_act_max_new'
                            ,'day_before_last_act_std_new','day_before_last_act_median_new']]

allfeat = pd.merge(allfeat,launch_dates,on='id',how='left')

print(allfeat.head(10))


allfeat.to_csv("allfeat_third.csv",index=False)
