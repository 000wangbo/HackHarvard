import pandas  as pd
import numpy as np
import datetime
def doit(t):
    t=t/1000
    d = datetime.datetime.fromtimestamp(t)
    str1 = d.strftime("%Y-%m-%d-%H")
    return int(str1.split('-')[-1])


def make_day_gap(x):
    rs = []
    for i in range(1, len(x)):
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


ori_train = pd.read_csv("train",header=None,sep="\t",names=["id","sex","age","edu","dis","label","install","video_id",
                    "video_class","video_tag","author","upload","video_len","display","click",
                    "recommend_class","video_play_len","time","discuss","favorite","transmit",
                   ])
ori_test = pd.read_csv("test",header=None,sep="\t",names=["id","sex","age","edu","dis","install","video_id",
                    "video_class","video_tag","author","upload","video_len","display","click",
                    "recommend_class","video_play_len","time","discuss","favorite","transmit",
                   ])

train = pd.concat([ori_train,ori_test])


print(train['id'].nunique())
train['hour'] = train['time'].apply(lambda x:doit(x))
train['upload'] = train['upload'].apply(lambda x:int(x))
train['bj'] = train['discuss'].apply(lambda x:0 if x=='-' else 1)
train['time'] = train['time'].apply(lambda x:int(x)/1000 - 1539532800)
train['video_play_len'] = train['video_play_len'].apply(lambda x:0 if x=='-' else float(x))


allfeat = train.groupby('id')['sex'].max().reset_index()
allfeat.columns = ['id','sex']

add = train.groupby('id')['age'].max().reset_index()
add.columns = ['id','age']
allfeat=pd.merge(allfeat,add,on='id',how='left')

add = train.groupby('id')['edu'].max().reset_index()
add.columns = ['id','edu']
allfeat=pd.merge(allfeat,add,on='id',how='left')

add = train.groupby('id')['label'].max().reset_index()
add.columns = ['id','label']
allfeat=pd.merge(allfeat,add,on='id',how='left')

add = train.groupby('id')['install'].max().reset_index()
add.columns = ['id','install']
allfeat=pd.merge(allfeat,add,on='id',how='left')


###最先登录时间的时间差.
add = train.groupby('id')['time'].agg('max').reset_index()
add.columns = ['id', 'last_launch_time']
allfeat = allfeat.merge(add, on='id', how='left')

add = train.groupby('id')['time'].agg('min').reset_index()
add.columns = ['id', 'first_launch_time']
allfeat = allfeat.merge(add, on='id', how='left')
allfeat['cha'] = allfeat['last_launch_time'] - allfeat['first_launch_time']

###最先登录时间的时间差.
add = train.groupby('id')['upload'].agg('max').reset_index()
add.columns = ['id', 'last_upload_time']
allfeat = allfeat.merge(add, on='id', how='left')

add = train.groupby('id')['upload'].agg('min').reset_index()
add.columns = ['id', 'first_upload_time']
allfeat = allfeat.merge(add, on='id', how='left')
allfeat['upload_cha'] = allfeat['last_upload_time'] - allfeat['first_upload_time']


add = train.groupby('id')['hour'].agg('min').reset_index()
add.columns = ['id', 'first_launch_hour']
allfeat = allfeat.merge(add, on='id', how='left')

add = train.groupby('id')['hour'].agg('max').reset_index()
add.columns = ['id', 'last_launch_hour']
allfeat = allfeat.merge(add, on='id', how='left')

#############################

launch_cnts = train.groupby(['id'])['time'].agg('count').reset_index()
launch_cnts.columns = ['id', 'launch_count']
allfeat = pd.merge(allfeat,launch_cnts,on='id',how='left')

add = train.groupby('id')['bj'].agg('sum').reset_index()
add.columns = ['id','launch_ava_count']
allfeat = allfeat.merge(add, on='id', how='left')


##########################
add = train.groupby('id')['video_len'].agg('sum').reset_index()
add.columns = ['id', 'video_len_sum']
allfeat = allfeat.merge(add, on='id', how='left')

add = train.groupby('id')['video_play_len'].agg('sum').reset_index()
add.columns = ['id', 'video_play_len_sum']
allfeat = allfeat.merge(add, on='id', how='left')

allfeat['video_len_avg'] = allfeat.apply(lambda x:x['video_len_sum']/x['launch_count'],axis=1)
allfeat['video_play_len_avg'] = allfeat.apply(lambda x:x['video_play_len_sum']/x['launch_count'],axis=1)
allfeat['video_play_len_ava_avg'] = allfeat.apply(lambda x:0 if x['launch_ava_count']==0 else x['video_play_len_sum']/x['launch_ava_count'],axis=1)
allfeat['video_play_rt'] = allfeat.apply(lambda x:0 if x['video_play_len_ava_avg']==0 else x['video_len_ava_avg']/x['video_play_len_ava_avg'],axis=1)


#############################

launch_dates = train.groupby('id')['time'].agg(lambda x: sorted(list(x))).reset_index()
launch_dates.columns = ['id', 'time']

launch_dates['day_gaps'] = launch_dates['time'].apply(make_day_gap)
launch_dates['day_before_last_act_avg'] = launch_dates['day_gaps'].apply(make_day_gap_mean)
launch_dates['day_before_last_act_min'] = launch_dates['day_gaps'].apply(make_day_gap_min)
launch_dates['day_before_last_act_max'] = launch_dates['day_gaps'].apply(make_day_gap_max)
launch_dates['day_before_last_act_std'] = launch_dates['day_gaps'].apply(make_day_gap_std)
launch_dates['day_before_last_act_median'] = launch_dates['day_gaps'].apply(make_day_gap_median)
launch_dates = launch_dates[['id','day_before_last_act_avg','day_before_last_act_min','day_before_last_act_max'
                            ,'day_before_last_act_std','day_before_last_act_median']]
allfeat = pd.merge(allfeat,launch_dates,on='id',how='left')

###用户关注的用户数量
add = train.groupby('id')['author'].nunique().reset_index()
add.columns = ['id', 'user_act_author_cnt']
allfeat = allfeat.merge(add, on='id', how='left')
####用户观看的视频数量
add = train.groupby('id')['video_id'].nunique().reset_index()
add.columns = ['id', 'user_act_video_cnt']
allfeat = allfeat.merge(add, on='id', how='left')

add = train.groupby('install')['id'].agg('count').reset_index()
add.columns = ['install', 'install_cnt']
add = add.sort_values(by='install_cnt', ascending=False)
add['install_cnt_rank'] = add.index + 1
allfeat = allfeat.merge(add, on='install', how='left')
###############################

add = train.groupby(['id','recommend_class'])['time'].agg('count').reset_index()
add = add.pivot(index='id',columns='recommend_class',values='time').reset_index()
add = add.fillna(0)
allfeat = allfeat.merge(add, on='id', how='left')

add = train.groupby(['id','video_class'])['time'].agg('count').reset_index()
add = add.pivot(index='id',columns='video_class',values='time').reset_index()
add = add.fillna(0)
allfeat = allfeat.merge(add, on='id', how='left')

#################################


train['discuss'] = train['discuss'].apply(lambda x:0 if x=='-' else int(x))
train['favorite'] = train['favorite'].apply(lambda x:0 if x=='-' else int(x))
train['transmit'] = train['transmit'].apply(lambda x:0 if x=='-' else int(x))
train['display'] = train['display'].apply(lambda x:0 if x=='-' else int(x))
train['click'] = train['click'].apply(lambda x:0 if x=='-' else int(x))

add = train.groupby('id')['discuss'].agg('sum').reset_index()
add.columns = ['id', 'discuss_cnt']
allfeat = allfeat.merge(add, on='id', how='left')

add = train.groupby('id')['favorite'].agg('sum').reset_index()
add.columns = ['id', 'favorite_cnt']
allfeat = allfeat.merge(add, on='id', how='left')

add = train.groupby('id')['transmit'].agg('sum').reset_index()
add.columns = ['id', 'transmit_cnt']
allfeat = allfeat.merge(add, on='id', how='left')

add = train.groupby('id')['display'].agg('sum').reset_index()
add.columns = ['id', 'display_cnt']
allfeat = allfeat.merge(add, on='id', how='left')

add = train.groupby('id')['click'].agg('sum').reset_index()
add.columns = ['id', 'click_cnt']
allfeat = allfeat.merge(add, on='id', how='left')

###################
add = allfeat.groupby('install')['discuss_cnt'].agg('sum').reset_index()
add.columns = ['install', 'discuss_cnt_install']
allfeat = allfeat.merge(add, on='install', how='left')

add = allfeat.groupby('install')['transmit_cnt'].agg('sum').reset_index()
add.columns = ['install', 'transmit_cnt_install']
allfeat = allfeat.merge(add, on='install', how='left')


add = allfeat.groupby('install')['favorite_cnt'].agg('sum').reset_index()
add.columns = ['install', 'favorite_cnt_install']
allfeat = allfeat.merge(add, on='install', how='left')

add = allfeat.groupby('install')['display_cnt'].agg('sum').reset_index()
add.columns = ['install', 'display_cnt_install']
allfeat = allfeat.merge(add, on='install', how='left')

add = allfeat.groupby('install')['click_cnt'].agg('sum').reset_index()
add.columns = ['install', 'click_cnt_install']
allfeat = allfeat.merge(add, on='install', how='left')

add = allfeat.groupby('install')['rec_type_1'].agg('sum').reset_index()
add.columns = ['install', 'rec_type1_install']
allfeat = allfeat.merge(add, on='install', how='left')

add = allfeat.groupby('install')['rec_type_2'].agg('sum').reset_index()
add.columns = ['install', 'rec_type2_install']
allfeat = allfeat.merge(add, on='install', how='left')

add = allfeat.groupby('install')['rec_type_3'].agg('sum').reset_index()
add.columns = ['install', 'rec_type3_install']
allfeat = allfeat.merge(add, on='install', how='left')

#############


print(allfeat.columns)

allfeat.to_csv("allfeat.csv",index=False)