import pandas as pd
import numpy as np
import datetime

allfeat = pd.read_csv("allfeat.csv")

add = allfeat.groupby('install')['launch_count'].agg('sum').reset_index()
add.columns = ['install', 'launch_count_install']
allfeat = allfeat.merge(add, on='install', how='left')

add = allfeat.groupby('install')['launch_ava_count'].agg('sum').reset_index()
add.columns = ['install', 'launch_ava_count_install']
allfeat = allfeat.merge(add, on='install', how='left')

add = allfeat.groupby('install')['video_len_sum'].agg('sum').reset_index()
add.columns = ['install', 'video_len_sum_install']
allfeat = allfeat.merge(add, on='install', how='left')

add = allfeat.groupby('install')['video_play_len_sum'].agg('sum').reset_index()
add.columns = ['install', 'video_play_len_sum_install']
allfeat = allfeat.merge(add, on='install', how='left')

add = allfeat.groupby('install')['video_play_len_avg'].agg('sum').reset_index()
add.columns = ['install', 'video_play_len_avg_install']
allfeat = allfeat.merge(add, on='install', how='left')

add = allfeat.groupby('install')['video_len_avg'].agg('sum').reset_index()
add.columns = ['install', 'video_len_avg_install']
allfeat = allfeat.merge(add, on='install', how='left')

add = allfeat.groupby('install')['video_play_len_ava_avg'].agg('sum').reset_index()
add.columns = ['install', 'video_play_len_ava_avg_install']
allfeat = allfeat.merge(add, on='install', how='left')

allfeat.to_csv("allfeat_second.csv",index=False)