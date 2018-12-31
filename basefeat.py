import pandas as pd
from tqdm import tqdm
train = pd.read_csv("train",header=None,sep="\t",names=["id","sex","age","edu","dis","label","install","video_id",
                    "video_class","video_tag","author","upload","video_len","display","click",
                    "recommend_class","video_play_len","time","discuss","favorite","transmit"])


#
# for i in tqdm(train['id'].unique()):
#     dealtrain = train[train['id'] == i]
#     id.append(i)
#     label.append(dealtrain['label'].iloc[0])
#     if "男" in dealtrain['sex'].unique():
#         sex.append(0)
#     elif "女" in dealtrain['sex'].unique():
#         sex.append(1)
#     else:
#         sex.append(2)
#
#
#     if "18以下" in dealtrain['age'].unique():
#         age.append(0)
#     elif "18-24" in dealtrain['age'].unique():
#         age.append(1)
#     elif "25-34" in dealtrain['age'].unique():
#         age.append(2)
#     elif "35-44" in dealtrain['age'].unique():
#         age.append(3)
#     elif "45-54" in dealtrain['age'].unique():
#         age.append(4)
#     elif "55-64" in dealtrain['age'].unique():
#         age.append(5)
#     elif "65以上" in dealtrain['age'].unique():
#         age.append(6)
#     else:
#         age.append(7)
#
#     if "高中及以下" in dealtrain['edu'].unique():
#         edu.append(0)
#     elif "大专" in dealtrain['edu'].unique():
#         edu.append(1)
#     elif "本科及以上" in dealtrain['edu'].unique():
#         edu.append(2)
#     else:
#         edu.append(3)
#
# basefeat['id'] = id
# basefeat['label'] = label
# basefeat['sex'] = sex
# basefeat['edu'] = edu
# basefeat['age'] = age
# basefeat.to_csv("basefeat.csv",index=False)

basefeat = train.groupby('id')['sex'].max().reset_index()
basefeat.columns = ['id','sex']
add = train.groupby('id')['age'].max().reset_index()
add.columns = ['id','age']

basefeat=pd.merge(basefeat,add,on='id',how='left')

add = train.groupby('id')['edu'].max().reset_index()
add.columns = ['id','edu']
basefeat=pd.merge(basefeat,add,on='id',how='left')

add = train.groupby('id')['label'].max().reset_index()
add.columns = ['id','label']
basefeat=pd.merge(basefeat,add,on='id',how='left')

add = train.groupby('id')['install'].max().reset_index()
add.columns = ['id','install']
basefeat=pd.merge(basefeat,add,on='id',how='left')

basefeat.to_csv("basefeat.csv",index=False)


