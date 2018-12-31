import pandas as pd
import datetime
def convert_date(t):
    t=t/1000
    d = datetime.datetime.fromtimestamp(t)
    dd = d.strftime("%d")
    return int(dd)

train = pd.read_csv("test",header=None,sep="\t",names=["id","sex","age","edu","dis","install","video_id",
                    "video_class","video_tag","author","upload","video_len","display","click",
                    "recommend_class","video_play_len","time","discuss","favorite","transmit"])

print(train.shape)
train['video_play_len'] = train['video_play_len'].apply(lambda x:0 if x=='-' else float(x))
train = train[train['video_play_len'] !=0]
train['time_new'] = train['time']+train['video_play_len']*1000
train['time_new'] = train['time_new'].apply(convert_date)


add = train.groupby('id')['time_new'].nunique().reset_index()
add.columns = ['id', 'time_new_cnt']
add = add[add['time_new_cnt'] !=1]

submit= pd.read_csv("lgb_dart_submit.csv",header=None,names=['id','prob'])
submit = pd.merge(submit,add,on='id',how='left')
submit = submit.fillna(1)
submit['prob'] = submit[['prob','time_new_cnt']].apply(lambda x:x.prob if x.time_new_cnt==1 else 0.9,axis=1)
submit['prob'] = submit['prob'].apply(lambda x:"%.3f" % x)
submit[['id','prob']].to_csv("deal_lgb_dart_submit.csv",index=False,header=None)
