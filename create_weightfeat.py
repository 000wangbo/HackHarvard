import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import datetime

bj = [1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3]


train = pd.read_csv("all_video_tag.csv")
print("finish")

# add = train.groupby(['id'])['video_tag'].agg('sum').reset_index()
#
# add[['id','video_tag']].to_csv("all_video_tag.csv",index=False)

corpus = train['video_tag']
vectorizer = CountVectorizer(max_features=300,ngram_range=(1,1))
transformer = TfidfTransformer()
tfidf = vectorizer.fit_transform(corpus)
word = vectorizer.get_feature_names()
weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
weight=pd.DataFrame(weight)
weight['id']=train['id']
weight.to_csv("weight_feat.csv",index=False)




