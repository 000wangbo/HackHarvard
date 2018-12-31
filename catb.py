from catboost import CatBoostRegressor, CatBoostClassifier
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

print(lgb.__version__)
all = pd.read_csv("allfeat_third.csv")
# weight = pd.read_csv("weight_feat.csv")
# all = pd.merge(all,weight,on='id',how='left')
train = all[all.label.notnull()]
df_test = all[all.label.isnull()]
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
# features=['sex','age','install','edu','last_launch_time','first_launch_time','cha','launch_count',
#           'day_before_last_act_max','day_before_last_act_avg','day_before_last_act_median',
#           'install_cnt','install_cnt_rank','user_act_author_cnt','user_act_video_cnt',
#           'discuss_cnt','favoDFASFrite_cnt','transmit_cnt','rec_type_1','rec_type_2','rec_type_3',
#           ]

features = [x for x in train.columns if x not in ['id', 'label','category_102','category_144','date','video_class','video_class_count','week',
    'category_156', 'category_182','category_206','category_207','category_232','category_61','category_83','category_91','category_99',
  ]]

le = LabelEncoder()
sex = le.fit_transform(train['sex'].values)
train['sex'] = sex
sex = le.fit_transform(df_test['sex'].values)
df_test['sex'] = sex

le = LabelEncoder()
age = le.fit_transform(train['age'].values)
train['age'] = age
age = le.fit_transform(df_test['age'].values)
df_test['age'] = age

le = LabelEncoder()
edu = le.fit_transform(train['edu'].values)
train['edu'] = edu
edu = le.fit_transform(df_test['edu'].values)
df_test['edu'] = edu

le = LabelEncoder()
install = le.fit_transform(train['install'].values)
train['install'] = install
install = le.fit_transform(df_test['install'].values)
df_test['install'] = install

le = LabelEncoder()
install = le.fit_transform(train['video_class'].values)
train['video_class'] = install
install = le.fit_transform(df_test['video_class'].values)
df_test['video_class'] = install


df_test['probability'] = 0


X_train = train[features].values
y_train = train['label'].values
cat_features_index = [0,1,2,4]
print('train size: ', train.shape)
print('test size: ', df_test.shape)
print(train['label'].value_counts())


from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
cv_score = []
for idx, (train_idx, test_idx) in enumerate(kf.split(X_train, y_train)):
    print('-' * 50)
    print('iter {}'.format(idx + 1))
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_te, y_te = X_train[test_idx], y_train[test_idx]

    print(y_te)
    cb_model = CatBoostClassifier(bootstrap_type='Bernoulli',
                                  iterations=30000,
                                  learning_rate=0.02,
                                  depth=6,
                                  eval_metric='AUC',
                                  l2_leaf_reg=0.1,
                                  subsample=0.8,
                                  colsample_bylevel=0.8,
                                  od_type='Iter',
                                  metric_period=50,
                                  od_wait=150)

    cb_model.fit(X_tr, y_tr,eval_set=(X_te, y_te),use_best_model=True)

    preds = cb_model.predict_proba(X_te)

    score = roc_auc_score(y_te, preds[:, 1])
    cv_score.append(score)
    df_test['probability'] += cb_model.predict_proba(df_test[features].values)[:, 1]

print('mean offline score: ', np.mean(cv_score))
df_test['probability'] /= 5
print('done.')

df_test[['id','probability']].to_csv("cat_submit.csv",index=False,header=None)
