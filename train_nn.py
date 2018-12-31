from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Merge, merge, Reshape, Dropout, Input, Flatten, Concatenate
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np


def build_embedding_network(len_embed_cols):
    model_out = []
    model_in = []

    for dim in len_embed_cols:
        input_dim = Input(shape=(1,), dtype='int32')
        embed_dim = Embedding(dim, dim // 2, input_length=1)(input_dim)
        embed_dim = Dropout(0.25)(embed_dim)
        embed_dim = Reshape((dim // 2,))(embed_dim)
        model_out.append(embed_dim)
        model_in.append(input_dim)

    input_num = Input(shape=(274,), dtype='float32')
    # outputs = Concatenate(axis=1)([*model_out, input_num])
    outputs = input_num

    outputs = (Dense(1024))(outputs)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.35))(outputs)
    outputs = (Dense(512))(outputs)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.35))(outputs)
    outputs = (Dense(256))(outputs)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.35))(outputs)
    outputs = (Dense(128))(outputs)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.35))(outputs)
    outputs = (Dense(64))(outputs)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.15))(outputs)
    outputs = (Dense(32))(outputs)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.15))(outputs)
    outputs = (Dense(16))(outputs)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.15))(outputs)
    outputs = (Dense(1))(outputs)
    outputs = (Activation('sigmoid'))(outputs)

    model = Model([*model_in, input_num], outputs)

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def preproc(X_train, X_val, X_test):
    input_list_train = []
    input_list_val = []
    input_list_test = []

    # the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)

    # the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)

    return input_list_train, input_list_val, input_list_test

all = pd.read_csv("allfeat_third.csv")
train = all[all.label.notnull()]
df_test = all[all.label.isnull()]
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold

features = [x for x in train.columns if x not in ['id', 'label','category_102','category_144',
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

df_test['probability'] = 0
X_train, y_train = train[features], train['label']
X_test = df_test[features]


embed_cols = []
len_embed_cols = []
for c in embed_cols:
    len_embed_cols.append(len(X_train[c].unique()))
    print(c + ': %d values' % len(X_train[c].unique()))

num_cols = [x for x in X_train.columns if x not in embed_cols]
print(embed_cols)
print(num_cols)

# Impute missing values in order to scale
X_train[num_cols] = X_train[num_cols].fillna(value=0)
X_test[num_cols] = X_test[num_cols].fillna(value=0)

# Fit the scaler only on train data
trainsc = pd.concat([X_train[num_cols], X_test[num_cols]])
scaler = MinMaxScaler().fit(trainsc)
X_train.loc[:, num_cols] = scaler.transform(X_train[num_cols])
X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

K = 5
runs_per_fold = 1
n_epochs = 800
patience = 10

cv_aucs = []
full_val_preds = np.zeros(np.shape(X_train)[0])
y_preds = np.zeros((np.shape(X_test)[0], K))

kfold = StratifiedKFold(n_splits=K,
                        shuffle=True, random_state=1)

print(X_train.shape)

for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

    X_train_f, X_val_f = X_train.iloc[f_ind].copy(), X_train.iloc[outf_ind].copy()
    y_train_f, y_val_f = y_train.iloc[f_ind], y_train.iloc[outf_ind]
    X_test_f = X_test.copy()
    print(y_train_f)
    # Shuffle data
    idx = np.arange(len(X_train_f))
    np.random.shuffle(idx)
    X_train_f = X_train_f.iloc[idx]
    y_train_f = y_train_f.iloc[idx]

    # preprocessing
    print(type(X_train_f))
    print(type(X_val_f))
    print(type(X_test_f))
    proc_X_train_f, proc_X_val_f, proc_X_test_f = preproc(X_train_f, X_val_f, X_test_f)

    # track oof prediction for cv scores
    val_preds = 0

    for j in range(runs_per_fold):
        NN = build_embedding_network(len_embed_cols)

        # Set callback functions to early stop training and save the best model so far
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]

        flag = 1
        maxauc = 0
        bs = 0  # 步数
        for k in range(500):
            if (flag == 1):
                NN.fit(proc_X_train_f, y_train_f.values, epochs=1, batch_size=4096, verbose=1, callbacks=callbacks,
                       validation_data=(proc_X_val_f, y_val_f))

                # 提取2个epoch的auc和之前的比较
                val_preds_bj = NN.predict(proc_X_val_f)[:, 0]
                bj_auc = roc_auc_score(y_val_f.values, val_preds_bj)
                print(maxauc)
                print('\nFold %i prediction cv AUC: %.5f\n' % (i, bj_auc))
                if maxauc < bj_auc:
                    bs = 0
                    maxauc = bj_auc
                    y_preds_i = NN.predict(proc_X_test_f)[:, 0]
                    val_preds = val_preds_bj
                else:
                    bs = bs + 1
                if bs >= 20:
                    flag = 0

        y_preds[:, i] += y_preds_i / runs_per_fold
        cv_aucs.append(maxauc)
        full_val_preds[outf_ind] += val_preds

    # cv_auc = roc_auc_score(y_val_f.values, val_preds)
    # cv_aucs.append(cv_auc)
    print('\nFold %i prediction cv AUC: %.5f\n' % (i, maxauc))

print('Mean out of fold AUC: %.5f' % np.mean(cv_aucs))
print('Full validation AUC: %.5f' % roc_auc_score(y_train.values, full_val_preds))

df_test['probability']=y_preds.mean(axis=1)
df_test[['id','probability']].to_csv('submission_keras.txt',header=None,index=None)