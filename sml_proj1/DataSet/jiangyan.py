import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
df = pd.read_csv('train_Path.csv', header=0)
label = df['exist']
y = label.values
print([column for column in df])
df.drop(['source', 'target', 'exist'], axis=1, inplace=True)
x = df.values
X_train, X_dev, y_train, y_dev = train_test_split(x, y, test_size=0.3, random_state=78, stratify=y)
test_csv = pd.read_csv('test_Path.csv', header=0)
test_csv.drop(['source', 'target'], axis=1, inplace=True)
X_test = test_csv.values
del label
del df
del x
del y
del test_csv
gc.collect()
lgb_train = lgb.Dataset(X_train,y_train)
lgb_eval = lgb.Dataset(X_dev,y_dev,reference=lgb_train)


params = {
    'task': 'train',
    'boosting_type': 'gbdt',  
    'objective': 'binary',  
    'metric': 'auc',  
    'max_bin': 255,  
    'learning_rate': 0.01,  
    'num_leaves': 8,  
    'max_depth': 3,  
    'feature_fraction': 0.8,  
    'bagging_freq': 5,  
    'bagging_fraction': 0.8,  
    'min_data_in_leaf': 21,  
    'min_sum_hessian_in_leaf': 3.0,  
    'header': True,  
    'colsample_bytree': 0.9497036,
    'reg_alpha': 0.041545473,
    'reg_lambda': 0.0735294,
    'min_child_weight': 60  
}


print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=100)

print('Save model...')

gbm.save_model('model.txt')

print('Start predicting...')

y_pred = gbm.predict(X_test)
print(y_pred)

submission = pd.read_csv("sample.csv")
ids = submission['Id'].values

lgb.plot_importance(gbm)
plt.show()

output = pd.DataFrame({'id': ids, 'Prediction': y_pred})
output.to_csv("submission.csv", index=False)