import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

train = pd.read_csv("train_set.csv")
test = pd.read_csv("test_set.csv")

y_train = train.exist.values


train.drop(['exist','source','target'],axis=1)
test.drop(['source','target'],axis=1)
X_train = train.values
X_test = test.values
X_train,X_dev,y_train,y_dev = train_test_split(X_train,y_train,test_size=0.3, random_state=200)
lgb_train = lgb.Dataset(X_train,y_train)
lgb_eval = lgb.Dataset(X_dev,y_dev,reference=lgb_train)


# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',  # GBDT算法为基础
#     'objective': 'binary',  # 因为要完成预测用户是否买单行为，所以是binary，不买是0，购买是1
#     'metric': 'auc',  # 评判指标
#     'max_bin': 255,  # 大会有更准的效果,更慢的速度
#     'learning_rate': 0.02,  # 学习率
#     'num_leaves': 64,  # 大会更准,但可能过拟合
#     'max_depth': 2,  # 小数据集下限制最大深度可防止过拟合,小于0表示无限制
#     'feature_fraction': 0.8,  # 防止过拟合
#     'bagging_freq': 5,  # 防止过拟合
#     'bagging_fraction': 0.8,  # 防止过拟合
#     'min_data_in_leaf': 21,  # 防止过拟合
#     'min_sum_hessian_in_leaf': 3.0,  # 防止过拟合
#     'header': True  # 数据集是否带表头
# }
params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'nthread': 4,
            'learning_rate': 0.02,  # 02,
            'num_leaves': 20,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'subsample_freq': 1,
            'max_depth': 2,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 60, # 39.3259775,
            'seed': 0,
            'verbose': -1,
            'metric': 'auc',
}



print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_eval,
                early_stopping_rounds=20)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')

y_pred = gbm.predict(X_test)
print(y_pred)
#X_test[['source', 'target']].to_csv("submission.csv", index= False)
submission = pd.read_csv("sample.csv")
ids = submission['Id'].values
#submission.drop('id', inplace=True, axis=1)


#x = submission.values
#y = model.predict(x)

output = pd.DataFrame({'id': ids, 'Prediction': y_pred})
#output.to_csv("submission.csv", index=False)