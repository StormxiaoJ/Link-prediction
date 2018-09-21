import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
y_pred = 0

submission = pd.read_csv("sample.csv")
ids = submission['Id'].values
output = pd.DataFrame({'id': ids, 'Prediction': y_pred})

output.to_csv("submission.csv", index=False)