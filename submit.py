import pandas as pd
import zipfile
import numpy as np
import sys
import os
import joblib
import gc
from lightgbm.sklearn import LGBMClassifier
from sklearn.utils import shuffle
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from datetime import timedelta 
from train import get_history_fea, rank_result, get_division_fea

USECOL = "serial_number,manufacturer,model,smart_1_normalized,smart_1raw,smart_3_normalized,smart_4_normalized,smart_4raw,smart_5_normalized,smart_5raw,smart_7_normalized,smart_7raw,smart_9_normalized,smart_9raw,smart_10_normalized,smart_12_normalized,smart_12raw,smart_184_normalized,smart_184raw,smart_187_normalized,smart_187raw,smart_188_normalized,smart_188raw,smart_189_normalized,smart_189raw,smart_190_normalized,smart_190raw,smart_191_normalized,smart_191raw,smart_192_normalized,smart_192raw,smart_193_normalized,smart_193raw,smart_194_normalized,smart_194raw,smart_195_normalized,smart_195raw,smart_197_normalized,smart_197raw,smart_198_normalized,smart_198raw,smart_199_normalized,smart_199raw,smart_240raw,smart_241raw,smart_242raw,dt"
USECOL = USECOL.split(",")

start_date = datetime(2018, 8, 11)
end_date = datetime(2018, 9, 30)
test_date = []
while start_date <= end_date:
    test_date.append(start_date.strftime("%Y%m%d"))
    start_date += timedelta(days=1)

test_data_path = ["/tcdata/disk_sample_smart_log_round2/disk_sample_smart_log_%s_round2.csv"%dt for dt in test_date]


data = pd.DataFrame()
for pth in test_data_path:
    if not os.path.exists(pth):
        print(pth, "skip")
        continue
    cur_data = pd.read_csv(pth, usecols=USECOL)[USECOL]
    print(pth, len(cur_data))
    data = data.append(cur_data, ignore_index=True)
data, division_fea_col = get_division_fea(data)
print("all sample: ", len(data))

# key_cols = ["serial_number", "model"]
# old_id = joblib.load("pakdd_model_valid.pkl")
# new_id = data[key_cols].drop_duplicates(subset=key_cols, keep="first")
# merge_id = pd.merge(old_id, new_id, on = key_cols, how="inner")
#
# print("old_disk:{}, new_disk:{}, intersection_disk:{}".format(len(old_id), len(new_id), len(merge_id)))
# print("new_disk:")
# print(new_id["model"].value_counts())
# print("intersection_disk:")
# print(merge_id["model"].value_counts())

data["dt"] = data["dt"].astype(int)
data["dt"] = data["dt"].astype(str)
data["dt"] = data["dt"].apply(lambda x: datetime.strptime(x, "%Y%m%d"))
data = get_history_fea(data)
data = data[data.dt >= datetime.strptime(str(20180901), "%Y%m%d")].reset_index(drop=True)



no_fea_col = ['manufacturer', 'model', 'serial_number', 'dt']
submit_all = pd.DataFrame()
for model in [1, 2]:
    data1 = data[data["model"] == model]
    data1.reset_index(drop = True, inplace = True)
    test_sub = data1[no_fea_col].copy()
    X_test = data1.drop(columns=no_fea_col)

    clf = joblib.load('pakdd_model{}.pkl'.format(model))
    y_pred = clf.predict_proba(X_test)[:, 1]
    y_ranked = rank_result(y_pred, test_sub, verbose=True)

    submit_all = submit_all.append(y_ranked)

submit = submit_all[['manufacturer', 'model', 'serial_number', 'dt']]
print("submit shape", submit.shape)
submit.to_csv("/result.csv", index=False, header=None)
cmd = "cd /"
os.system(cmd)

with zipfile.ZipFile('result.zip', 'w') as z:
    z.write('result.csv')
