import warnings
warnings.filterwarnings("ignore")
import pandas as pd
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
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from datetime import timedelta 
import time
from tqdm import tqdm


TEST_SUB = None
TRUE = None
F1 = []
PARAMS = {"test": {"n_estimators": 100, "early_stopping_rounds": None, "eval_date": "2018-09"},
          "valid": {"n_estimators": 200, "early_stopping_rounds": 100, "eval_date": "2018-06"}}

# use_cols = [1,3,4,5,7,9,10,12,191, 192,193,194,197,198,199]
# use_cols_smart = []
# for i in range(len(use_cols)):
#     use_cols_smart.append("smart_{}_normalized".format(use_cols_smart[i]))
#     use_cols_smart.append("smart_{}raw".format(use_cols_smart[i]))
#
# use_cols_base = ["manufacturer","serial_number","model", "dt", "tag"]

def custom_f1_eval(y_true, y_pred):
    test_sub = TEST_SUB.copy()
    y_ranked = rank_result(y_pred, test_sub)
    f1 = evaluate_classification_new(y_ranked)
    F1.append(f1)
    f1_mean = sum(F1[-10:]) / len(F1[-10:])
    return "f1", f1_mean, True


def up_sampling(X_train, y_train, ratio=2):
    pos_num = (y_train == 1).sum()
    if pos_num == 0:
        return X_train, y_train
    pos_sap_num = int(pos_num * ratio) 
    X_train.fillna(0, inplace=True)
    smo = BorderlineSMOTE(sampling_strategy={1: pos_sap_num}, random_state=2019, n_jobs=8)
    X_train, y_train = smo.fit_resample(X_train, y_train)

    return X_train, y_train


def down_sampling(data, neg_sap_num=None, ratio=20):
    neg_num = (data["tag"] == 0).sum()
    if neg_num == 0:
        return data
    if neg_sap_num is None:
        neg_sap_num = int(neg_num / ratio)
    rus = RandomUnderSampler(sampling_strategy={0: neg_sap_num}, random_state=2019)
    data, _ = rus.fit_resample(data, data["tag"])
    return data.reset_index(drop=True)


def get_history_fea(data):

    drop_col = ["smart_10_normalized", "smart_191raw"]
    data = data.drop(columns=drop_col).sort_values(by="dt").reset_index(drop=True)

    # data["weekday"] = data["dt"].apply(lambda x: x.weekday())
    # data["dayofmonth"] = data["dt"].apply(lambda x: x.day // 10)
    # data.loc[data["dayofmonth"] >= 2, "dayofmonth"] = 2

    data = data.groupby(["manufacturer", "model", "serial_number"], as_index=False).fillna(method="ffill")

    ###
    index_fea_col = [1,4,5,7,9, 12, 184, 187, 188, 189, 190,191,192, 193, 194, 195, 197, 198, 199]
    #index_fea_col = [191, 7, ]
    division_fea_col = ["division{}".format(i) for i in index_fea_col]
    division_fea = data.groupby(["manufacturer", "model", "serial_number"])[division_fea_col].apply(
        lambda df: df.rolling(20, min_periods=5).mean())
    division_fea.columns = [col + "_mean" for col in division_fea.columns]
    data = pd.concat([data, division_fea], axis=1)

    division_fea = data.groupby(["manufacturer", "model", "serial_number"])[division_fea_col].apply(
        lambda df: df.rolling(20, min_periods=5).max())
    division_fea.columns = [col + "_max" for col in division_fea.columns]
    data = pd.concat([data, division_fea], axis=1)

    division_fea = data.groupby(["manufacturer", "model", "serial_number"])[division_fea_col].apply(
        lambda df: df.rolling(20, min_periods=5).min())
    division_fea.columns = [col + "_min" for col in division_fea.columns]
    data = pd.concat([data, division_fea], axis=1)

    for c in division_fea_col:
        data["{}_max_min".format(c)] = data["{}_max".format(c)] - data["{}_min".format(c)]
        data["{}_max_mean".format(c)] = data["{}_max".format(c)] - data["{}_mean".format(c)]
        data["{}_mean_min".format(c)] = data["{}_mean".format(c)] - data["{}_min".format(c)]
    ###

    moving_fea_col = ["smart_7raw", "smart_5raw", "smart_193raw", "smart_188raw", "smart_190raw", "smart_197raw", "smart_189raw", "smart_192raw", "smart_187raw"]
    moving_fea = data.groupby(["manufacturer", "model", "serial_number"])[moving_fea_col].apply(lambda df: df - df.shift(1))
    moving_fea.columns = [col+"_diff_1day" for col in moving_fea.columns]
    data = pd.concat([data, moving_fea], axis=1)
    return data

def get_division_fea(data):
    index_fea_col = [1,4,5,7,9, 12, 184, 187, 188, 189, 190,191,192, 193, 194, 195, 197, 198, 199]
    division_fea_col = []
    for c in tqdm(index_fea_col):
        data["division" + str(c)] = data["smart_{}raw".format(c)] / (data["smart_{}_normalized".format(c) ]+ 0.1)
        division_fea_col.append("division".format(c))
    return data, division_fea_col


def repair_label(data, test_month):
    tmp_fault = TRUE.copy()
    tmp_fault["fault_time"] += timedelta(days=1)
    tmp_fault = tmp_fault.rename(columns={"fault_time": "dt", "month": "fault_month"})

    data = data.merge(tmp_fault, on=["manufacturer", "model", "serial_number", "dt"], how="left")
    condi = (data["fault_month"].notnull()) & (data.month == test_month)
    data = data[~condi].reset_index(drop=True)
    
    return data.drop(columns="fault_month")

def copy_pos_sample(data, test_month, ratio=2):
    neg_sample = data[(data.tag == 1) & (data.month < test_month)].copy()
    data_list = [data] + [neg_sample] * ratio
    data = pd.concat(data_list, axis=0, ignore_index=True)
    return data

def load_dataset(test_month, model_type = 1):

    t1 = time.time()

    all_data = pd.DataFrame()
    # for pth in ["../data2/data_set_model1.feather", "../data2/data_set_model2.feather"]:
    path_list = ["../data2/data_set_model1.z", "../data2/data_set_model2.z"]
    for i in range(len(path_list)):
        if i + 1 == model_type:
            pth =  path_list[i]
            print("loading",pth)
            data = joblib.load(pth)
            data["dt"] = data["dt"].apply(lambda x: datetime.strftime(x, "%Y-%m-%d"))
            data["month"] = data["dt"].apply(lambda s: s[:7])
            data["dt"] = pd.to_datetime(data['dt'])
            print(len(data))

            #删去2017年的正样本
            condi = (data.tag == 1) & (data.dt < datetime.strptime("2018-01-01", "%Y-%m-%d"))
            data = data[~condi]

            # condi = (data.tag == 0) & (data.dt < datetime.strptime("2018-02-01", "%Y-%m-%d"))
            # data = data[~condi]
            print("after drop 2017 positive: {}".format(len(data)))

            data, division_fea_col = get_division_fea(data)
            data = get_history_fea(data)
            all_data = all_data.append(data, ignore_index=True)

    test_data = all_data[all_data.month == test_month].reset_index(drop=True)
    test_posi = test_data[test_data["tag"] == 1]
    test_nega = test_data[test_data["tag"] == 0]
    sum_sample = 326027 // 2
    test_nega = shuffle(test_nega,random_state = 2019).reset_index(drop=True)
    test_nega = test_nega.loc[0:(sum_sample - len(test_posi)), :]
    test_data = pd.concat([test_posi, test_nega]).reset_index(drop = True)

    del test_nega, test_posi;gc.collect()

    train_data = all_data[all_data.month < test_month].reset_index(drop=True)

    print("load data used time:{}m".format((time.time() - t1)// 60))

    return train_data, test_data

def get_dataset(test_month="2018-06", mode = "valid", model_type = 1):
    data, test_data = load_dataset(test_month, model_type)
    print("data:{}, test_data:{}".format(len(data), len(test_data)))
    test_datetime = datetime.strptime(test_month, "%Y-%m")
    no_fea_col = ['manufacturer', 'model', 'serial_number', 'dt', 'tag', 'month']

    ########################################### train ##################################################

    # # test：删去7月份所有负样本
    # # valid：删去5月份所有负样本
    # print("delete negative one month before test month ")
    delete_month = datetime(test_datetime.year, test_datetime.month - 1, 1).strftime("%Y-%m")
    # # data = data[data["month"] != delete_month]
    # condi1 = (data["month"] == delete_month) & (data["tag"] == 0)
    # data = data[~condi1]

    # 只保留4月一半正样本
    # specil_month = datetime(test_datetime.year, test_datetime.month - 2, 1).strftime("%Y-%m")
    # for model in tqdm([1, 2]):
    #     test_bad_id = TRUE[(TRUE.month == delete_month) & (TRUE.model == model)]["serial_number"].unique()
    #     condi1 = (data.month == specil_month) & (data.tag == 0)
    #     condi2 = (data.month == specil_month) & (data.serial_number.isin(test_bad_id)) & (data.model == model)
    #     data = data[~(condi1 | condi2)]

    # 坏盘删掉负样本
    print("delete bad negative")

    for model in [1, 2]:
        if model == model_type:
            train_bad_id = TRUE[(TRUE.month < delete_month) & (TRUE.model == model)]["serial_number"].unique()
            condi = (data["tag"] == 0) & (data["serial_number"].isin(train_bad_id)) & (data.model == model)
            data = data[~condi]

    # 只保留3月负样本
    # keep_month = datetime(test_datetime.year, test_datetime.month - 3, 1).strftime("%Y-%m")
    # condi = (data['month'] != keep_month) & (data["tag"] == 0)
    # data = data[~condi]
    # print(data.month.value_counts().sort_index())

    # 分层采样负样本
    print("stratified sample")
    df_posi = data[data.tag == 1]
    data = data[data.tag == 0]
    num_nega = len(df_posi) * 200
    df_data = pd.DataFrame()
    if mode == "valid":
        time_list = list(range(1, 6))

    else:
        time_list = list(range(2, 7))
    for i in tqdm(time_list):
        month_i = (test_datetime - timedelta(days= 30 * i)).strftime("%Y-%m")
        print(month_i)
        df_i = data[data["month"] == month_i]
        df_i.reset_index(drop = True, inplace=True)
        df_i = shuffle(df_i, random_state=2019).reset_index(drop=True)
        df_data = df_data.append(df_i.loc[0:(num_nega // len(time_list)), :])
        print(len(df_data))
    del data;gc.collect()
    df_data = df_data.append(df_posi)

    # 构建训练集
    #data = down_sampling(data, (data["tag"] == 1).sum() * 30)
    df_data = shuffle(df_data, random_state=2019).reset_index(drop=True)
    train_y = df_data["tag"]
    df_data = df_data.drop(columns=no_fea_col)

    ########################################### test ##################################################

    # 构建测试集
    global TEST_SUB
    test_data = repair_label(test_data, test_month)
    TEST_SUB = test_data[no_fea_col].copy()
    test_y = test_data["tag"]
    test_data = test_data.drop(columns=no_fea_col)

    gc.collect()
    return df_data, train_y, test_data, test_y


def evaluate_classification_new(pred, verbose=False):

    # precision
    npp = len(pred) #评估窗口内被预测出未来30天会坏的盘数
    ntpp = (pred["tag"] == 1).sum() #评估窗口内第一次预测故障的日期后30天内确实发生故障的盘数
    precision = ntpp / npp if npp != 0 else 0

    # recall
    pred_month = pred["month"].unique()
    true_tmp = TRUE.copy().reset_index(drop=True)
    true_tmp = true_tmp[true_tmp.month.isin(pred_month)]

    true_tmp = true_tmp.merge(pred, on=["manufacturer", "model", "serial_number", "month"], how="left")
    true_tmp["diff_day"] = (true_tmp['fault_time'] - true_tmp['dt']).dt.days
    true_tmp["result"] = 0
    true_tmp.loc[(true_tmp['diff_day']>=0)&(true_tmp['diff_day']<=30), 'result'] = 1

    npr = len(true_tmp) #评估窗口内所有的盘故障数
    ntpr = (true_tmp["result"] == 1).sum()  #评估窗口内第一次预测故障的日期后30天内确实发生故障的盘数
    recall = ntpr / npr if npr != 0 else 0

    # f1
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 100 * 2 * precision * recall / (precision + recall)

    if verbose:
        print(true_tmp[true_tmp["diff_day"] < 0])
        true_tmp.to_csv("../offline/pred_tmp.csv", index=False)
        print("ntpp:", ntpp, "npp:", npp) 
        print("ntpr:", ntpr, "npr:", npr)
        print("precision:", precision)
        print("recall:", recall)
        print("f1:", f1)
    return f1


def rank_result(y_pred, test_sub, verbose=False):
    test_sub["p_test"] = y_pred
    test_sub = test_sub.sort_values(by="p_test", ascending=False).reset_index(drop=True)
    # 仿线上真实故障预测逻辑,后续换成概率值
    test_sub = test_sub.iloc[:250, :]
    test_sub = test_sub.sort_values(by="dt").reset_index(drop=True)
    test_sub = test_sub.drop_duplicates(['manufacturer', 'serial_number', 'model']).reset_index(drop=True)
    #test_sub = test_sub.iloc[:160, :]
    if verbose:
        print(test_sub["p_test"].head(10))
        print(test_sub["p_test"].tail(10))
    return test_sub


def main(mode, params, model_type):

    n_estimators = params["n_estimators"]
    early_stopping_rounds = params["early_stopping_rounds"]
    eval_date = params["eval_date"]
    print("******************** eval_month: %s ********************" % eval_date)

    t2 = time.time()
    X_train, y_train, X_test, y_test = get_dataset(eval_date, mode, model_type)
    print("used time:{}m".format((time.time() - t2) // 60))

    gc.collect()

    print(X_train.shape, X_test.shape)
    print(y_train.value_counts())
    print(y_test.value_counts())

    print('************** training **************')
    # class weight
    clf = LGBMClassifier(
        learning_rate=0.01,
        n_estimators=n_estimators,
        num_leaves=127, # 20, max_depth:5
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2019,
        #scale_pos_weight=50,
        metric=None
    )

    F1.clear()

    if mode == "test":
        clf.fit(X_train, y_train, eval_set=[(X_train, y_train)],
                early_stopping_rounds=early_stopping_rounds, verbose=10)
        now_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        joblib.dump(clf, 'pakdd_model{}.pkl'.format(model_type))
        return

    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
            eval_metric=lambda y_true, y_pred: [custom_f1_eval(y_true, y_pred)], 
            early_stopping_rounds=early_stopping_rounds, verbose=10)
    joblib.dump(clf, 'pakdd_model_valid.pkl')

    y_pred = clf.predict_proba(X_test)[:, 1]
    test_sub = TEST_SUB.copy()
    y_ranked = rank_result(y_pred, test_sub, verbose=True)
    evaluate_classification_new(y_ranked, verbose=True)

    file_path = "../offline/"
    now_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    submit = y_ranked[['manufacturer', 'model', 'serial_number', 'dt']]
    print("submit shape", submit.shape)
    submit.to_csv(file_path+"submit_%s.csv"%now_time, index=False, header=None)

    feature_importaces = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    feature_importaces.to_frame().to_csv(file_path+'lgb_feat_imp.csv', header=None)


if __name__ == "__main__":
    start = time.time()
    print(start)
    mode = sys.argv[1]
    model_type = int(sys.argv[2])
    assert(mode in ["valid", "test"])
    #assert(model in [1, 2])

    TRUE = pd.read_csv("../raw_data/disk_sample_fault_tag.csv", parse_dates=["fault_time"])
    TRUE = TRUE.drop(columns="tag").drop_duplicates()
    TRUE = TRUE.reset_index(drop=True)
    TRUE["month"] = TRUE["fault_time"].apply(lambda dt: dt.strftime("%Y-%m"))

    main(mode, PARAMS[mode], model_type)

    print("used time: {}m".format((time.time() - start) / 60))
