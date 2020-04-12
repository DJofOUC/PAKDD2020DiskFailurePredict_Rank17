import pandas as pd
import os
import joblib
import gc
from datetime import datetime, timedelta


# PARAMS
key_col = ["manufacturer", "model", "serial_number", "dt"]
train_suffix_list = ["201707", "201708", "201709", "201710", "201711", "201712", "201801", "201804", "201807",
                     "201802", "201803", "201805", "201806"]

MODEL = 1
fault_data_path = "../raw_data/disk_sample_fault_tag.csv"
test_data_path = "../raw_data/disk_sample_smart_log_test_b.csv"
train_data_path_list = ["../raw_data/disk_sample_smart_log_%s.csv"%suffix for suffix in train_suffix_list]

if not os.path.exists("../data2"):
    os.mkdir("../data2")

if not os.path.exists("../offline"):
    os.mkdir("../offline")

# get useful columns
drop_col = ["smart_3raw", "smart_10raw", "smart_240_normalized", "smart_241_normalized", "smart_242_normalized"]
data = pd.read_csv(test_data_path)
data = data.drop_duplicates(key_col)
drop_col += [col for col in data.columns if data[col].notnull().sum() == 0]
use_col = [col for col in data.columns if col not in drop_col]
print(len(use_col), use_col)


# 2/3/5/6四个月份取所有数据
data_set = pd.DataFrame()
for data_path in train_data_path_list[-7:]:
    print(data_path)
    cur_data = pd.read_csv(data_path, usecols=use_col, parse_dates=["dt"])
    cur_data = cur_data[cur_data.model == MODEL]
    data_set = data_set.append(cur_data, ignore_index=True)
    print(len(data_set))


# 其余月份只取坏盘数据
serial_number_set = pd.read_csv(fault_data_path, usecols=["model", "serial_number"])
serial_number_set = serial_number_set[serial_number_set.model == MODEL]
serial_number_set = serial_number_set.drop_duplicates(subset=["model", "serial_number"]).reset_index(drop=True)
serial_number_set["keep"] = 1
print("serial_number num: ", len(serial_number_set))

for data_path in train_data_path_list[:-7]:
    print(data_path)
    cur_data = pd.read_csv(data_path, usecols=use_col, parse_dates=["dt"])
    cur_data = cur_data[cur_data.model == MODEL]
    cur_data = cur_data.merge(serial_number_set, on=["model", "serial_number"], how="left")
    cur_data = cur_data[cur_data["keep"] == 1]
    cur_data = cur_data.drop(columns="keep")
    data_set = data_set.append(cur_data, ignore_index=True)
    print(len(data_set))


# process tag
fault = pd.read_csv(fault_data_path, parse_dates=["fault_time"])
fault["tag"] = 1
fault = fault.drop_duplicates()
all_fault = pd.DataFrame()
for i in range(-1, 10):
    tmp_fault = fault.copy()
    tmp_fault["fault_time"] -= timedelta(days=i)
    all_fault = all_fault.append(tmp_fault, ignore_index=True)
all_fault.rename(columns={"fault_time": "dt"}, inplace=True)
all_fault = all_fault.drop_duplicates()


# merge tag to dateset
data_set = data_set.drop_duplicates(key_col)
data_set = data_set.merge(all_fault, on=key_col, how="left")
data_set["tag"] = data_set["tag"].fillna(0).astype(int)
data_set = data_set.reset_index(drop=True)

# data_set.to_csv("../data/data_set_model%d.csv"%MODEL, index=False)
#
#
# data_set = pd.read_csv("../data/data_set_model%d.csv"%MODEL)
# # 删除部分无用样本
# condi1 =("2018-05-01" <= data_set.dt) & (data_set.dt <= "2018-05-09") & (~data_set.serial_number.isin(serial_number_set["serial_number"]))
# data_set = data_set[~condi1]
#
# condi2 =("2018-02-01" <= data_set.dt) & (data_set.dt <= "2018-02-09") & (~data_set.serial_number.isin(serial_number_set["serial_number"]))
# data_set = data_set[~condi2]

joblib.dump(data_set, "../data/data_set_model%d.z"%MODEL)
