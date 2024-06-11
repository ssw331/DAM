import os
import pickle

import pandas as pd

# pkl_1 = []
pkl_2_1 = []
# pkl_2_2 = []
# pkl_3_2 = []

pkl_str_3_1 = ''


def transfer_2_1(file, folder):
    global pkl_str_3_1
    pkl_3_1 = []
    path = "../dataset/diabetes_datasets/" + folder + "/" + file
    # print(path)

    # 加载Excel文件
    # df = pd.read_csv(path)
    df = pd.read_pickle(path)
    # print(df[df.columns[-2]])
    # df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df.drop(columns=df.columns[1:3], axis=1, inplace=True)

    # df['Date'] = str(df['Date']) + '-' + str(df['GL'])
    for (index, row) in df.iterrows():
        df.loc[index, 'Date'] = str(df.loc[index, 'Date']) + '-' + str(df.loc[index, 'GL'])
    #
    # 如果需要，将时间列设置为索引
    df.set_index('Date', inplace=True)

    se = pd.Series(df[df.columns[0]].values, index=df.index)

    print(se)

    # data_str = df.apply(lambda x: '    '.join(x.astype(str)), axis=1).str.cat(sep='\n')
    # data_str = df.to_csv(sep='    ', index=False, header=False)

    # values = df.values
    # values = values.tolist()

    pkl_3_1.append(se)
    pkl_2_1.append(pkl_3_1)
    # pkl_str_3_1 = data_str
    # pkl_3_1.append(pkl_str_3_1)
    # print(pkl_3_1)


# def transfer_2_2(file):
#     path = "../dataset/diabetes_datasets/" + file
#
#     # 加载Excel文件
#     df = pd.read_excel(path)
#
#     values = df.values
#     values = values.T
#     values = values.tolist()
#
#     pkl_2_2.extend(values)


files1 = os.listdir(r'E:\PycharmP\DAM\dataset\diabetes_datasets\Shanghai_T1DMProcessed')
files2 = os.listdir(r'E:\PycharmP\DAM\dataset\diabetes_datasets\Shanghai_T2DMProcessed')
summary1 = 'Shanghai_T1DM_Summary.xlsx'
summary2 = 'Shanghai_T2DM_Summary.xlsx'

files = os.listdir(r'E:\PycharmP\DAM\dataset\diabetes_datasets\Proccessed_PKL\updated_Proccessed_PKL\Proccessed_PKL')

for file in files:
    transfer_2_1(file, 'Proccessed_PKL/updated_Proccessed_PKL/Proccessed_PKL')

# for filename in files1:
#     transfer_2_1(filename, 'Shanghai_T1DMProcessed')

# for filename in files2:
#     transfer_2_1(filename, 'Shanghai_T2DMProcessed')

# transfer_2_2(summary1)
# transfer_2_2(summary2)

# pkl_2_1.append(pkl_3_1)

# pkl_1.append(pkl_2_1)
# pkl_1.append(pkl_2_2)

# print(pkl_2_1)

# 保存为Pickle文件
pickle.dump(pkl_2_1, open('data.pkl', 'wb'))
