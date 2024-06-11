import os
import re

import numpy as np
import pandas as pd

dietary = pd.read_csv('../dataset/diabetes_datasets/Dietary_GL_Estimation.csv')
memo = {
    'filename': [],
    'index': [],
}
default = 40

cnt = 0


def correct(file, folder):
    path = "../dataset/diabetes_datasets/" + folder + "/" + file

    print(path)

    diabetes = pd.read_csv(path)

    for (index, row) in diabetes.iterrows():
        if row['GL'] == 75.0:
            diabetes.loc[index, 'GL'] = default
            print(diabetes.loc[index, 'GL'])

    diabetes.to_csv(path)


# print(dietary)


def dietary_process(file, folder):
    memo['filename'].append(file)
    indices = []
    path = "../dataset/diabetes_datasets/" + folder + "/" + file

    print(path)

    diabetes = pd.read_excel(path)

    diabetes['Date'] = pd.to_datetime(diabetes['Date'])

    GLs = np.empty((len(diabetes), 1), dtype=float)

    for (index, row) in diabetes.iterrows():
        # nullCnt = 0
        if pd.isnull(row['Dietary intake']):
            GLs[index] = np.nan
        elif row['Dietary intake'] == 'data not available':
            GLs[index] = default
        else:
            intakes = re.split(r'\s*[0-9]+\.?[0-9]*\s*[kgmlKGML]+\n*', row['Dietary intake'])
            if len(intakes) == 1:
                indices.append(index + 2)
            intakes.pop()
            if len(intakes) == 1 and intakes[0] == '':
                indices.append(index + 2)
                continue
            GL = 0.0
            # cnt = cnt + len(intakes)
            # print(cnt)
            # print(intakes)
            # print(dietary.loc[:len(intakes), 'dish'])
            for i in range(len(intakes)):
                GL += float(dietary.loc[i, 'GL'])
            len_o = len(dietary)
            # print(index)
            # if len(intakes) != 0:
            #     if intakes[0] != dietary.loc[0, 'dish']:
            #         print(intakes, '===?', dietary.loc[0, 'dish'])
            #         print("-------------------------")
            dietary.drop(dietary.index[: len(intakes)], inplace=True)
            dietary.reset_index(inplace=True, drop=True)
            GLs[index] = GL

    memo['index'].append(indices)

    diabetes.insert(loc=diabetes.columns.get_loc('Dietary intake'), column='GL', value=GLs)
    diabetes.drop(columns=['Dietary intake', '饮食'], inplace=True)

    if file.endswith('.xlsx'):
        file = file.replace('.xlsx', '.csv')
    elif file.endswith('.xls'):
        file = file.replace('.xls', '.csv')

    diabetes.to_csv(
        '../dataset/diabetes_datasets/' + folder + 'Processed' + '/' + 'processed' + file)


# files1 = os.listdir(r'E:\PycharmP\DAM\dataset\diabetes_datasets\Shanghai_T1DM')
# files2 = os.listdir(r'E:\PycharmP\DAM\dataset\diabetes_datasets\Shanghai_T2DM')
#
# for filename in files1:
#     dietary_process(filename, 'Shanghai_T1DM')
#
# for filename in files2:
#     dietary_process(filename, 'Shanghai_T2DM')
#
# memo = pd.DataFrame(memo)
#
# memo.to_csv('../dataset/diabetes_datasets/ProcessedMemo.csv')
#
# print(len(dietary))
# print('-----')
# print(memo)

files1 = os.listdir(r'E:\PycharmP\DAM\dataset\diabetes_datasets\Shanghai_T1DMProcessed')
files2 = os.listdir(r'E:\PycharmP\DAM\dataset\diabetes_datasets\Shanghai_T2DMProcessed')

for filename in files1:
    correct(filename, 'Shanghai_T1DMProcessed')

for filename in files2:
    correct(filename, 'Shanghai_T2DMProcessed')
