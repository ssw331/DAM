import numpy as np
import pandas as pd
import re
import os

dietary = {
    'dish': [],
    'net': [],
}


def dietary_process(diet, file, folder):
    path = "../dataset/diabetes_datasets/" + folder + "/" + file

    print(path)

    diabetes = pd.read_excel(path)

    diabetes['Date'] = pd.to_datetime(diabetes['Date'])

    for (index, row) in diabetes.iterrows():
        if pd.isnull(row['Dietary intake']) or row['Dietary intake'] == 'data not available':
            continue
        else:
            intakes = re.split(r'(\s*[0-9]+\.?[0-9]*\s*[kgmlKGML]+\n*)', row['Dietary intake'])
            intakes.pop()
            # intakes = row['Dietary intake'].split('\n')
            diet['dish'].extend(intakes[i] for i in range(0, len(intakes), 2))
            diet['net'].extend(intakes[i] for i in range(1, len(intakes), 2))

    # s.c.皮下注射 i.v.静脉注射  Non-insulin hypoglycemic agents非胰岛素的口服降糖药
    # Non-insulin hypoglycemic agents，Insulin dose - s.c.，Insulin dose - i.v. => 是否在一定时间内摄入N种药物：有则值>=1，无则=0
    # collect all the dietary intakes


files1 = os.listdir(r'E:\PycharmP\DAM\dataset\diabetes_datasets\Shanghai_T1DM')
files2 = os.listdir(r'E:\PycharmP\DAM\dataset\diabetes_datasets\Shanghai_T2DM')

for filename in files1:
    dietary_process(dietary, filename, 'Shanghai_T1DM')

for filename in files2:
    dietary_process(dietary, filename, 'Shanghai_T2DM')

dietary = pd.DataFrame(dietary)

for each in dietary:
    if pd.isnull(each):
        dietary.remove(each)

dietary.dropna(inplace=True)
dietary.reset_index(drop=True, inplace=True)

print(dietary)

dietary.to_csv("../dataset/diabetes_datasets/dietary.csv")
