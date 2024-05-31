import numpy as np
import pandas as pd

diabetes = pd.read_excel("../dataset/diabetes_datasets/Shanghai_T1DM/1001_0_20210730.xlsx")

diabetes['Date'] = pd.to_datetime(diabetes['Date'])
diabetes.insert(len(diabetes.columns), 'Had Deal in 2 Hours', value=np.zeros(len(diabetes), dtype=bool).tolist())

for index in range(len(diabetes)):
    if index < 8:
        for i in range(index):
            if not pd.isna(diabetes.loc[index - i, 'Dietary intake']):
                diabetes.loc[index, 'Had Deal in 2 Hours'] = True
                break
    else:
        for i in range(8):
            if not pd.isna(diabetes.loc[index - i, 'Dietary intake']):
                diabetes.loc[index, 'Had Deal in 2 Hours'] = True
                break

# s.c.皮下注射 i.v.静脉注射  Non-insulin hypoglycemic agents非胰岛素的口服降糖药
diabetes = diabetes.drop(columns=['CBG (mg / dl)', '饮食', 'Insulin dose - s.c.', 'Insulin dose - i.v.',
                                  'Blood Ketone (mmol / L)'])

print(diabetes.columns)
