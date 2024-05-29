import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest

# series  a one-dimensional labeled array holding data of any type

wines_red = pd.read_csv("../dataset/winequality/winequality-red.csv", sep=';')
wines_white = pd.read_csv("../dataset/winequality/winequality-white.csv", sep=';')
# for combine two dataset
wines = pd.concat([wines_red, wines_white], axis='rows', ignore_index=True)

# Missing Value Handle
wines_red.dropna(how='any', inplace=True)
wines_white.dropna(how='any', inplace=True)
wines.dropna(how='any', inplace=True)

# Duplicate Data Handle
# index 行标签; column 列标签
if wines_red.index.is_unique:
    wines_red.drop_duplicates()

if wines_white.index.is_unique:
    wines_white.drop_duplicates()

if wines.index.is_unique:
    wines.drop_duplicates()

# Combine data
wines_red['total_acidity'] = wines_red['fixed acidity'] + wines_red['volatile acidity']
wines_white['total_acidity'] = wines_white['fixed acidity'] + wines_white['volatile acidity']
wines['total_acidity'] = wines['fixed acidity'] + wines['volatile acidity']

# Normalization

wines_red['quality'] = wines_red['quality'] / (wines_red['quality'].max() - wines_red['quality'].min())
wines_white['quality'] = wines_white['quality'] / (wines_white['quality'].max() - wines_white['quality'].min())
wines['quality'] = wines['quality'] / (wines['quality'].max() - wines['quality'].min())

# Feature Selection
top_three = SelectKBest(f_classif, k=3)

feature_select = wines_red.copy()
feature_select.pop('quality')
top_three.fit(feature_select, wines_red['quality'])

print(top_three.get_feature_names_out())

feature_select = wines_white.copy()
feature_select.pop('quality')
top_three.fit(feature_select, wines_white['quality'])

print(top_three.get_feature_names_out())

feature_select = wines.copy()
feature_select.pop('quality')
top_three.fit(feature_select, wines['quality'])

print(top_three.get_feature_names_out())

# Discretion

dis = pd.qcut(wines_red['fixed acidity'], q=3, labels=['Low', 'Medium', 'High'])  # qcut 基于样本分组; cut 基于数值分组
wines_red['fixed acidity'] = dis

dis = pd.qcut(wines_white['fixed acidity'], q=3, labels=['Low', 'Medium', 'High'])  # qcut 基于样本分组; cut 基于数值分组
wines_white['fixed acidity'] = dis

dis = pd.qcut(wines['fixed acidity'], q=3, labels=['Low', 'Medium', 'High'])  # qcut 基于样本分组; cut 基于数值分组
wines['fixed acidity'] = dis

# output
wines_red.to_csv("../results/wines_red.csv", sep=';')
wines_white.to_csv("../results/wines_white.csv", sep=';')
wines.to_csv("../results/wines.csv", sep=';')
