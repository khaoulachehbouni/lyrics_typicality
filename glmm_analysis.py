import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import gpboost as gpb 
import statsmodels as st
from scipy.stats import norm 



df_typ_csv = pd.read_csv("/content/df_all.csv")
df_typ_csv.head(2)

#Create typicality variables
df_typ_csv['typ_valence'] = abs(df_typ_csv['valence_1'] - df_typ_csv['valence_1'].mean())
df_typ_csv['typ_arousal'] = abs(df_typ_csv['arousal_1'] - df_typ_csv['arousal_1'].mean())
df_typ_csv['typ-hpoint'] = abs(df_typ_csv['hpoint'] - df_typ_csv['hpoint'].mean())
df_typ_csv['typ_nbChorus'] = abs(df_typ_csv['nbChorus'] - df_typ_csv['nbChorus'].mean())

df_typ_csv['score_typ'] = df_typ_csv[['Typ-variety', 'Typ-complexity', 'typ_valence', 'typ_arousal', 'typ-hpoint', 'typ_nbChorus', 'distTop3', 'kl_div']].sum(axis=1)
df_typ_csv['intercept'] = 1



def glmm(df, target_var, predictors, random_effects, type):

  X1 = df[predictors]
  Y1 = df[target_var]
  group_data = df[random_effects]

  gp_model = gpb.GPModel(group_data=group_data, likelihood=type)
  gp_model.fit(y=Y1, X=X1, params={"std_dev": True})
  gp_model.summary()


#SCORE TYPICALITE VS POPULARITY1
glmm(df_typ_csv, "popularity1", ["score_typ", "intercept"], ["artist","label"], "poisson")
#SCORE TYPICALITE VS POPULARITY2
glmm(df_typ_csv, "popularity2", ["score_typ", "intercept"], ["artist","label"], "poisson")
#SCORE TYPICALITE VS SKEWNESS
glmm(df_typ_csv, "skewness", ["score_typ", "intercept"], ["artist","label"], "gaussian")
#SCORE TYPICALITE VS KURTOSIS
glmm(df_typ_csv, "kurtosis", ["score_typ", "intercept"], ["artist","label"], "gaussian")
