# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 02:23:13 2020

@author: shara
"""

import pandas  as pd
import numpy as np
from matplotlib import pylab as plt
comp = pd.read_csv("G://DS Assignments//MLR//50_Startups.csv")
comp.head()
startups = startups.drop(columns = ["State"])
startups
startups.head()
startups.describe()
startups.corr()
import seaborn as sns
sns.pairplot(startups)
def norm_func(i):
    x = (i - i.min())/(i.max() - i.min())
    return x
startups.columns = startups.columns.str.replace(' ', '')
type(comp)
startups.columns = startups.columns.str.replace('&', '')
from sklearn import preprocessing
df_scale = pd.DataFrame(preprocessing.scale(startups.iloc[: , :3]))
type(df_scale)
df_scale.columns = ['RDSpend', 'Administration', 'MarketingSpend']
df_scale["Profit"] = startups["Profit"]
import statsmodels.formula.api as smf
df_model1 = smf.ols("Profit ~ RDSpend + Administration + MarketingSpend" , data = df_scale).fit()
df_model1.summary()
df_model2 = smf.ols("Profit ~ RDSpend " , data = df_scale).fit()
df_model2.summary()
df_model3 = smf.ols("Profit ~ Administration " , data = df_scale).fit()
df_model3.summary()
df_model4 = smf.ols("Profit ~ MarketingSpend " , data = df_scale).fit()
df_model4.summary()
df_model5 = smf.ols("Profit ~ RDSpend + Administration " , data = df_scale).fit()
df_model5.summary()
df_model6 = smf.ols("Profit ~ RDSpend + MarketingSpend " , data = df_scale).fit()
df_model6.summary()
df_model7 = smf.ols("Profit ~ Administration + MarketingSpend " , data = df_scale).fit()
df_model7.summary()
import statsmodels.api as sm
sm.graphics.influence_plot(df_model6)
vif_RD = smf.ols("RDSpend ~ Administration + MarketingSpend" , data = df_scale).fit()
vif_Admin = smf.ols("Administration ~ RDSpend + MarketingSpend" , data = df_scale).fit()
vif_MS = smf.ols("MarketingSpend ~ RDSpend + Administration" , data = df_scale).fit()
vif_R = 1/(1-vif_RD.rsquared)
vif_Ad = 1/(1-vif_Admin.rsquared)
vif_Market = 1/(1-vif_MS.rsquared)
from sklearn.model_selection import train_test_split
df_train,df_test = train_test_split(df_scale, test_size = 0.2)
df_train_model = smf.ols("Profit ~ MarketingSpend + RDSpend" , data = df_train).fit()
df_train_model.summary()
df_train_pred = df_train_model.predict(df_train)
df_train_error = df_train_pred - df_train.Profit
RMSE_train = np.sqrt(np.mean(df_train_error * df_train_error))
df_test_pred = df_train_model.predict(df_test)
df_test_error = df_test_pred - df_test.Profit
RMSE_test = np.sqrt(np.mean(df_test_error * df_test_error))
df_test_model = smf.ols("Profit ~ MarketingSpend + RDSpend" , data = df_test).fit()
df_test_model.summary()


# comp = norm_func(startups.iloc[: , :3])
# comp["Profit"] = startups["Profit"]
# import statsmodels.formula.api as smf
# comp.columns = comp.columns.str.replace(' ', '')
# type(comp)
# comp.columns = comp.columns.str.replace('&', '')
# model = smf.ols("Profit ~ RDSpend + Administration + MarketingSpend" , data = comp).fit()
# model.summary()
# import statsmodels.api as sm
# sm.graphics.influence_plot(model)
# comp_new = comp.drop(comp.index[[49]], axis = 0)
# model1 = smf.ols("Profit ~ RDSpend + Administration + MarketingSpend" , data = comp_new).fit()
# model1.summary()
# sm.graphics.influence_plot(model1)
# modelRDSpend = smf.ols("Profit ~ RDSpend ", data = comp_new).fit()
# modelRDSpend.summary()
# modelAdmin = smf.ols("Profit ~ RDSpend + Administration", data = comp_new).fit()
# modelAdmin.summary()
# modelMS = smf.ols("Profit ~ RDSpend + Administration + MarketingSpend", data = comp_new).fit()
# modelMS.summary()
# modelMarket = smf.ols("Profit ~ RDSpend + MarketingSpend", data = comp_new).fit()
# modelMarket.summary()
# pred_model1 = model1.predict(comp_new.iloc[ : , [0,1,2]])
# vif_RD = smf.ols("RDSpend ~ Administration + MarketingSpend" , data = comp_new).fit()
# vif_Admin = smf.ols("Administration ~ RDSpend + MarketingSpend" , data = comp_new).fit()
# vif_MS = smf.ols("MarketingSpend ~ RDSpend + Administration" , data = comp_new).fit()
# vif_R = 1/(1-vif_RD.rsquared)
# vif_Ad = 1/(1-vif_Admin.rsquared)
# vif_Market = 1/(1-vif_MS.rsquared)
# pred = modelMarket.predict(comp_new.iloc[ : , [0,1,2]])
# sm.graphics.plot_partregress_grid(model1)
# sm.graphics.plot_partregress_grid(modelMarket)
# sm.graphics.influence_plot(model1)
# pred_1 = modelMarket.predict(comp_new)
# pred.head()
# pred_1.head()
# RMSE = np.sqrt(np.mean((comp_new.Profit - pred)**2))
# plt.hist(comp['RDSpend'])
# plt.hist(comp['MarketingSpend'])
# plt.hist(comp['Administration'])
# plt.hist(comp['RDSpend'])
# logr = np.log(comp.RDSpend)
# logr.head()
# plt.hist('logr')
# logtable = comp
# logtable['logr'] = logr
# plt.hist(logtable['logr'])
# startups.head()
# df = startups
# df.columns = df.columns.str.replace(' ', '')
# type(comp)
# df.columns = df.columns.str.replace('&', '')
# df['logr'] = np.log(df.RDSpend)
# df.describe()
# plt.hist(df['RDSpend'])
# df.iloc[49,4] = 0
# DF.HEAD()
# df.head()
# startups.head()
# startups = pd.read_csv("G://DS Assignments//MLR//50_Startups.csv")
# startups = startups.drop(columns = ["State"])
# startups.columns = startups.columns.str.replace(' ', '')
# type(comp)
# startups.columns = startups.columns.str.replace('&', '')
# sns.boxplot(data= startups,orient = "n", palette="Set3")
# ?boxplot
# Q1 = startups["Administration"].quantile(0.25)
# Q3 = startups["Administration"].quantile(0.75)
# startups.describe()
# IQR = Q3 -Q1
# LIV = Q1 - (1.5*IQR)
# UIV = Q3 + 1.5 * IQR
# startups[startups["Administration"] > UIV]
# startups["MarketingSpend"].median()
# startups["MarketingSpend"].mean()
# startups["RDSpend"].median()
# startups["RDSpend"].mean()
# startups["Administration"].median()
# startups["Administration"].mean()
# startups = startups.replace(to_replace = 0, value = startups.mean())
# startups_new = norm_func(startups.iloc[: , :3])
# plt.hist(startups_new["RDSpend"])
# startups_new["Profit"] = startups["Profit"]
# model_new1 = smf.ols("Profit ~ RDSpend + Administration + MarketingSpend" , data = startups_new).fit()
# model_new1.summary()
# sm.graphics.influence_plot(model_new1)
# startups_new = startups_new.drop(comp.index[[49]], axis = 0)
# model_new1 = smf.ols("Profit ~ RDSpend + Administration + MarketingSpend" , data = startups_new).fit()
# model_new1.summary()
# model_new2 = smf.ols("Profit ~ (RDSpend) + Administration + MarketingSpend" , data = startups_new).fit()
# pred_new1 = model_new1.predict(startups_new)
# RMSE_new1 = np.sqrt(np.mean((startups_new.Profit - pred_new1)**2))
# plt.hist()
# del logr
# del logtable
# df = df.drop(columns = ["State"])
# from sklearn.neighbors import KNeighborsClassifier as KNC
# neigh = KNC(n_neighbors = 9)
# df_train = df.iloc[0:39, 0:4]
# df_test = df.iloc[39:50, 0:4]
# dec = (df_train.iloc[:, 0]).round(decimals = 0)
# K_model = neigh.fit(df_train.iloc[:, 1:4], dec)
# pred_k  = pd.DataFrame(neigh.predict(df_test.iloc[:, 1:4])) == df_test.iloc[:, 0])
# pred_k
# dec1 = (df.iloc[:, [0,1,2,3]]).round(decimals = 0)
# dec1_model = neigh.fit(dec1.iloc[:, 1:4], dec1.iloc[:, 0])
# pred_dec1  = pd.DataFrame(neigh.predict(df.iloc[:, 1:4])) == df.iloc[:, 0]
# dec2 = dec1.drop(dec1.index[[47,49]], axis = 0)
# dec2_model = neigh.fit(dec2.iloc[:, [0,1,3]], dec2.iloc[:, 2])
# pred_dec2  = pd.DataFrame(neigh.predict(dec2.iloc[:, [0,1,3]]))
# dec2["MarketingSpend"].median()
# dec2.iloc[19,2] = 225529
# dec2_model2 = smf.ols("Profit ~ RDSpend + Administration + MarketingSpend" , data = dec2).fit()
# dec2_model2.summary()
# dec2_norm = norm_func(dec2.iloc[: , :3])
# dec2_norm["Profit"] = dec2["Profit"]
# dec2_norm_model2 = smf.ols("Profit ~ RDSpend + Administration + MarketingSpend" , data = dec2_norm).fit()
# dec2_norm_model2.summary()
# dec3 = smf.ols("Profit ~ RDSpend + MarketingSpend" , data = dec2_norm).fit()
# dec3.summary()
# dec4 = smf.ols("Profit ~ Administration + MarketingSpend" , data = dec2_norm).fit()
# dec4.summary()
# dec5 = smf.ols("Profit ~ RDSpend + Administration" , data = dec2_norm).fit()
# dec5.summary()
# sm.graphics.influence_plot(dec3)
# dec2 = dec2_norm.drop(dec2_norm.index[[47]], axis = 0)
# Dec6 = smf.ols("Profit ~ RDSpend + Administration + MarketingSpend" , data = dec2).fit()
# Dec6.summary()
# dec7 = smf.ols("Profit ~ RDSpend + MarketingSpend" , data = dec2).fit()
# dec7.summary()
# dec8 = smf.ols("Profit ~ Administration + MarketingSpend" , data = dec2).fit()
# dec8.summary()
# dec9 = smf.ols("Profit ~ RDSpend + Administration" , data = dec2).fit()
# dec9.summary()
# z1 = dec1
# z1 = z1.drop(z1.index[[47,49]], axis = 0)
# z1.iloc[19,2] = 225529
# z1["MarketingSpend"].median()
# z1.iloc[47,2] = 227345
# z_model1 = smf.ols("Profit ~ RDSpend + Administration +MarketingSpend" , data = z1).fit()
# z_model1.summary()
# final_model = smf.ols("Profit ~ MarketingSpend + RDSpend" , data = z1).fit()
# final_model.summary()
# from sklearn.model_selection import train_test_split
# z_train,z_test = train_test_split(z1, test_size = 0.2)
# model_train = smf.ols("Profit ~ MarketingSpend + RDSpend" , data = z_train).fit()
# model_train.summary()
# train_pred = model_train.predict(z_train[['MarketingSpend','RDSpend']])
# train_error = train_pred - z1.Profit
# z1 = z1.reset_index(drop = True)
# train_error = train_pred - z1.Profit



# from sklearn import preprocessing
# z_scale = pd.DataFrame(preprocessing.scale(z1.iloc[: , :3]))
# type(z_scale)
# z_scale.columns = ['RDSpend', 'Administration', 'MarketingSpend']
# z_scale["Profit"] = z1["Profit"]
# z_scale.iloc[47,3] = 35673.0
# z_model2 = smf.ols("Profit ~ RDSpend + Administration + MarketingSpend" , data = z_scale).fit()
# z_model2.summary()
# z_model3 = smf.ols("Profit ~ RDSpend" , data = z_scale).fit()
# z_model3.summary()
# z_model4 = smf.ols("Profit ~ RDSpend + Administration " , data = z_scale).fit()
# z_model4.summary()
# z_model5 = smf.ols("Profit ~ RDSpend + MarketingSpend " , data = z_scale).fit()
# z_model5.summary()
# sm.graphics.influence_plot(z_model5)
# z_model6 = smf.ols("Profit ~ Administration + MarketingSpend " , data = z_scale).fit()
# z_model6.summary()
# plt.hist(z_scale["RDSpend"])
# plt.hist(z_scale["Administration"])
# plt.hist(z_scale["MarketingSpend"])
# z_model7 = smf.ols("Profit ~ MarketingSpend" , data = z_scale).fit()
# z_model7.summary()
# z_model8 = smf.ols("Profit ~ Administration " , data = z_scale).fit()
# z_model8.summary()
# z_scale["Administration"].mean()
# x1 = z1
# plt.hist(x1["RDSpend"])
# plt.hist(x1["Administration"])
# plt.hist(x1["MarketingSpend"])
# x1["logRD"] = np.log(x1["RDSpend"])
# plt.hist(x1["logRD"])
# x1["RDSQ"] = (x1["RDSpend"]) * (x1["RDSpend"])
# plt.hist(x1["RDSQ"])
# x1["logSQ"] = np.log(x1["Administration"])
# plt.hist(x1["logSQ"])
# x1 = x1.drop(columns = ["RDSQ" , "ADSQ"])
# x1["logMS"] = np.log(x1["MarketingSpend"])
# x1_model = smf.ols("Profit ~ logRD + logSQ + logMS" , data = x1).fit()
# x1_model.summary()
# x1_model2 = smf.ols("Profit ~ logRD" , data = x1).fit()
# x1_model2.summary()
# x1_model3 = smf.ols("Profit ~ logSQ" , data = x1).fit()
# x1_model3.summary()
# x1_model4 = smf.ols("Profit ~ logMS" , data = x1).fit()
# x1_model4.summary()
# x1_model5 = smf.ols("Profit ~ logMS + logSQ" , data = x1).fit()
# x1_model5.summary()
# x1_model6 = smf.ols("Profit ~ logRD + logSQ" , data = x1).fit()
# x1_model6.summary()
# x1_model7 = smf.ols("Profit ~ logRD + logMS" , data = x1).fit()
# x1_model7.summary()
