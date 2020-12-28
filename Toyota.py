import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt
cars = pd.read_csv("G://DS Assignments//MLR//ToyotaCorolla.csv")
cars.head()
cars = cars[cars.columns[2,3,6,8,12,13,14,15,16]]
type(cars)
cars = pd.DataFrame(cars)
cars1 = pd.DataFrame(cars, columns = ['Price','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'])
cars = cars1
print(cars1.head())
cars1.corr()
import seaborn as sns
sns.pairplot(cars1)
corr, _ = pearsonr(cars.HP, cars.cc)
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
model = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight" , data = cars1).fit()
model.summary()
pred = model.predict(cars1.iloc[ : , 1:9])
RMSE_pred = np.sqrt(np.mean((cars1.Price - pred)**2))
print(RMSE_pred)
Final_Table = ["Model" , "R_square", "RMSE"]
type(Final_Table)
Final_Table = pd.DataFrame(Final_Table, columns =["Model" , "R_square", "RMSE"])
print(Final_Table)
Final_Table = [[1,2,3],[4,5,6],[7,8,9]]
RMSE_model = RMSE_pred
model_pval = smf.ols("Price ~ Age_08_04 + KM + HP + Gears + Quarterly_Tax + Weight" , data = cars1).fit()
model_pval.summary()
pred_pval = model_pval.predict(cars1.iloc[ : ,[1,2,3,6,7,8]])
print(cars1.iloc[ :,8])
RMSE_pval = np.sqrt(np.mean((cars1.Price - pred_pval)**2))
print(RMSE_pval)
print(RMSE_model, RMSE_pval)
model_age = smf.ols("Price ~ Age_08_04" , data = cars1).fit()
model_age.summary()
model_KM = smf.ols("Price ~ Age_08_04 + KM",data = cars1).fit()
model_KM.summary()
model_HP = smf.ols("Price ~ Age_08_04 + KM + HP", data = cars1).fit()
model_HP.summary()
model_cc = smf.ols("Price ~ Age_08_04 + KM + HP + cc", data = cars1).fit()
model_cc.summary()
model_drs= smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors", data = cars1).fit()
model_drs.summary()
model_2= smf.ols("Price ~ Age_08_04 + KM + HP + Doors", data = cars1).fit()
model_2.summary()
model_Gr= smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors + Gears", data = cars1).fit()
model_drs.summary()
model_3 = smf.ols("Price ~ Age_08_04 + KM + HP + Gears", data = cars1).fit()
model_3.summary()
model_QT= smf.ols("Price ~ Age_08_04 + KM + HP + Quarterly_Tax", data = cars1).fit()
model_QT.summary()
model_3= smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors + Gears +  Quarterly_Tax", data = cars1).fit()
model_3.summary()
model_WT= smf.ols("Price ~ Age_08_04 + KM + HP + Quarterly_Tax + Weight", data = cars1).fit()
model_WT.summary()
pred_WT = model_WT.predict(cars1.iloc[ : , [1,2,3,7,8]])
RMSE_wt = np.sqrt(np.mean((cars1.Price - pred_WT)**2))
print(RMSE_wt)
print(cars1.iloc[ :,[1,2,3,7,8]])
import statsmodels.api as sm
sm.graphics.influence_plot(model)
print(cars[cars.rows[79]])
cars1.iat[79, 0]
cars_new = cars1.drop(cars1.index[[80]],axis = 0)
model_new = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight" , data = cars_new).fit()
model_new.summary()
sm.graphics.influence_plot(model_new)
print(cars[cars.rows[80]])
cars_new = cars1.drop(cars1.index[[80,960,221]],axis = 0)
model_WT_new = smf.ols("Price ~ Age_08_04 + KM + HP + Quarterly_Tax + Weight", data = cars_new).fit()
model_WT_new.summary()
model_new2 = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight" , data = cars_new).fit()
model_new2.summary()
model_final = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Quarterly_Tax + Weight" , data = cars_new).fit()
model_final.summary()
pred_final = model_final.predict(cars_new.iloc[ : , [1,2,3,4,7,8]])
RMSE_final = np.sqrt(np.mean((cars_new.Price - pred_final)**2))
print(RMSE_final)
pred_new2 = model_new2.predict(cars_new.iloc[ : , [1,2,3,4,5,6,7,8]])
RMSE_new2 = np.sqrt(np.mean((cars_new.Price - pred_new2)**2))
print(RMSE_new2)
np.mean(cars_new.Price - pred_new2)
a = (cars_new.Price - pred_new2) * (cars_new.Price - pred_new2)
np.sqrt(np.mean(a))
sm.graphics.plot_partregress_grid(model_final)
sm.graphics.plot_partregress_grid(model_new2)
mode_age = smf.ols("Age_08_04 ~ KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight" , data = cars_new).fit()
vif_age = 1/(1-model_age.rsquared)
print(vif_age)
model_KM = smf.ols("KM ~ Age_08_04 + HP + cc + Doors + Gears + Quarterly_Tax + Weight" , data = cars_new).fit()
vif_km = 1/(1-model_KM.rsquared)
print(vif_km)
model_HP = smf.ols("HP ~ Age_08_04 + KM + cc + Doors + Gears + Quarterly_Tax + Weight" , data = cars_new).fit()
vif_HP = 1/(1-model_KM.rsquared)
print(vif_HP)
model_cc = smf.ols("cc ~ Age_08_04 + KM + HP + Doors + Gears + Quarterly_Tax + Weight" , data = cars_new).fit()
vif_cc = 1/(1-model_cc.rsquared)
print(vif_cc)
model_Doors = smf.ols("Doors ~ Age_08_04 + KM + cc + HP + Gears + Quarterly_Tax + Weight" , data = cars_new).fit()
vif_Doors = 1/(1-model_Doors.rsquared)
print(vif_Doors)
model_Gears = smf.ols("Gears ~ Age_08_04 + KM + cc + HP + Doors + Quarterly_Tax + Weight" , data = cars_new).fit()
vif_Gears = 1/(1-model_Gears.rsquared)
print(vif_Gears)
model_QT = smf.ols("Quarterly_Tax ~ Age_08_04 + KM + cc + HP + Doors + Gears + Weight" , data = cars_new).fit()
vif_QT = 1/(1-model_QT.rsquared)
print(vif_QT)
model_wt = smf.ols("Weight ~ Age_08_04 + KM + cc + HP + Doors + Quarterly_Tax + Gears" , data = cars_new).fit()
vif_wt = 1/(1-model_wt.rsquared)
print(vif_wt)
plt.scatter(cars_new.Price, pred_final, c = "r")
plt.scatter(cars_new.Price, pred_final )
plt.hist(model_final.resid_pearson)
import pylab
import scipy.stats as st
st.probplot(model_final.resid_pearson, dist = "norm", plot = pylab)
print(cars_new)
sns.boxplot(data = cars_new, orient = "n", palette = "Set3")
cars_new.describe()
print(cars_new.describe().at["25%" , cars_new.KM])
KMQTR1 = cars_new['KM'].quantile(0.25)
print(KMQTR1)
print(cars_new["Price"].quantile(0.25))
KMQTR3 = cars_new["KM"].quantile(0.75)
KMIQR = KMQTR3 - KMQTR1
print(KMIQR)
LWRKM = KMQTR1 - (1.5*KMIQR)
UPRKM = KMQTR3 + (1.5*KMIQR)
Outlier_km = cars_new[cars_new["KM"] >UPRKM]
print(Outlier_km)
cars_final = cars_new.drop(cars_new.index[[185,186,187,188,189,190,377,378,379,380,381,602,603,604,605,606,607,608,609,610,611,612,1043,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069]],axis = 0)
cars_final.describe()
model_final1 = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight" , data = cars_final).fit()
model_final1.summary()
sm.graphics.plot_partregress_grid(model_final1)
sm.graphics.influence_plot(model_final1)
model_final2 = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Quarterly_Tax + Weight" , data = cars_final).fit()
model_final2.summary()
pred_final2 = model_final2.predict(cars_final.iloc[ : , [1,2,3,4,7,8]])
RMSE_final = np.sqrt(np.mean((cars_final.Price - pred_final2)**2))
model_outlier = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Quarterly_Tax + Weight" , data = Outlier_km).fit()
model_outlier.summary()
plt.hist(model_final2.resid_pearson)
#Model_final2. Cars_final, Pred_final2 and RMSE Final is the mail model#
def norm_func (i):
    x= (i - i.min())/(i.max() - (i.min()))
    return x

cars_norm = norm_func(cars_final.iloc[:, 1:])
cars_new.iloc[:, 0:]
from sklearn.model_selection import train_test_split
cars_norm['Price'] = cars_final['Price']
cars_train,cars_test = train_test_split(cars_norm,test_size = 0.2)
print(cars_train.head)
x = cars_norm.drop('Price', axis = 1)
y = cars_final[["Price"]]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 7)
x_train_norm = norm_func(x_train)
x_test_norm = norm_func(x_test)
cars_train_norm = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Quarterly_Tax + Weight" , data = cars_train).fit()
cars_train_norm.summary()
pred_train = cars_train_norm.predict(cars_train.iloc[ : , [0,1,2,3,6,7]])
RMSE_train = np.sqrt(np.mean((cars_train.Price - pred_train)**2))
sns.boxplot(data = cars_final, orient = "n", palette = "Set3")
pred_test = cars_train_norm.predict(cars_test.iloc[ : , [0,1,2,3,6,7]])
RMSE_test = np.sqrt(np.mean((cars_test.Price - pred_test)**2))
cars_test_model = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Quarterly_Tax + Weight" , data = cars_test).fit()
cars_test_model.summary()