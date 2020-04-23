
import tensorflow as tf
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('train.csv')

dataset = dataset.replace(["A","C","FV","I","RH","RL","RP","RM","C (all)"] , [0,1,2,3,4,5,6,7,8])
dataset = dataset.replace(["Reg","IR1","IR2","IR3"], [0,1,2,3])
dataset = dataset.replace(["AllPub","NoSewr","NoSeWa","ELO"], [0,1,2,3])
dataset = dataset.replace(["Blmngtn","Blueste","BrDale","BrkSide","ClearCr","CollgCr","Crawfor","Edwards","Gilbert","IDOTRR","MeadowV","Mitchel","Names","NAmes","NoRidge","NPkVill","NridgHt","NWAmes","OldTown","SWISU","Sawyer","SawyerW","Somerst","StoneBr","Timber","Veenker"] , [0,1,2,3,4,5,6,7,8,9,10,11,12,12,13,14,15,16,17,18,19,20,21,22,23,24])
dataset = dataset.replace(["1Fam","2FmCon","2fmCon","Duplx","Duplex","TwnhsE","TwnhsI","Twnhs"] , [0,1,1,2,2,3,4,5])
dataset = dataset.replace(["1Story","1.5Fin","1.5Unf","2Story","2.5Fin","2.5Unf","SFoyer","SLvl"] , [0,1,2,3,4,5,6,7])
dataset = dataset.replace(["GLQ","ALQ","BLQ","Rec","LwQ","Unf","NA"] , [0,1,2,3,4,5,6])
dataset = dataset.replace(["Ex","Gd","TA","Fa","Po"] , [0,1,2,3,4])
dataset = dataset.replace(["Elev","Gar2","Othr","Shed","TenC","NA"] , [0,1,2,3,4,5])
dataset = dataset.fillna(0)

y_train = dataset[["SalePrice"]]
x_train = dataset[["MSZoning","LotFrontage","LotArea","LotShape","Utilities","Neighborhood","BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinType1","TotalBsmtSF","HeatingQC","TotRmsAbvGrd","GarageCars","PoolArea","MiscFeature"]]

y_train = np.array(y_train)
x_train = np.array(x_train)

# y_train = y_train.astype(np.float32)
# x_train = x_train.astype(np.float32)

model = tf.keras.models.Sequential([
        Dense(19, input_dim=19 ,activation="sigmoid"),
        Dense(50, activation="sigmoid"),
        Dense(10, activation="sigmoid"),
        # Dense(4, activation="sigmoid"),
        Dense(1, activation="linear"),
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.1, nesterov=False), loss='mse')

out = model.fit(x_train, y_train, epochs=20)

plt.plot(out.history['loss'])
plt.show()

dataset = pd.read_csv('test.csv')

dataset = dataset.replace(["A","C","FV","I","RH","RL","RP","RM","C (all)"] , [0,1,2,3,4,5,6,7,8])
dataset = dataset.replace(["Reg","IR1","IR2","IR3"], [0,1,2,3])
dataset = dataset.replace(["AllPub","NoSewr","NoSeWa","ELO"], [0,1,2,3])
dataset = dataset.replace(["Blmngtn","Blueste","BrDale","BrkSide","ClearCr","CollgCr","Crawfor","Edwards","Gilbert","IDOTRR","MeadowV","Mitchel","Names","NAmes","NoRidge","NPkVill","NridgHt","NWAmes","OldTown","SWISU","Sawyer","SawyerW","Somerst","StoneBr","Timber","Veenker"] , [0,1,2,3,4,5,6,7,8,9,10,11,12,12,13,14,15,16,17,18,19,20,21,22,23,24])
dataset = dataset.replace(["1Fam","2FmCon","2fmCon","Duplx","Duplex","TwnhsE","TwnhsI","Twnhs"] , [0,1,1,2,2,3,4,5])
dataset = dataset.replace(["1Story","1.5Fin","1.5Unf","2Story","2.5Fin","2.5Unf","SFoyer","SLvl"] , [0,1,2,3,4,5,6,7])
dataset = dataset.replace(["GLQ","ALQ","BLQ","Rec","LwQ","Unf","NA"] , [0,1,2,3,4,5,6])
dataset = dataset.replace(["Ex","Gd","TA","Fa","Po"] , [0,1,2,3,4])
dataset = dataset.replace(["Elev","Gar2","Othr","Shed","TenC","NA"] , [0,1,2,3,4,5])
dataset = dataset.fillna(0)

x_test = dataset[["MSZoning","LotFrontage","LotArea","LotShape","Utilities","Neighborhood","BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinType1","TotalBsmtSF","HeatingQC","TotRmsAbvGrd","GarageCars","PoolArea","MiscFeature"]]

x_test = np.array(x_test)

x_test = x_test.astype(np.float32)
print(x_test[500])

dataset = pd.read_csv('sample_submission.csv')

y_test = dataset[["SalePrice"]]

y_test = np.array(y_test)

model.evaluate(x_test, y_test)

my_x_test = np.array([5,63,7560,0,0,12,0,0,5,5,1971,1971,3,864,2,5,2,0,0])

my_x_test = my_x_test.reshape(1, 19)

my_y_test = model.predict(my_x_test)

print(my_y_test)