#Yapay Zekaya Giris dersi odevi icin hazirlanmistir.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error , mean_squared_error
from sklearn.model_selection import train_test_split

veriseti = pd.read_csv("dataset.csv")
veriseti.head() #/ verinin ilk beş satırını göstermek için kullandık
veriseti.isnull()  # veri setindeki bos degerleri
veriseti.isnull().sum()
veriseti.describe() # verinin detaylı bilgilerini gösterir

X = veriseti.Leo #verisetinde lineer regresyon yapmak icin x degerlerine yazilacak veriler
Y = veriseti.Girlfriend #burasi da y degerleri icin
X.values
# Leonardo DiCaprio ve sevgililerinin yaslari arasındaki iliski
plt.scatter(X.values, Y.values)
plt.xlabel('Leonun yasi')
plt.ylabel('Kiz arkadas yasi')
plt.title('leonun sevgili yas grafigi')

#model kurma
lr = LinearRegression()
lr.fit(X.values.reshape(-1,1), Y.values.reshape(-1,1))
lr.coef_ , lr.intercept_

print("kurulan regresyon modeli Y = {} + {}*x".format(lr.intercept_[0].round(2), lr.coef_[0][0].round(2)))
y_predicted = lr.predict(X.values.reshape(-1,1))
y_predicted
r2_score(Y, y_predicted)
df = pd.DataFrame({'Y':Y.values.flatten(), 'y_predict':y_predicted.flatten()}) 
df
mean_absolute_error(Y, y_predicted) # ortalama mutlak
mean_squared_error(Y, y_predicted, squared=False) # ortalama karesel
b0 = lr.intercept_[0].round(2)
b1 = lr.coef_[0][0].round(2)
b0,b1
random_x = np.array([24,55])

plt.plot(random_x, b0 + b1*random_x, color='red', label='regresyon' )
plt.scatter(X.values, Y.values)
plt.legend()
plt.xlabel('Leonardo Yas')
plt.ylabel('GF yas')
plt.title('Leo ve Gf yaslari arasindaki regresyon')
# model denklem grafiği
plt.plot(random_x, b0 + b1*random_x, color='red', label='regresyon' )
plt.legend()
plt.xlabel('Leonardonun yasi')
plt.ylabel('Gflerinin yasi')
plt.title('leonardo ve gf yaslari arasindaki model denklemi')

#asagidaki satirlar ise bize Leonardo'nun dataset icerisinde bulunmadigi yasindaki
#bir veriyi lineer regresyon kullanarak tahmin edilmesini saglamaktadir.
print("kurulan regresyon modeli Y = {} + {}*x".format(b0, b1))
tahmin_yıllık_deneyim=np.array([24 , 32 , 45 , 50 , 55])
tahmin_maas = lr.predict(tahmin_yıllık_deneyim.reshape(-1,1))
plt.plot(random_x, b0 + b1*random_x, color='red', label='regresyon' )
plt.scatter(tahmin_yıllık_deneyim,tahmin_maas.reshape(-1,1) )
plt.legend()
plt.xlabel('Leo Age')
plt.ylabel('GF Age')
plt.title('Leo nun yaşına göre kız arkadaşlarının yaşı')







