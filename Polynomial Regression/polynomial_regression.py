import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting the polynomial to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualinsing the Linear Model
plt.scatter(X,y,color = 'red')
plt.plot(X, lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position levels')
plt.ylabel('Salary')
plt.show()


#Visualinsing the Polynomial Model
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position levels')
plt.ylabel('Salary')
plt.show()

#Predicting the result using Linear Regression
lin_reg.predict(6.5)

#Predicting the result using Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
