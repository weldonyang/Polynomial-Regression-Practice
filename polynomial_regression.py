# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)                  # transforms OG features into the correct matrix of degrees 
lin_reg_2 = LinearRegression()                      # creating instance of LinearRegression()
lin_reg_2.fit(X_poly, y)                            # lin_reg.fit(X_poly, y) is now a PR model with degree = 4 

# Visualising the Linear Regression results
plt.scatter(X, y, color='red')                      
plt.plot(X, lin_reg.predict(X), color='blue')    
plt.title('Salary vs Position Level (LR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')                    
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')            
plt.title('Salary vs Position Level (PR)')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)                                             # returns an array of features with increment of .1 instead of 1, increasing the # of data points                           
X_grid = X_grid.reshape((len(X_grid), 1))                                           # returns a nx1 matrix of X_grid 
plt.scatter(X, y, color='red')                       
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')   # transforms X_grid into the correct matrix of degrees 
plt.title('Salary vs Position Level (PR)')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))                         #.predict() expects a matrix

# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))