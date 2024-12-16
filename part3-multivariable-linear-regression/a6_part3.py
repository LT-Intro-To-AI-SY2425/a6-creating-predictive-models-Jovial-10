import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles","age"]].values
y = data["Price"].values

#split the data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)

#create linear regression model

model = LinearRegression().fit(xtrain, ytrain)

#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 

coef =np.round(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared=round(model.score(xtest,ytest),2)

print("R squared value: ")
print(r_squared)

#Loop through the data and print out the predicted prices and the 
#actual prices
print("***************")
print("Testing Results")

predict= model.predict(xtest)

# x_predict = 89000
# x_predicttwo = 10
# # plug that value into your model
# prediction = model.predict([[x_predict], [x_predicttwo]])

# print("Prediction", prediction)

predict=np.around(predict,2)
print(predict)

car1 = np.array([[89000, 10]])  # 10-year-old car with 89,000 miles
car2 = np.array([[150000, 20]])  # 20-year-old car with 150,000 miles

predicted_price1 = model.predict(car1)[0]
predicted_price2 = model.predict(car2)[0]

print(f"Predicted price for 10-year-old car with 89,000 miles: ${round(predicted_price1, 2)}")
print(f"Predicted price for 20-year-old car with 150,000 miles: ${round(predicted_price2, 2)}")

print("\nTesting Multivariable Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index] # gets the actual y value from the ytest dataset
    predicted_y = predict[index] # gets the predicted y value from the predict variable
    x_coord = xtest[index] # gets the x value from the xtest dataset
    print(f"Miles Driven: {x_coord[0]} Age: {x_coord[1]} Actual: {actual} Predicted: {predicted_y}")
