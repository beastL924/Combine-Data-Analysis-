import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt 
from scipy import stats
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


#Reading the data from the database

data = pd.read_csv('synthetic_data.csv')

#print(data.head())

print(data.describe())

print(data.info())

#Checking the Educational level
Educational_level = data['education']

#Print test of the educational level
print(Educational_level)

#This shows that a only  a column is being dropped not a row, so axis= 1 is being used. 
X = data.drop(['purchase_amount'], axis=1)
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

#tThis perfromes OneHotEncoding and it get rid of the dummy variable trap by dropping repeated encoded value
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)


#the predicted value
y=data['purchase_amount']

X_train, X_test, y_train, y_test = train_test_split(X_encoded,y,test_size=0.2, random_state=42)

model=LinearRegression()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

#Calculates the Mean Squared Error
mse =mean_squared_error(y_test, y_predict)
mae=mean_absolute_error(y_test, y_predict)
r_square= r2_score(y_test, y_predict)
rmse = np.sqrt(mse)

print(f"The Mean Square Error is: {mse}")
print(f"The Mean Absolute Error is: {mae}")
print(f"The R^2 value is: {r_square}")
print(f"The RMSE value is { rmse}")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_predict, alpha=0.5, color='blue')
plt.xlabel('Annual Purchase Cost')
plt.ylabel('Predicted Cost')
plt.title('Regression Analysis of Purchase Amount')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.show()



# Use the column, drop missing values
income = data['income'].dropna()

# Fit a normal distribution
mu, sigma = stats.norm.fit(income)

# Generate x-values across the income range
x = np.linspace(income.min(), income.max(), 200)

# Compute the normal distribution curve
pdf = stats.norm.pdf(x, mu, sigma)

# Plot histogram and normal curve
plt.hist(income, bins=30, density=True, alpha=0.6, edgecolor='k')
plt.plot(x, pdf, linewidth=2, color='red')

plt.title('Income Distribution with Normal Curve')
plt.xlabel('Income')
plt.ylabel('Density')
plt.show()



purchase= data['purchase_amount'].dropna()

# Create the box plot
plt.boxplot(purchase, vert=True, patch_artist=True, 
            boxprops=dict(facecolor="#37C8BE", color='black'),
            medianprops=dict(color='yellow'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))

plt.title('Box Plot of Purchase')
plt.ylabel('Purchase Amount')
plt.show()

#Displaying the various educational level
level = data['education'].value_counts()
plt.title('Education by level')
plt.bar(level.index, level.values)
plt.show()



#Counting educational level by degree

master_count = data[data['education'] == 'Master'].shape[0]
print(f"The number of master students: {master_count}")

bachelor_count = data[data['education']=='Bachelor'].shape[0]
print(f"The number of Bachelor students are : { bachelor_count}")

#Counting by region using pie chart
#value_count()----> is used to count the number of times a region appears
data['region'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Distribution of Regions')
plt.show()


