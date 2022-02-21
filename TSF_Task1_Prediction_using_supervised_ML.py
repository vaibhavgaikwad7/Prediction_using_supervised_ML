# THE SPARKS FOUNDATION

TASK1 - PREDICTION USING SUPERVISED ML (LEVEL BEGINEER)

Predicting the percentage of a student based on the no. of study hours.


By- Vaibhav Vikas Gaikwad

#importing all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#reading data from given Sample data
Sample_data = pd.read_csv('http://bit.ly/w-data')
print("data imported succesfully")
Sample_data.head(10)

#Checking for Null values in the given dataset
Sample_data.isnull == True

As there are no null values in the dataset, we can now proceed to visualization

sns.set_style('whitegrid')
sns.scatterplot(x= Sample_data['Hours'], y= Sample_data['Scores'])
plt.title('Marks Vs Study Hours', size=20)
plt.xlabel('Hours Studied:', size = 15)
plt.ylabel('Marks in percent:', size = 15)
plt.show()

From the plot, it appears there there is a correlation between "Marks in percentage" and "Hours Studied".

#plotting a regression line to confirm correlation
sns.regplot(x = Sample_data['Hours'], y=Sample_data['Scores'])
plt.title('Regression Line ', size = 20)
plt.xlabel('Hours Studied:', size = 15)
plt.ylabel('Marks in percent:', size = 15)
plt.show()
print(Sample_data.corr())

# TRAINING THE MODEL

1. DIVIDING THE DATA

Dividing the data into attributes(inputs) and labels(outputs)

#defining x and y from the sample data
x = Sample_data.iloc[:, :-1].values
y = Sample_data.iloc[:, 1].values

#Spliting the Data into training and test sets
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)

2. FITTING DATA INTO THE MODEL

regression = LinearRegression()
regression.fit(train_x, train_y)
print("Model is trained successfully")

PREDICTING THE PERCENTAGE OF MARKS

predict_y = regression.predict(val_x)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_x], 'Predicted Marks': [k for k in predict_y]})
prediction

COMPARING THE BIAS BETWEEN ACTUAL AND PREDICTED MARKS VISUALLY

plt.scatter(x=val_x, y=val_y, color='Black')
plt.plot(val_x, predict_y, color='Blue')
plt.title('Actual vs Predicted (Marks)', size=22)
plt.xlabel('Hours Studied', size=14)
plt.ylabel('Marks Percentage', size=14)
plt.show()

TESTING THE MODEL

# Calculating the accuracy of the model
print('Mean absolute error of the model: ',mean_absolute_error(val_y,predict_y))

The value of Mean absolute error is very small.
Therefore it implies that chances of error or wrong forecasting through the model are *very less*

# THE PREDICTED SCORE IF A STUDENT STUDIES FOR 9.25 HRS/DAY

hours = [9.25]
answer = regression.predict([hours])
print("Score if a student studies for 9.25 hrs/ day = {}".format(round(answer[0],3)))

# According to the above regression model, the predicted score if a student studies for 9.25 hrs/day is *93.893* Marks.

