import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


#Creating  simple dataset
data = {
    'distance_km': [2, 5, 8, 3, 10, 6, 7, 4, 12, 9],
    'traffic_level': [1, 3, 4, 2, 5, 3, 4, 2, 5, 4],
    'weather': [0, 1, 1, 0, 2, 1, 1, 0, 2, 2],
    'delivery_time_min': [15, 30, 45, 20, 60, 35, 40, 25, 70, 55]
}
df = pd.DataFrame(data)


#Spliting features and target

X = df[['distance_km', 'traffic_level', 'weather']]
y = df['delivery_time_min']

#Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#Train model

model = LinearRegression()
model.fit(X_train, y_train)

#Prediction

prediction = model.predict(X_test)

#Accuracy check

error = mean_absolute_error(y_test, prediction)
print("MAE Error:", error)

#New order prediction

distance = float(input("Enter the Distance: "))
Traffic= float(input("Enter the traffic level: "))
Weather =float(input("Enter the weather level: "))

new_order=[[distance,Traffic,Weather]]
time = model.predict(new_order)

print("Predicted delivery time:", round(time[0],2), "minutes")
