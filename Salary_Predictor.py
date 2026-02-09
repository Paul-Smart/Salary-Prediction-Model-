#salary prediction model based on 'Gender', 'Education Level', 'Job Title'.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pickle

#importing data 
df = pd.read_csv('Salarypred.csv')

#data cleaning before doing anything: delete empty rows 
df = df.dropna()

#separating the data: X values and Y values. 

X = df.drop('Salary', axis=1)
y = df['Salary']


#enccoding the X variables, because the computer cannot read the gender,...
X = pd.get_dummies(X, columns=['Gender', 'Education Level', 'Job Title'], drop_first= True)

#splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100) #we can print them out to know the %tages

#model building
lr = LinearRegression()
lr.fit(X_train, y_train)

#training the model to make prediction 
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

#evaluation the model accuracy(mae: to check how far are my predictions from the truth and r2 to check how well the model fits)
lr_train_mae = mean_absolute_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mae = mean_absolute_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

#cheking the results by printing them out
print(lr_train_mae, lr_train_r2, lr_test_mae, lr_test_r2)

#puting the data into a table 
lr_results = pd.DataFrame(['Linear Regression',lr_train_mae, lr_train_r2, lr_test_mae, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Trained MAE', 'Trained R2', 'Tested MAE', 'Tested R2']
print(lr_results)

#printing the r2 score
print(f"Testing R2: {lr_test_r2:.4f}")   #.4f means 4 decimals places
print(f"Testing R2: {lr_test_r2*100:.1f}%")

#evaluating the model performance
rf =  RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

#making the prediction'
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

#evaluating 
rf_train_mae = mean_absolute_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mae = mean_absolute_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

print(rf_train_mae, rf_train_r2, rf_test_mae, rf_test_r2)

rf_results = pd.DataFrame(['Random Forest', rf_train_mae, rf_train_r2, rf_test_mae, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Trained MAE', 'Trained R2', 'Tested MAE', 'Tested R2']
print(rf_results)

#comparing both tables 
df_models = pd.concat([lr_results, rf_results],axis=0).reset_index(drop=True)
print(df_models)

# Example: Predict salary for a new employee
new_employee = pd.DataFrame({
    'Age': [35],
    'Gender': ['Male'],
    'Education Level': ["Master's"],
    'Job Title': ['Data Scientist'],
    'Years of Experience': [7]
})

#encoding new employee data
new_employee_encoded = pd.get_dummies(new_employee, columns=['Gender', 'Education Level', 'Job Title'], drop_first=True)

#match training
new_employee_encoded = new_employee_encoded.reindex(columns=X_train.columns, fill_value=0)
print("New Employee Profile:")
print(new_employee)

predicted_salary = lr.predict(new_employee_encoded)[0]
print(f"Predicted salary: ${predicted_salary:,.0f}")

#visualization
z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.scatter(x= y_train, y=y_lr_train_pred, alpha=0.3, c="#90553c")
plt.plot(y_train, p(y_train), '#a3966a')

plt.xlabel("Actual Salaries ($)", fontsize=12, c='#80b9b1', )
plt.ylabel("Predicted Salaries ($)", fontsize=12, c='#2f7778')
plt.title("Actual Salaries vs Tested Salaries - Training Data", fontsize=14, c='#0b493a')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

#SAVING
#saving linear regression model
joblib.dump(lr, 'salary_model.pkl')
print("Model Saved as 'salary_model.pkl")

#saving random forest
joblib.dump(rf, 'random_forest_model.pkl')

#load model to use later
loaded_model = joblib.load('salary_model.pkl')

#load columns
with open ('training_columns.pkl') as  f:
    training_columns = pickle.load()
    
#save training columns(IMPORTANT!!!)
with open('training_columns.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

