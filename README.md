# Diabetes Prediction

## About Dataset
#### This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

## Column Description
#### • Pregnancies
#### • Glucose
#### • Blood Pressure
#### • Skin Thickness
#### • Insulin
#### • BMI
#### • Diabetes
#### • Age
#### • Outcome

## Parts
### Part 1: Created a model to pridict the
#### • I opened my dataset and attempted to understand the type of analysis expected. Upon examination, I discovered that the dataset contains data suitable for predictive analysis.
#### • Predictive analysis is a type of analysis where this means understanding the probable future trends and behavior
#### • To derive predictions from the given dataset, I partitioned the data into independent and dependent variables. Upon analysis, I observed that all seven variables i.e. Pregnancies, Glucose, Blood pressure, skin thickness, Insulin, BMI , Diabetes, Age are independent, while 'Outcome' serves as the dependent variable. 
#### • From the given dataset, 0 represents that the person is not diabetic and 1 shows that the person is diabetic.

### Part 2: Created a simple django project for showcasing the result of this model for the prediction of new data.
