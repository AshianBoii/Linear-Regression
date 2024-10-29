# Linear-Regression
Download data from https://archive.ics.uci.edu/ml/datasets/auto+mpg, where there are 9
attributes below:
1. mpg: continuous
2. cylinders: multi-valued discrete
3. displacement: continuous
4. horsepower: continuous
5. weight: continuous
6. acceleration: continuous
7. model year: multi-valued discrete
8. origin: multi-valued discrete
9. car name: string (unique for each instance)
You can remove the last column of “car name”, which is a unique value for each instance. The
regression problem is to predict “mpg” values using the other variables. So, the regression will
include 7 independent variables (2—8) and one dependent variable (1). You must use 10-fold
cross-validation. For each fold, you must provide the table below:
1. Coefficients of seven independent variables.
2. Root Mean Square Error (RMSE): The equation of RMSE is �∑ (𝑦𝑦𝑖𝑖 − 𝐗𝐗 𝐢𝐢 𝐛𝐛)𝑁𝑁
𝑖𝑖=1
2,
where 𝑦𝑦𝑖𝑖 is a ground truth value of the i-th test data, 𝐗𝐗 𝐢𝐢 𝐛𝐛 is a prediction from the linear
regression model, and N is a number of test data
<img width="709" alt="Screenshot 2024-10-15 at 1 48 13 PM" src="https://github.com/user-attachments/assets/991b90b5-ebd8-443d-bcc3-740b6a98bb8e">

To run- python3 Hw3.py
