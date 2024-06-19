from data_eda import data, colors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# ######################### Data Transformation And Preprocessing #########################
# Pivoting data from junction
df_J = data.pivot(columns="Junction", index="DateTime")
print("------------------ Statistical Characteristics of Vehicles on Junctions ------------------")
print(df_J.describe())

# Creating new sets
df_1 = df_J[[('Vehicles', 1)]]
df_2 = df_J[[('Vehicles', 2)]]
df_3 = df_J[[('Vehicles', 3)]]
df_4 = df_J[[('Vehicles', 4)]]
df_4 = df_4.dropna() # Junction 4 has limited data only for a few months

# Dropping level one in dfs's index as it is a multi index data frame
list_dfs = [df_1, df_2, df_3, df_4]
for i in list_dfs:
    i.columns = i.columns.droplevel(level=1)

# Function to plot comparative plots of dataframes
def sub_plots4(df_1, df_2, df_3, df_4, title):
    fig, axes = plt.subplots(4, 1, num="Dataframes", figsize=(12, 6), facecolor="DarkGray", sharey=True)
    fig.suptitle(title)
    # J1
    pl_1 = sns.lineplot(ax=axes[0], data=df_1, color=colors[0])
    axes[0].set(ylabel="Junction 1")
    # J2
    pl_2 = sns.lineplot(ax=axes[1], data=df_2, color=colors[1])
    axes[1].set(ylabel="Junction 2")
    # J3
    pl_3 = sns.lineplot(ax=axes[2], data=df_3, color=colors[2])
    axes[2].set(ylabel="Junction 3")
    # J4
    pl_4 = sns.lineplot(ax=axes[3], data=df_4, color=colors[3])
    axes[3].set(ylabel="Junction 4")

    plt.show()

# Normalize function
def normalize(df, col):
    average = df[col].mean()
    stdev = df[col].std()
    df_normalized = (df[col] - average) / stdev
    df_normalized = df_normalized.to_frame()
    return df_normalized, average, stdev

# Differencing function
def difference(df, col, interval):
    diff = []
    for i in range(interval, len(df)):
        value = df[col][i] - df[col][i - interval]
        diff.append(value)
    return diff

# Normalizing and Differencing to make the series stationary
df_N1, av_J1, std_J1 = normalize(df_1, "Vehicles")
Diff_1 = difference(df_N1, col="Vehicles", interval=(24*7)) # Taking a week's difference
df_N1 = df_N1[24*7:]
df_N1.columns = ["Norm"]
df_N1["Diff"] = Diff_1

df_N2, av_J2, std_J2 = normalize(df_2, "Vehicles")
Diff_2 = difference(df_N2, col="Vehicles", interval=(24)) # Taking a day's difference
df_N2 = df_N2[24:]
df_N2.columns = ["Norm"]
df_N2["Diff"] = Diff_2

df_N3, av_J3, std_J3 = normalize(df_3, "Vehicles")
Diff_3 = difference(df_N3, col="Vehicles", interval=1) # Taking an hour's difference
df_N3 = df_N3[1:]
df_N3.columns = ["Norm"]
df_N3["Diff"] = Diff_3

df_N4, av_J4, std_J4 = normalize(df_4, "Vehicles")
Diff_4 = difference(df_N4, col="Vehicles", interval=1) # Taking an hour's difference
df_N4 = df_N4[1:]
df_N4.columns = ["Norm"]
df_N4["Diff"] = Diff_4

# Stationary Check for the time series Augmented Dickey Fuller test
def stationary_check(df):
    check = adfuller(df.dropna())
    print(f"ADF Statistic: {check[0]}")
    print(f"p-value: {check[1]}")
    print("Critical Values:")
    for key, value in check[4].items():
        print('\t%s: %.3f' % (key, value))
    if check[0] > check[4]["1%"]:
        print("Time Series is Non-Stationary")
    else:
        print("Time Series is Stationary")

# Checking if the series is stationary
List_df_ND = [df_N1["Diff"], df_N2["Diff"], df_N3["Diff"], df_N4["Diff"]]
print("------------------ Checking the Transformed Series for Stationarity ------------------")
for i in List_df_ND:
    print()
    stationary_check(i)

# Differencing created some NA values as we took a weeks data into consideration while differencing
df_J1 = df_N1["Diff"].dropna()
df_J1 = df_J1.to_frame()

df_J2 = df_N2["Diff"].dropna()
df_J2 = df_J2.to_frame()

df_J3 = df_N3["Diff"].dropna()
df_J3 = df_J3.to_frame()

df_J4 = df_N4["Diff"].dropna()
df_J4 = df_J4.to_frame()

# Splitting the dataset
def split_data(df):
    training_size = int(len(df)*0.90)
    data_len = len(df)
    train, test = df[0:training_size], df[training_size:data_len]
    train, test = train.values.reshape(-1, 1), test.values.reshape(-1, 1)
    return train, test

# Splitting the training and test datasets
J1_train, J1_test = split_data(df_J1)
J2_train, J2_test = split_data(df_J2)
J3_train, J3_test = split_data(df_J3)
J4_train, J4_test = split_data(df_J4)

# Target and Feature
def target_and_feature(df):
    end_len = len(df)
    X = []
    y = []
    steps = 32
    for i in range(steps, end_len):
        X.append(df[i - steps:i, 0])
        y.append(df[i, 0])
    X, y = np.array(X), np.array(y)
    return X, y

# Fixing the shape of X_test and X_train
def feature_fix_shape(train, test):
    train = np.reshape(train, (train.shape[0], train.shape[1], 1))
    test = np.reshape(test, (test.shape[0], test.shape[1], 1))
    return train, test

# Assigning features and target
X_trainJ1, y_trainJ1 = target_and_feature(J1_train)
X_testJ1, y_testJ1 = target_and_feature(J1_test)
X_trainJ1, X_testJ1 = feature_fix_shape(X_trainJ1, X_testJ1)

X_trainJ2, y_trainJ2 = target_and_feature(J2_train)
X_testJ2, y_testJ2 = target_and_feature(J2_test)
X_trainJ2, X_testJ2 = feature_fix_shape(X_trainJ2, X_testJ2)

X_trainJ3, y_trainJ3 = target_and_feature(J3_train)
X_testJ3, y_testJ3 = target_and_feature(J3_test)
X_trainJ3, X_testJ3 = feature_fix_shape(X_trainJ3, X_testJ3)

X_trainJ4, y_trainJ4 = target_and_feature(J4_train)
X_testJ4, y_testJ4 = target_and_feature(J4_test)
X_trainJ4, X_testJ4 = feature_fix_shape(X_trainJ4, X_testJ4)

if __name__ == "__main__":
    # Plotting the dataframe to check for stationarity
    sub_plots4(df_1.Vehicles, df_2.Vehicles, df_3.Vehicles, df_4.Vehicles, "Dataframes Before Transformation")

    # Plots of transformed dataframe
    sub_plots4(df_N1.Diff, df_N2.Diff, df_N3.Diff, df_N4.Diff, "Dataframes After Transformation")
