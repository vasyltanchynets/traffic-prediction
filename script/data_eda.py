from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# ######################### Loading Data #########################
# data = pd.read_csv("./traffic.csv")
# print("------------------ Loading Data ------------------")
# display(data.head())
# # print(data)
# print()

uid = "elt"
pwd = "demopass"
# sql db details
server = "localhost"
database = "traffic_prediction"
port = "5432"

# Create database connection
engine = create_engine(f"postgresql://{uid}:{pwd}@{server}:{port}/{database}")
sql = "SELECT * FROM public.traffic"

data = pd.read_sql_query(sql, engine)
print("------------------ Loading Data ------------------")
display(data.head())
print()

# ######################### Data Exploration #########################
# Parsing dates
data["DateTime"] = pd.to_datetime(data["DateTime"])
data = data.drop(["ID"], axis=1)  # dropping IDs
print("------------------ Structure and Сontent of the DataFrame ------------------")
data.info()
print()

# df to be used for EDA
df = data.copy()
colors = ["#800080", "#FFA500", "#FF0000", "#008000"]

# Plotting timeseries
def plotting_timeseries():
    plt.figure(num="Traffic On Junctions Over Years", figsize=(12, 5), facecolor="DarkGray")
    time_series = sns.lineplot(x=df['DateTime'], y="Vehicles", data=df, hue="Junction", palette=colors)
    time_series.set_title("Traffic On Junctions Over Years")
    time_series.set_ylabel("Number of Vehicles")
    time_series.set_xlabel("Date")
    plt.show()

# ######################### Feature Engineering #########################
# Exploring more features
def features_datetime():
    df["Year"] = df['DateTime'].dt.year
    df["Month"] = df['DateTime'].dt.month
    df["Date_no"] = df['DateTime'].dt.day
    df["Hour"] = df['DateTime'].dt.hour
    df["Day"] = df.DateTime.dt.strftime("%A")
    print("------------------ Feature Engineering ------------------")
    display(df.head())
    print()
features_datetime()

# ######################### Exploratory Data Analysis #########################
# Let's plot the timeseries
new_features = ["Year", "Month", "Date_no", "Hour", "Day"]

def plotting_timeseries_with_new_datetime_features():
    for i in new_features:
        plt.figure(figsize=(10, 5), facecolor="DarkGray")
        ax = sns.lineplot(x=df[i], y="Vehicles", data=df, hue="Junction", palette=colors)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def count_of_traffic_on_junctions_over_years():
    plt.figure(num="Count Of Traffic On Junctions Over Years", figsize=(12, 5), facecolor="DarkGray")
    count = sns.countplot(data=df, x=df["Year"], hue="Junction", palette=colors)
    count.set_title("Count Of Traffic On Junctions Over Years")
    count.set_ylabel("Number of Vehicles")
    count.set_xlabel("Date")
    plt.show()

def correlation():
    corrmat = df.corr(numeric_only=True)
    plt.subplots(num="Correlation", figsize=(10, 5), facecolor="DarkGray")
    sns.heatmap(corrmat, cmap="Pastel2", annot=True, square=True)
    plt.show()

def pair_plotting():
    sns.pairplot(data=df, hue="Junction", palette=colors)
    plt.show()

if __name__ == "__main__":
    plotting_timeseries()
    plotting_timeseries_with_new_datetime_features()
    count_of_traffic_on_junctions_over_years()
    correlation()
    # pair_plotting()
