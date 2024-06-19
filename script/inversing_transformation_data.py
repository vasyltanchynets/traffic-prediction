from data_eda import colors
from data_transformation_preprocessing import df_1, df_2, df_3, df_4, df_N1, df_N2, df_N3, df_N4, std_J1, std_J2, std_J3, std_J4, av_J1, av_J2, av_J3, av_J4
from model_training import PredJ1, PredJ2, PredJ3, PredJ4
import matplotlib.pyplot as plt
import seaborn as sns

# ######################### Inversing The Transformation Of Data #########################
# Functions to inverse transforms and Plot comparative plots invert difference forecast
def inverse_difference(last_ob, value):
    inversed = value + last_ob
    return inversed

# Plotting the comparison
def sub_plots2(df_1, df_2, title, m):
    fig, axes = plt.subplots(1, 2, num="Predictions And Originals", figsize=(12, 3), sharey=True, facecolor="DarkGray")
    fig.suptitle(title)

    pl_1 = sns.lineplot(ax=axes[0], data=df_1, color=colors[m])
    axes[0].set(ylabel="Prediction")

    pl_2 = sns.lineplot(ax=axes[1], data=df_2["Vehicles"], color="#627D78")
    axes[1].set(ylabel="Original")

    plt.show()

def inversing_J1():
    # Invert the Difference forecast for Junction 1
    recover1 = df_N1.Norm[-1412:-1].to_frame() # Length as per the diff
    recover1["Pred"] = PredJ1
    transform_reversed_J1 = inverse_difference(recover1.Norm, recover1.Pred).to_frame()
    transform_reversed_J1.columns = ["Pred_Normed"]

    # Invert the Normalization J1
    final_J1_Pred = (transform_reversed_J1.values * std_J1) + av_J1
    transform_reversed_J1["Pred_Final"] = final_J1_Pred

    # Plotting the Predictions with Originals
    sub_plots2(transform_reversed_J1["Pred_Final"], df_1[-1412:-1], "Predictions And Originals For Junction 1", 0)

def inversing_J2():
    # Invert the Difference J2
    recover2 = df_N2.Norm[-1426:-1].to_frame() # Length as per the diff
    recover2["Pred"] = PredJ2
    transform_reversed_J2 = inverse_difference(recover2.Norm, recover2.Pred).to_frame()
    transform_reversed_J2.columns = ["Pred_Normed"]

    # Invert the Normalization J2
    final_J2_Pred = (transform_reversed_J2.values * std_J2) + av_J2
    transform_reversed_J2["Pred_Final"] = final_J2_Pred

    # Plotting the Predictions with Originals
    sub_plots2(transform_reversed_J2["Pred_Final"], df_2[-1426:-1], "Predictions And Originals For Junction 2", 1)

def inversing_J3():
    # Invert the Difference J3
    recover3 = df_N3.Norm[-1429:-1].to_frame() # Length as per the diff
    recover3["Pred"] = PredJ3
    transform_reversed_J3 = inverse_difference(recover3.Norm, recover3.Pred).to_frame()
    transform_reversed_J3.columns = ["Pred_Normed"]

    # Invert the Normalization J3
    final_J3_Pred = (transform_reversed_J3.values * std_J3) + av_J3
    transform_reversed_J3["Pred_Final"] = final_J3_Pred

    # Plotting the Predictions with Originals
    sub_plots2(transform_reversed_J3["Pred_Final"], df_3[-1429:-1], "Predictions And Originals For Junction 3", 2)

def inversing_J4():
    # Invert the Difference J4
    recover4 = df_N4.Norm[-404:-1].to_frame() # Length as per the testset
    recover4["Pred"] = PredJ4
    transform_reversed_J4 = inverse_difference(recover4.Norm, recover4.Pred).to_frame()
    transform_reversed_J4.columns = ["Pred_Normed"]

    # Invert the Normalization J4
    final_J4_Pred = (transform_reversed_J4.values * std_J4) + av_J4
    transform_reversed_J4["Pred_Final"] = final_J4_Pred

    # Plotting the Predictions with Originals
    sub_plots2(transform_reversed_J4["Pred_Final"], df_4[-404:-1], "Predictions And Originals For Junction 4", 3)

inversing_J1()
inversing_J2()
inversing_J3()
inversing_J4()
