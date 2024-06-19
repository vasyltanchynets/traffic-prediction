from model_building import GRU_model, RMSE_value, predictions_plot
from data_transformation_preprocessing import X_trainJ1, X_trainJ2, X_trainJ3, X_trainJ4, y_trainJ1, y_trainJ2, y_trainJ3, y_trainJ4, X_testJ1, X_testJ2, X_testJ3, X_testJ4, y_testJ1, y_testJ2, y_testJ3, y_testJ4

# ######################### Fitting The Model #########################
# Predictions for First Junction
PredJ1 = GRU_model(X_trainJ1, y_trainJ1, X_testJ1)

# Results for J1
RMSE_J1 = RMSE_value(y_testJ1, PredJ1)
# predictions_plot(y_testJ1, PredJ1, 0)

# Predictions for Second Junction
PredJ2 = GRU_model(X_trainJ2, y_trainJ2, X_testJ2)

# Results for J2
RMSE_J2 = RMSE_value(y_testJ2, PredJ2)
# predictions_plot(y_testJ2, PredJ2, 1)

# Predictions for Third Junction
PredJ3 = GRU_model(X_trainJ3, y_trainJ3, X_testJ3)

# Results for J3
RMSE_J3 = RMSE_value(y_testJ3, PredJ3)
# predictions_plot(y_testJ3, PredJ3, 2)

# Predictions for Forth Junction
PredJ4 = GRU_model(X_trainJ4, y_trainJ4, X_testJ4)

# Results for J4
RMSE_J4 = RMSE_value(y_testJ4, PredJ4)
# predictions_plot(y_testJ4, PredJ4, 3)
