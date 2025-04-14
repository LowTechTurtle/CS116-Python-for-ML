import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import plotly.express as px
import plotly.graph_objects as go

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

data = pd.read_csv("USA_Housing.csv")

print(data.isna().sum())

data = data.drop(columns=["Address"])

scaler = MinMaxScaler()
data["Avg. Area Income"] = scaler.fit_transform(np.array(data['Avg. Area Income']).reshape(-1, 1))
data["Area Population"] = scaler.fit_transform(np.array(data['Area Population']).reshape(-1, 1))

y = data["Price"]
x = data.drop(columns=["Price"])
x_train, x_test = train_test_split(x, random_state=42, test_size=0.01)
y_train, y_test = train_test_split(y, random_state=42, test_size=0.01)

"""
models = {
    "SVR": (SVR(), {"C": [0.1, 1], "kernel": ["linear", "rbf"]}),
    "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5]}),
    "RandomForest": (RandomForestRegressor(), {"n_estimators": [50, 100]}),
    "DecisionTree": (DecisionTreeRegressor(), {"max_depth": [None, 10]}),
    "Ridge": (Ridge(), {"alpha": [0.1, 1]})
}
"""

models = {
    "SVR": (SVR(), {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "gamma": ["scale", "auto"],
        "epsilon": [0.01, 0.1, 1]
    }),
    "KNN": (KNeighborsRegressor(), {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2]  # Manhattan (1) or Euclidean (2) distance
    }),
    "RandomForest": (RandomForestRegressor(), {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }),
    "DecisionTree": (DecisionTreeRegressor(), {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "splitter": ["best", "random"]
    }),
    "Ridge": (Ridge(), {
        "alpha": [0.01, 0.1, 1, 10, 100],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        "tol": [1e-3, 1e-4, 1e-5]
    })
}
"""
best_model = None
best_score = float("inf")

for name, (model, params) in models.items():
    grid_search = GridSearchCV(model, params, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(x_train, y_train)
    
    y_pred = grid_search.best_estimator_.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"{name} Best Params: {grid_search.best_params_}, Test MSE: {mse}")

    if mse < best_score:
        best_score = mse
        best_model = (name, grid_search.best_estimator_)

print(f"\nBest Model: {best_model[0]} with Test MSE: {best_score}")
"""
results = []
predictions = {}

for name, (model, params) in models.items():
    # GridSearchCV tìm mô hình tối ưu
    grid_search = GridSearchCV(model, params, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True)
    grid_search.fit(x_train, y_train)

    # Lưu kết quả tìm kiếm siêu tham số
    for i in range(len(grid_search.cv_results_["params"])):
        results.append({
            "Model": name,
            "MSE": -grid_search.cv_results_["mean_test_score"][i],
            **grid_search.cv_results_["params"][i]
        })
    
    # Dự đoán trên tập test với mô hình mặc định
    default_model = model.fit(x_train, y_train)
    y_pred_default = default_model.predict(x_test)
    
    # Dự đoán với mô hình tối ưu
    best_model = grid_search.best_estimator_
    y_pred_optimized = best_model.predict(x_test)
    
    # Lưu kết quả dự đoán
    predictions[name] = {
        "y_test": y_test,
        "y_pred_default": y_pred_default,
        "y_pred_optimized": y_pred_optimized,
        "best_params": grid_search.best_params_
    }

# Chuyển kết quả tìm kiếm siêu tham số thành DataFrame
df_results = pd.DataFrame(results)

# Vẽ biểu đồ Parallel Coordinates
fig = px.parallel_coordinates(df_results, dimensions=df_results.columns[1:], color="MSE",
                              color_continuous_scale=px.colors.sequential.Viridis)
fig.update_layout(title="Parallel Coordinates Plot - Hyperparameter Tuning")
fig.show()

# Vẽ biểu đồ so sánh giá trị dự đoán và giá trị thực tế
for model_name, data in predictions.items():
    fig = go.Figure()
    
    # Scatter plot cho mô hình mặc định
    fig.add_trace(go.Scatter(x=data["y_test"], y=data["y_pred_default"], mode='markers', 
                             name="Default Model", marker=dict(color='red', opacity=0.5)))
    
    # Scatter plot cho mô hình tối ưu
    fig.add_trace(go.Scatter(x=data["y_test"], y=data["y_pred_optimized"], mode='markers', 
                             name="Optimized Model", marker=dict(color='blue', opacity=0.5)))
    
    # Đường y = x (đường tối ưu)
    fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], 
                             mode='lines', name="Optimal Line (y=x)", line=dict(color='green', dash='dash')))
    
    fig.update_layout(title=f"Predicted vs Actual - {model_name}",
                      xaxis_title="Actual Values",
                      yaxis_title="Predicted Values",
                      legend=dict(x=0.02, y=0.98),
                      width=700, height=500)
    
    fig.show()