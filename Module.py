# ===============================
# Import packages
# ===============================

import shap
import pickle
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from typing import Dict, Any, Tuple, List, Union
from pathlib import Path
from sklearn.base import BaseEstimator

# ===============================
# Set visualization
# ===============================

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
FEATURE_VIZ = False # viewing the data in a web browser

# ===============================
# Classifier definitions
# ===============================

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

dt = DecisionTreeClassifier(splitter="random", random_state=1) # Decision Tree
knn = KNeighborsClassifier(n_neighbors=8) # K-Nearest Neighbors
rf = RandomForestClassifier(n_estimators=500, random_state=1) # Random Forest
gbm = GradientBoostingClassifier(n_estimators=500, random_state=1) # Gradient Boosting
xgbm = XGBClassifier(objective="binary:logistic", use_label_encoder=False, eval_metric="logloss", random_state=1) # XGBoost
lgbm = LGBMClassifier(objective="binary", random_state=1) # LightGBM

# optional: dictionary of classifiers
classifiers = {
    "Decision Tree": dt,
    "KNN": knn,
    "Random Forest": rf,
    "Gradient Boosting": gbm,
    "XGBoost": xgbm,
    "LightGBM": lgbm
}

# ===============================
# Train the model
# ===============================

def classify(
    x_train,
    y_train,
    algorithm: BaseEstimator,
    k_num: Union[int, str] = "all"
) -> Pipeline:
    """
    Build and train a classification pipeline.
    """

    selector = SelectKBest(
        score_func=f_classif,
        k=k_num
    )

    pipeline = Pipeline([
        ("selector", selector),
        ("clf", algorithm),
    ])

    pipeline.fit(x_train, y_train)

    return pipeline

def run_balance_classification(
    x_train,
    y_train,
    x_test,
    y_test,
    algorithm: BaseEstimator,
    max_iter: int,
    filename: Union[str, Path],
) -> float:
    """
    Train multiple models and save the one with the best F1 score.
    """

    best_f1 = 0.0
    filename = Path(filename)

    for i in range(max_iter):
        model = classify(x_train, y_train, algorithm, "all")

        predictions = model.predict(x_test)
        current_f1 = f1_score(y_test, predictions)

        if current_f1 > best_f1:
            best_f1 = current_f1

            with open(filename, "wb") as f:
                pickle.dump(model, f)

            print(f"[Iteration {i+1}] Model saved")
            print(f"Best F1 score: {best_f1:.4f}\n")

    return best_f1

# ===============================
# Load and evaluate the model
# ===============================

def evaluate_saved_model(
    model_pkl_path: Union[str, Path],
    x_test,
    y_test,
) -> float:
    """
    Load a saved model and evaluate performance.
    """

    model_pkl_path = Path(model_pkl_path)

    with open(model_pkl_path, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(x_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Precision score : {precision:.4f}")
    print(f"Recall score    : {recall:.4f}")
    print(f"F1 score        : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return f1


class ModelEvaluator:
    def __init__(
            self, 
            features, 
            labels, 
            test_size: float = 0.3, 
            random_state: int = 1
    ):
        """
        Initialize evaluator and perform train/test split.
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state
        )

    def evaluate_models(
            self, 
            model_paths: Dict[str, Union[str, Path]]
    ) -> Dict[str, float]:
        """
        Evaluate multiple saved models and return their F1 scores.
        """
        results = {}

        for name, path in model_paths.items():
            print(f"Evaluating {name.value}...")
            try:
                results[name.value] = evaluate_saved_model(
                    model_pkl_path=path,
                    x_test=self.x_test,
                    y_test=self.y_test
                )
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                results[name] = None

        return results

    def get_max_f1_score(
            self, 
            scores: Dict[Any, float]
    ) -> Tuple[Any, float]:
        """
        Return the key and value corresponding to the highest F1 score.
        """
        filtered_scores = {
            k: v for k, v in scores.items() if v is not None
        }

        if not filtered_scores:
            raise ValueError("No valid scores to evaluate.")

        best_key = max(filtered_scores, key=filtered_scores.get)
        return best_key, filtered_scores[best_key]

# ===============================
# Interpret the model
# ===============================

def shap_interpretation(
    pipeline: Pipeline,
    x_values,
):
    """
    Compute SHAP values for tree-based models inside a pipeline.
    """

    model = pipeline["clf"]
    explainer = shap.TreeExplainer(model)

    return explainer.shap_values(x_values)


class ShapInterpreter:
    def __init__(
            self, 
            model_path: Union[str, Path], 
            features
    ):
        """
        SHAP interpretation for a trained model.
        """
        self.model_path = Path(model_path)
        self.features = features
        self.model = self._load_model()
        self.shap_values = self._compute_shap_values()

    def _load_model(self):
        """
        Load trained model from disk.
        """
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def _compute_shap_values(self):
        """
        Compute SHAP values once and store them.
        """
        return shap_interpretation(self.model, self.features)

    def summary_plot(self):
        """
        Display SHAP summary dot plot.
        """
        shap.summary_plot(self.shap_values, self.features)

    def summary_bar_plot(self):
        """
        Display SHAP summary bar plot.
        """
        shap.summary_plot(self.shap_values, self.features, plot_type="bar")

    def dependence_plots(
            self, 
            configs: List[Dict[str, str]]
    ):
        """
        Create multiple SHAP dependence plots.
        """
        fig, axes = plt.subplots(1, len(configs), figsize=(7 * len(configs), 5))

        if len(configs) == 1:
            axes = [axes]

        for ax, cfg in zip(axes, configs):
            shap.dependence_plot(
                cfg["feature"],
                self.shap_values,
                self.features,
                interaction_index=cfg.get("interaction"),
                ax=ax,
                show=False
            )
            ax.set_title(
                f'Dependence Plot: {cfg["feature"]} vs {cfg.get("interaction", "None")}'
            )

        plt.tight_layout()
        plt.show()

# ===============================
# Feature visualization
# ===============================

import plotly.io as pio
import plotly.graph_objs as go
pio.renderers.default = "browser" # viewing the data in a web browser

def set_plot(
    used_data: list,
    x_axis: str,
    y_axis: str,
    z_axis: str,
    title: str = "Solution Feature"
):
    """
    Display a 3D scatter chart.
    """
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title=x_axis, range=[0, 14]),
            yaxis=dict(title=y_axis, range=[0, 640]),
            zaxis=dict(title=z_axis, range=[0, 600])
        ),
        width=1000
    )

    fig = go.Figure(data=used_data, layout=layout)
    fig.show()

def _create_scatter3d(
        df: pd.DataFrame, 
        target_value: int, 
        x_axis: str, 
        y_axis: str, 
        z_axis: str, 
        size: int = 3, 
        color: str = "green", 
        line_width: int = 0, 
        name: str = ""
) -> go.Scatter3d:
    """
    Helper to create a 3D scatter plot for a specific target class.
    """
    color_map = {"green": "rgb(0,255,0)", "blue": "rgb(0,0,255)"}
    return go.Scatter3d(
        x=df[df["Target"] == target_value][x_axis],
        y=df[df["Target"] == target_value][y_axis],
        z=df[df["Target"] == target_value][z_axis],
        mode="markers",
        marker=dict(size=size, color=color_map.get(color, color), line=dict(width=line_width)),
        name=name
    )

def complete_comparison(
    used_df: pd.DataFrame,
    label1: str,
    label2: str,
    x_axis: str,
    y_axis: str,
    z_axis: str
):
    """
    Show comparison of 2 variants of data in 3D scatter plot.
    """
    used_df["Target"] = pd.to_numeric(used_df["Target"], errors="coerce")
    good = _create_scatter3d(used_df, 0, x_axis, y_axis, z_axis, color="green", name=label1)
    bad = _create_scatter3d(used_df, 1, x_axis, y_axis, z_axis, color="blue", name=label2)
    set_plot([good, bad], x_axis, y_axis, z_axis)


def mean_comparison(
    used_df: pd.DataFrame,
    label1: str,
    label2: str,
    x_axis: str,
    y_axis: str,
    z_axis: str
):
    """
    Show comparison of mean points of 2 variants of data in 3D scatter plot.
    """
    used_df["Target"] = pd.to_numeric(used_df["Target"], errors="coerce")

    def mean_point(target_val: int) -> dict:
        return {
            x_axis: used_df[used_df["Target"] == target_val][x_axis].mean(),
            y_axis: used_df[used_df["Target"] == target_val][y_axis].mean(),
            z_axis: used_df[used_df["Target"] == target_val][z_axis].mean()
        }

    mean_good = mean_point(0)
    mean_bad = mean_point(1)

    good_scatter = go.Scatter3d(
        x=[mean_good[x_axis]],
        y=[mean_good[y_axis]],
        z=[mean_good[z_axis]],
        mode="markers",
        marker=dict(size=10, color="rgb(0,255,0)", line=dict(width=3)),
        name=label1
    )

    bad_scatter = go.Scatter3d(
        x=[mean_bad[x_axis]],
        y=[mean_bad[y_axis]],
        z=[mean_bad[z_axis]],
        mode="markers",
        marker=dict(size=10, color="rgb(0,0,255)", line=dict(width=3)),
        name=label2
    )

    set_plot([good_scatter, bad_scatter], x_axis, y_axis, z_axis)