import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

import os
import joblib
from datetime import datetime


class DriftPredictor:
    """
    Used as a simple baseline. It works on the assumption that the movement from recent weeks would likely continue
    Next embedding = last embedding + average recent week movement/drift.
    """

    def __init__(self, lag):
        self.lag = lag # number of recent weeks to use for averaging recent velocity.

    def predict_next(self, history):
        
        if len(history) < 2: # handles short history
            return history[-1].copy()
        
        deltas = np.diff(history, axis=0) # calculates the weekly drifts.
        return history[-1] + deltas[-self.lag:].mean(axis=0) # average the n-lag average drift and adds it to the last centroid position 


# class PolynomialPredictor:
#     """
#     Fits a low-degree polynomial over time independently per embedding dimension.
#     """

#     def __init__(self, degree: int = degree):
#         self.degree = degree

#     def predict_next(self, history: np.ndarray) -> np.ndarray:
#         t = np.arange(len(history))
        
#         # Creates the polynomial design matrix.
#         X = np.vander(t, N=self.degree + 1, increasing=True)
        
#         # This fits the polynomial coefficients using least squares.
#         coef, *_ = np.linalg.lstsq(X, history, rcond=None)
        
#         # Creates the polynomial features for the next week.
#         x_next = np.vander([len(history)], N=self.degree + 1, increasing=True)
        
#         # Multiplies the next-time polynomial row by the fitted coefficients to predict the next centroid.
#         return (x_next @ coef)[0]
    
class PolynomialPredictor:
    """
    Polynomial regression with Ridge regularization.
    Usually safer than plain polynomial regression for noisy centroid trajectories.
    """

    def __init__(self, degree: int = 2, alpha: float = 1.0):
        self.degree = degree
        self.alpha = alpha

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        t = np.arange(len(history)).reshape(-1, 1)

        model = make_pipeline(
            PolynomialFeatures(degree=self.degree, include_bias=False),
            Ridge(alpha=self.alpha)
        )

        model.fit(t, history)

        t_next = np.array([[len(history)]])
        return model.predict(t_next)[0]

class KalmanPredictor:
    """
    Damped constant-velocity Kalman filter for centroid trajectory prediction.

    The state is [position, velocity]. For example, 32 PCA dimensions produce
    32 position values and 32 velocity values.

    This version is conservative for centroid movement that is small,
    jittery week-to-week, and slowly drifting over longer history.

    Instead of extending recent velocity forever, it damps velocity:

        next_position = current_position + velocity_decay * current_velocity
        next_velocity = velocity_decay * current_velocity
    """

    def __init__(
        self,
        dim,

        # How much the true centroid position can deviate from the motion model.
        q_pos,

        # How much the centroid velocity can change. Lower values make velocity
        # smoother and less reactive to week-to-week jitter.
        q_vel,

        # How noisy the observed weekly centroids are. Higher values smooth more.
        r,

        # Dampens velocity during filtering and forecasting to reduce overshoot.
        velocity_decay,
        
        # Number of recent historical weeks used to calculate the anchor.
        anchor_n_weeks,

        # Weight assigned to the recent-average centroid.
        # 0.0 = use only the filtered Kalman position.
        # 1.0 = use only the recent-average centroid.
        anchor_weight,
    ):
        self.dim = dim
        self.velocity_decay = velocity_decay
        self.anchor_n_weeks = anchor_n_weeks
        self.anchor_weight = anchor_weight

        I = np.eye(dim) # creates identity matrix
        Z = np.zeros((dim, dim)) # creates zero matrix

        # Damped transition matrix:
        #   new_position = old_position + velocity_decay * old_velocity
        #   new_velocity = velocity_decay * old_velocity
        self.A = np.block([
            [I, velocity_decay * I],
            [Z, velocity_decay * I],
        ])

        # We observe only centroid position. Velocity is estimated indirectly
        # from changes in centroid position over time.
        self.H = np.block([I, Z])

        # Q controls model flexibility. R controls how strongly the filter
        # trusts the observed weekly centroids.
        self.Q = np.block([
            [q_pos * I, Z],
            [Z, q_vel * I],
        ])
        
        # This controls how noisy the observed weekly centroids are.
        self.R = r * I

        # Initial covariance: This initializes uncertainty about the state.
        self.P0 = np.eye(2 * dim)

    def _filter(self, history: np.ndarray):
        """
        Smooth the historical centroid trajectory and estimate the latest
        position and velocity state.
        """
        # Ensures the input is a NumPy array of floats.
        history = np.asarray(history, dtype=float)

        if len(history) == 0:
            raise ValueError("history must contain at least one centroid.")

        # calculates initial velocity from week 1 and 2.
        initial_velocity = (
            history[1] - history[0]
            if len(history) > 1
            else np.zeros(self.dim)
        )
        
        # 1st week and the calculated initial velocity
        x = np.r_[history[0], initial_velocity]
        
        # Starts with initial uncertainty.
        P = self.P0.copy()

        for y in history[1:]:
            # 1. Predict this week's state from the previous state.
            x = self.A @ x # Predicts the current state: predicted_position = previous_position + damped_velocity \ predicted_velocity = damped_velocity
            P = self.A @ P @ self.A.T + self.Q # Updates uncertainty: After predicting forward, uncertainty changes \ Also add process noise Q because the motion model is not perfect.

            # 2. Correct the prediction using the observed weekly centroid.
            innovation = y - self.H @ x # actual observed centroid - predicted centroid
            S = self.H @ P @ self.H.T + self.R # This estimates how uncertain the prediction error is. model uncertainty + observation noise. How much should I trust this difference between actual and predicted?
            K = P @ self.H.T @ np.linalg.pinv(S) # The Kalman gain controls how much the model adjusts toward the observed centroid: K larger if observations are trusted more, K smaller if observations are noisy

            x = x + K @ innovation # Adjusts the hidden state based on correction.
            P = (np.eye(2 * self.dim) - K @ self.H) @ P # Updates uncertainty after using the actual observation: After seeing the actual centroid, the model is more informed.

        return x, P # Returns filtered state: x = latest smoothed position and velocity, P = latest uncertainty

    def predict_next(
        self,
        history: np.ndarray,
    ) -> np.ndarray:
        """
        Predict H1 using the anchored starting position and the original
        Kalman-estimated velocity.
        """
        return self.predict_path(
            history=history,
            n_steps=1,
        )[0]
    
    def predict_path(
        self,
        history: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """
        Generate a recursive multi-horizon forecast.

        Anchoring changes only the forecast starting position. The velocity and
        future trajectory remain based on the Kalman state estimated from the
        historical observations.

        The historical data is filtered only once. Predicted points are not fed
        back through the Kalman correction step.
        """
        history = np.asarray(history, dtype=float)

        if history.ndim != 2:
            raise ValueError(
                "history must have shape (n_weeks, dim)."
            )

        if len(history) == 0:
            raise ValueError(
                "history must contain at least one centroid."
            )

        if history.shape[1] != self.dim:
            raise ValueError(
                f"history has {history.shape[1]} dimensions, "
                f"but the model expects {self.dim}."
            )

        if n_steps <= 0:
            raise ValueError(
                "n_steps must be greater than 0."
            )

        # Estimate the latest position and velocity once from actual history.
        x, _ = self._filter(history)

        filtered_position = x[:self.dim].copy()
        filtered_velocity = x[self.dim:].copy()

        # Calculate the robust recent position anchor.
        n_anchor = min(
            self.anchor_n_weeks,
            len(history),
        )

        recent_average_position = history[
            -n_anchor:
        ].mean(axis=0)

        # Modify only the starting position.
        anchored_position = (
            self.anchor_weight * recent_average_position
            + (1.0 - self.anchor_weight) * filtered_position
        )

        # Preserve the original Kalman-estimated velocity.
        position = anchored_position
        velocity = filtered_velocity

        predictions = []

        for _ in range(n_steps):
            # Same transition as the original Kalman forecast.
            position = (
                position
                + self.velocity_decay * velocity
            )

            velocity = (
                self.velocity_decay * velocity
            )

            predictions.append(position.copy())

        return np.asarray(predictions)

def make_model(name,
               dim,
               model_config,
               lag,
               degree,
               q_pos,
               q_vel,
               r,
               velocity_decay,
               anchor_n_weeks,
               anchor_weight,
               **model_kwargs,
               ):
    """
    Instantiates the statistical model approaches given the approach name.

    model_config lets us pass tuned hyperparameters from Optuna while keeping
    the rest of the prediction code unchanged.
    """
    name = name.lower()

    config = {}
    if model_config is not None:
        config.update(model_config)
    config.update(model_kwargs)

    if name == "drift":
        return DriftPredictor(
            lag=config.get("lag", lag),
        )

    if name == "polynomial":
        return PolynomialPredictor(
            degree=config.get("degree", degree),
            alpha=config.get("alpha", 1.0),
        )

    if name == "kalman":
        return KalmanPredictor(
            dim=config.get("dim", dim),
            q_pos=config.get("q_pos", q_pos),
            q_vel=config.get("q_vel", q_vel),
            r=config.get("r", r),
            velocity_decay=config.get("velocity_decay", velocity_decay),
            anchor_n_weeks=config.get("anchor_n_weeks", anchor_n_weeks),
            anchor_weight=config.get("anchor_weight", anchor_weight),
        )

    raise ValueError(f"Unknown model: {name}")


def rolling_predictions(
    model,
    train_history,
    actual_future,
):
    """
    One-step-ahead evaluation: predict next, then reveal the true next point.
    """
    
    history = np.asarray(
        train_history,
        dtype=float,
    ).copy()

    actual_future = np.asarray(
        actual_future,
        dtype=float,
    )

    predictions = []

    for actual in actual_future:
        prediction = model.predict_next(
            history
        )

        predictions.append(prediction)

        # Add the actual observed centroid.
        history = np.vstack([
            history,
            actual,
        ])

    return np.asarray(predictions)


def recursive_predictions(
    model,
    history,
    n_steps,
):
    """
    Generate recursive multi-horizon predictions.

    Kalman uses predict_path so that:
    - history is filtered once;
    - anchoring modifies only the starting position;
    - the original estimated velocity continues through all horizons.
    """
    history = np.asarray(history, dtype=float)

    if isinstance(model, KalmanPredictor):
        return model.predict_path(
            history=history,
            n_steps=n_steps,
        )

    prediction_history = history.copy()
    predictions = []

    for _ in range(n_steps):
        prediction = model.predict_next(
            prediction_history
        )

        predictions.append(prediction)

        prediction_history = np.vstack([
            prediction_history,
            prediction,
        ])

    return np.asarray(predictions)

def save_stat_model(
    model_name,
    model_config,
    scaler,
    pca,
    project_name,
    input_prefix,
    model_prefix,
    cluster_col,
    seed,
    extra,
):
    """
    Save stat-forecast model configuration and fitted preprocessing objects.

    This is intended for stat models such as:
    - drift
    - polynomial
    - kalman

    It saves the fitted scaler and PCA so the same model-space representation
    can be reused later without refitting preprocessing.
    """
    artifact = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "model_config": model_config,
        "input_prefix": input_prefix,
        "model_prefix": model_prefix,
        "cluster_col": cluster_col,
        "seed": seed,
        "scaler": scaler,
        "pca": pca,
        "extra": extra or {},
    }
    
    
    model_out_path = f'projects/{project_name}/centroid_trajectory_predictions/{model_name}/model_configs'
    os.makedirs(model_out_path, exist_ok=True)

    joblib.dump(artifact, f'{model_out_path}/{model_name}_model.joblib')
    return artifact


def load_stat_model(
    model_name,
    project_name,
):
    """
    Load a saved stat-forecast model artifact.
    """
    model_out_path = f'projects/{project_name}/centroid_trajectory_predictions/{model_name}/model_configs'
    
    return joblib.load(f'{model_out_path}/{model_name}_model.joblib')