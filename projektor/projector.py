# projektor/projector.py
"""
Module: projector.py

This module provides two classes:

  1. Projector
     - Aligns experimental logs (e.g. training error, OT distance, accuracy) to a fixed length.
     - Computes binary masks and scaling matrices.
     - Prepares a regression data matrix (features and targets) from experimental logs.
     - Fits a linear regression model to predict performance (e.g. test accuracy) from the generated features.
     - Provides performance prediction for new feature vectors.
     - Extrapolates performance to new data scales.

     Example usage:
     --------------
     >>> from projektor.projector import Projector
     >>> # Suppose reserrlog, otlog, and accs are lists of 1-D arrays from experiments.
     >>> proj = Projector(pad_length=11, threshold=5)
     >>> aligned_res = proj.align_data(reserrlog)
     >>> mask_low, mask_high = proj.compute_masks(aligned_res)
     >>> # Use the high-index mask for feature preparation.
     >>> X, y = proj.prepare_features(reserrlog, otlog, accs, mask_high)
     >>> results = proj.fit(X, y, verbose=True, plot=True)
     >>> new_feature_vector = X[0]  # Example: use the first feature vector
     >>> predicted_perf = proj.predict(new_feature_vector)
     >>> print("Predicted performance:", predicted_perf)
     >>> # Extrapolate performance to a new data scale:
     >>> projected_perf = proj.project_performance(n=10000, n0=1000, n1=1500, perf0=0.75, perf1=0.80)
     >>> print("Projected performance for n=10000:", projected_perf)

  2. OptimalDataSourceSelector
     - Implements a gradient–based optimization to select an optimal mixing ratio
       among an arbitrary number (k) of data sources.
     - Uses dual solutions and OT distances from two experiments (at data scales n0 and n1)
       along with user-specified gradient parameters for each source.
     - Updates the mixing ratios using a decayed learning rate and projects the result
       onto the probability simplex.

     Example usage:
     --------------
     >>> from projektor.projector import OptimalDataSourceSelector
     >>> # data_n0 and data_n1 are dictionaries with keys:
     >>> # 'dual': concatenated dual solution vector,
     >>> # 'ds_lengths': list or array of lengths for each source,
     >>> # 'full_len': total number of samples,
     >>> # 'ot_dist': measured OT distance (scalar).
     >>> # q_init is an initial numpy array of shape (k,) that sums to 1.
     >>> selector = OptimalDataSourceSelector(lr=5e-4, iterations=100, decay=0.90)
     >>> optimal_q = selector.optimize(n=10000, n0=1000, n1=1500,
     ...                               data_n0=data_n0, data_n1=data_n1,
     ...                               q_init=np.array([0.33, 0.33, 0.34]),
     ...                               params0=[{'b2':0, 'b1':0, 'b0':0, 'c2':0, 'c1':0}]*3,
     ...                               params1=[{'b2':0, 'b1':0, 'b0':0, 'c2':0, 'c1':0}]*3)
     >>> print("Optimal mixing ratios:", optimal_q)

Dependencies:
    - numpy, scikit-learn, matplotlib

"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


# ----------------------------------
# Projector Class
# ----------------------------------
class Projector:
    def __init__(self, pad_length: int = 11, threshold: int = 5):
        """
        Initialize the Projector.

        Parameters:
            pad_length (int): The fixed length to which each experimental log array is padded.
            threshold (int): The index threshold used in mask generation.
        """
        self.pad_length = pad_length
        self.threshold = threshold
        self.model = None  # Will hold the fitted regression model

    def align_data(self, data_list: list) -> np.ndarray:
        """
        Aligns a list of 1-D arrays (or lists) to a fixed length by padding with zeros.

        Parameters:
            data_list (list): A list of numeric arrays/lists.

        Returns:
            np.ndarray: An array of shape (n, pad_length), where n is the number of arrays.
        """
        aligned = []
        for arr in data_list:
            arr = np.asarray(arr, dtype=float)
            if arr.size < self.pad_length:
                pad = np.zeros(self.pad_length - arr.size)
                arr = np.concatenate([arr, pad])
            aligned.append(arr)
        return np.array(aligned)

    def compute_masks(self, aligned_data: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Computes two binary masks from the aligned data.
        
        - mask_low: selects entries with index less than threshold where data is nonzero.
        - mask_high: selects entries with index greater or equal to threshold (subject to row+column constraint).

        Parameters:
            aligned_data (np.ndarray): Array of shape (n, pad_length).

        Returns:
            tuple: (mask_low, mask_high) as boolean arrays.
        """
        n, m = aligned_data.shape
        mask_low = np.zeros((n, m), dtype=bool)
        mask_high = np.zeros((n, m), dtype=bool)
        for i in range(n):
            for j in range(m):
                if aligned_data[i, j] != 0:
                    if j < self.threshold:
                        mask_low[i, j] = True
                    elif (i + j) <= (m - 1):
                        mask_high[i, j] = True
        return mask_low, mask_high

    def compute_scaling_matrices(self, n: int, m: int) -> (np.ndarray, np.ndarray):
        """
        Computes two scaling matrices A1 and A2 of shape (n, m):
          - A1[i, j] = i / (n - 1)
          - A2[i, j] = j / (m - 1)

        Parameters:
            n (int): Number of rows.
            m (int): Number of columns.

        Returns:
            tuple: (A1, A2)
        """
        A1 = np.linspace(0, 1, n).reshape(n, 1) * np.ones((1, m))
        A2 = np.linspace(0, 1, m).reshape(1, m) * np.ones((n, 1))
        return A1, A2

    def prepare_features(self, reserrlog: list, otlog: list, accs: list, mask: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Prepares regression features and the target vector from experimental logs.

        The process:
          1. Align each input list to a fixed length.
          2. Zero out entries as specified by the mask.
          3. Extract nonzero indices from the OT log.
          4. Compute the logarithm of the error values for stability.
          5. Compute scaling matrices from the aligned data dimensions.
          6. Build a feature matrix with columns:
             [OT, scaling1, scaling2, scaling1 * OT, scaling2 * OT, (scaling2)^2 * OT, (scaling1)^2 * OT]
          7. The target vector is taken from the accuracy log.

        Parameters:
            reserrlog (list): List of error log arrays.
            otlog (list): List of OT log arrays.
            accs (list): List of accuracy arrays.
            mask (np.ndarray): Binary mask (same shape) to zero out selected entries.

        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector.
        """
        qs_res = self.align_data(reserrlog)
        qs_ot = self.align_data(otlog)
        qs_acc = self.align_data(accs)

        qs_res[mask] = 0
        qs_ot[mask] = 0
        qs_acc[mask] = 0

        nonzero_idx = np.nonzero(qs_ot)
        if nonzero_idx[0].size == 0:
            raise ValueError("No nonzero entries found in OT data after applying the mask.")

        res_nonzero = np.log(qs_res[nonzero_idx] + 1e-8)
        ot_nonzero = qs_ot[nonzero_idx]
        acc_nonzero = qs_acc[nonzero_idx]

        n_rows, n_cols = qs_res.shape
        A1, A2 = self.compute_scaling_matrices(n_rows, n_cols)
        scaling1 = A1[nonzero_idx]
        scaling2 = A2[nonzero_idx]

        num_points = ot_nonzero.shape[0]
        X = np.zeros((num_points, 7))
        X[:, 0] = ot_nonzero
        X[:, 1] = scaling1
        X[:, 2] = scaling2
        X[:, 3] = scaling1 * ot_nonzero
        X[:, 4] = scaling2 * ot_nonzero
        X[:, 5] = (scaling2 ** 2) * ot_nonzero
        X[:, 6] = (scaling1 ** 2) * ot_nonzero

        y = acc_nonzero.flatten()
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True, plot: bool = False) -> dict:
        """
        Fits a linear regression model to predict performance from the feature matrix X.

        Returns a dictionary with the fitted model and performance metrics (r2, mse, mae).

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            verbose (bool): If True, prints model parameters and metrics.
            plot (bool): If True, plots actual vs. predicted values.

        Returns:
            dict: Dictionary with keys 'model', 'r2', 'mse', and 'mae'.
        """
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        metrics = {
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred)
        }
        if verbose:
            print("Fitted model coefficients:", reg.coef_)
            print("Intercept:", reg.intercept_)
            print(f"R²: {metrics['r2']:.3f}, MSE: {metrics['mse']:.3f}, MAE: {metrics['mae']:.3f}")
        if plot:
            plt.figure(figsize=(6, 6))
            plt.scatter(y, y_pred, c='blue', alpha=0.7, label='Predicted')
            plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='Ideal')
            plt.xlabel("Actual Performance")
            plt.ylabel("Predicted Performance")
            plt.title("Projector: Actual vs Predicted")
            plt.legend()
            plt.show()
        self.model = reg
        return {'model': reg, **metrics}

    def predict(self, X_new: np.ndarray) -> float:
        """
        Predicts performance for a new feature vector X_new using the fitted model.

        Parameters:
            X_new (np.ndarray): A 1D array of features (or a 2D array with one row).

        Returns:
            float: The predicted performance.
        """
        if self.model is None:
            raise ValueError("No model has been fitted. Call fit() first.")
        X_new = X_new.reshape(1, -1)
        return self.model.predict(X_new)[0]

    @staticmethod
    def project_performance(n: float, n0: float, n1: float, perf0: float, perf1: float) -> float:
        """
        Extrapolates performance (e.g., test accuracy) to a new data size n,
        given measured performances at data sizes n0 and n1.

        Formula:
            performance_proj = (1 / ln(n1/n0)) * [ln(n/n0) * perf1 - ln(n/n1) * perf0]

        Parameters:
            n (float): Target data size.
            n0 (float): Baseline data size 0.
            n1 (float): Baseline data size 1.
            perf0 (float): Performance at n0.
            perf1 (float): Performance at n1.

        Returns:
            float: Extrapolated performance.
        """
        if n0 <= 0 or n1 <= 0 or n <= 0:
            raise ValueError("Data sizes must be positive.")
        return (1.0 / np.log(n1 / n0)) * (np.log(n / n0) * perf1 - np.log(n / n1) * perf0)


# -----------------------------------------
# Helper Function: Projection onto Simplex
# -----------------------------------------
def project_onto_simplex(v: np.ndarray) -> np.ndarray:
    """
    Projects a vector v onto the probability simplex (nonnegative entries summing to 1)
    using the algorithm of Duchi et al. (2008).

    Parameters:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: Projected vector.
    """
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)


# -----------------------------------------
# OptimalDataSourceSelector Class
# -----------------------------------------
class OptimalDataSourceSelector:
    def __init__(self, lr: float = 5e-4, iterations: int = 100, decay: float = 0.90):
        """
        Initialize the OptimalDataSourceSelector.

        Parameters:
            lr (float): Initial learning rate.
            iterations (int): Number of gradient descent iterations.
            decay (float): Exponential decay factor for the learning rate.
        """
        self.lr = lr
        self.iterations = iterations
        self.decay = decay

    def optimize(self, n: float, n0: float, n1: float,
                 data_n0: dict, data_n1: dict,
                 q_init: np.ndarray,
                 params0: list = None, params1: list = None) -> np.ndarray:
        """
        Optimizes mixing ratios q = [q_1, q_2, ..., q_k] for an arbitrary number (k) of data sources,
        using gradient descent with projection onto the simplex.

        For each source i, the gradient at each scale is computed as:
            g_i = (b2^i * q_i^2 + b1^i * q_i + b0^i) * calib_i + (2*b2^i*q_i + b1^i)*ot + (2*c2^i*q_i + c1^i),
        where calib_i is a calibrated gradient computed from the dual solutions.

        The combined gradient for source i is:
            grad_i = (1 / ln(n1/n0)) * [ln(n/n0)*g_i^(1) - ln(n/n1)*g_i^(0)].
        The mixing ratios are updated with a decayed learning rate and projected onto the simplex.

        Parameters:
            n (float): Target data size.
            n0 (float): Base data size for experiment 0.
            n1 (float): Base data size for experiment 1.
            data_n0 (dict): For experiment at n0. Must contain:
                'dual': concatenated dual solution vector (1D array),
                'ds_lengths': list or array of lengths for each source,
                'full_len': total number of samples,
                'ot_dist': OT distance (scalar).
            data_n1 (dict): Same as data_n0 but for experiment at n1.
            q_init (np.ndarray): Initial mixing ratios (shape (k,)), summing to 1.
            params0 (list): List of parameter dicts for experiment 0 (one per source). Each dict should have keys:
                'b2', 'b1', 'b0', 'c2', 'c1'. If None, defaults to zeros.
            params1 (list): List of parameter dicts for experiment 1 (same as params0).

        Returns:
            np.ndarray: Optimized mixing ratios of shape (k,).
        """
        k = q_init.shape[0]

        def default_params():
            return {'b2': 0.0, 'b1': 0.0, 'b0': 0.0, 'c2': 0.0, 'c1': 0.0}

        if params0 is None:
            params0 = [default_params() for _ in range(k)]
        if params1 is None:
            params1 = [default_params() for _ in range(k)]

        q = q_init.copy()

        # Unpack data dictionaries for experiment 0.
        dual0 = data_n0['dual']
        lengths0 = np.array(data_n0['ds_lengths'])
        full_len0 = data_n0['full_len']
        ot0 = data_n0.get('ot_dist', 0.0)

        # Unpack for experiment 1.
        dual1 = data_n1['dual']
        lengths1 = np.array(data_n1['ds_lengths'])
        full_len1 = data_n1['full_len']
        ot1 = data_n1.get('ot_dist', 0.0)

        # Function to compute calibrated gradient for a given source.
        def calib(dual, start, L, full_len):
            if L <= 0 or full_len - L <= 0:
                return 0.0
            part = np.sum(dual[start: start + L])
            return (part - (np.sum(dual) - part) * (L / (full_len - L))) / L

        # Compute calibration for each source.
        calib0 = []
        start = 0
        for L in lengths0:
            calib0.append(calib(dual0, start, L, full_len0))
            start += L
        calib1 = []
        start = 0
        for L in lengths1:
            calib1.append(calib(dual1, start, L, full_len1))
            start += L

        # Gradient function for source i.
        def grad_component(q_i, calib_i, ot, params):
            return (params['b2'] * q_i**2 + params['b1'] * q_i + params['b0']) * calib_i \
                   + (params['b2'] * 2 * q_i + params['b1']) * ot \
                   + (params['c2'] * 2 * q_i + params['c1'])

        # Gradient descent iterations.
        for it in range(self.iterations):
            step = self.lr * (self.decay ** it)
            grad = np.zeros_like(q)
            for i in range(k):
                g0 = grad_component(q[i], calib0[i], ot0, params0[i])
                g1 = grad_component(q[i], calib1[i], ot1, params1[i])
                grad[i] = (1.0 / np.log(n1 / n0)) * (np.log(n / n0) * g1 - np.log(n / n1) * g0)
            q = q + step * grad
            # Project the updated vector onto the simplex.
            q = project_onto_simplex(q)
        return q

