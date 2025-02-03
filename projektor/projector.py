# projektor/projector.py
"""
Module: projector.py

This module provides two classes:

Projector
---------
The Projector class encapsulates methods to process experimental logs,
generate pseudo–quadratic regression features, fit a linear regression model,
and make performance predictions. It also provides a method for extrapolating
performance to new data scales.

Example usage:

    >>> from projektor.projector import Projector
    >>> proj = Projector(pad_length=11, threshold=5)
    >>> reserrlog = [ [0.1, 0.2, 0.3], [0.15, 0.25] ]
    >>> otlog = [ [1.0, 1.2, 1.1], [0.9, 1.0] ]
    >>> accs = [ [0.75, 0.78, 0.80], [0.74, 0.76] ]
    >>> aligned = proj.align_data(reserrlog)
    >>> mask_low, mask_high = proj.compute_masks(aligned)
    >>> X, y = proj.prepare_features(reserrlog, otlog, accs, mask_high)
    >>> results = proj.fit(X, y, verbose=True, plot=False)
    >>> pred = proj.predict(X[0])
    >>> print("Predicted performance:", pred)

OptimalDataSourceSelector
---------------------------
The OptimalDataSourceSelector class implements a gradient-based optimizer to select
an optimal mixing ratio among an arbitrary number (k) of data sources.

Example usage:

    >>> from projektor.projector import OptimalDataSourceSelector, project_onto_simplex
    >>> import numpy as np
    >>> k = 3
    >>> q_init = np.array([0.33, 0.33, 0.34])
    >>> # data_n0 and data_n1 should be dictionaries with keys:
    >>> #   'dual': 1D array of dual solutions,
    >>> #   'ds_lengths': list of source lengths,
    >>> #   'full_len': total samples,
    >>> #   'ot_dist': scalar OT distance.
    >>> data_n0 = {'dual': np.random.rand(90), 'ds_lengths': [30, 30, 30], 'full_len': 90, 'ot_dist': 1.0}
    >>> data_n1 = {'dual': np.random.rand(90), 'ds_lengths': [30, 30, 30], 'full_len': 90, 'ot_dist': 1.2}
    >>> selector = OptimalDataSourceSelector(lr=5e-4, iterations=100, decay=0.90)
    >>> optimal_q = selector.optimize(n=10000, n0=1000, n1=1500,
    ...                               data_n0=data_n0, data_n1=data_n1,
    ...                               q_init=q_init)
    >>> print("Optimal mixing ratios:", optimal_q)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class Projector:
    def __init__(self, pad_length: int = 11, threshold: int = 5):
        """
        Initialize the Projector.

        Parameters:
            pad_length (int): Fixed length for each experimental log array.
            threshold (int): Threshold used in mask generation.
        """
        self.pad_length = pad_length
        self.threshold = threshold
        self.model = None

    def align_data(self, data_list: list) -> np.ndarray:
        """
        Aligns a list of 1-D arrays to a fixed length by padding with zeros.

        Parameters:
            data_list (list): List of numeric arrays or lists.

        Returns:
            np.ndarray: Array of shape (n, pad_length).
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

        mask_low:
            True for entries with index less than threshold.
        mask_high:
            True for entries with index greater than or equal to threshold and where row+col <= pad_length-1.

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
        Computes scaling matrices A1 and A2.

        A1[i, j] = i / (n - 1)
        A2[i, j] = j / (m - 1)

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
        Prepares regression features and target vector from experimental logs.

        Process:
          1. Align each log to a fixed length.
          2. Zero out entries indicated by the mask.
          3. Extract nonzero indices from the OT log.
          4. Compute the logarithm of error values.
          5. Compute scaling matrices.
          6. Assemble a feature matrix with 7 columns:
             [OT, scaling1, scaling2, scaling1*OT, scaling2*OT, (scaling2)^2*OT, (scaling1)^2*OT]
          7. The target is taken from the accuracy log.

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
            raise ValueError("No nonzero entries found in OT data after masking.")

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
        Fits a linear regression model to predict performance from features.

        Returns:
            dict: Contains the fitted model and performance metrics (r2, mse, mae).
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
            plt.scatter(y, y_pred, c='blue', alpha=0.7, label='Predictions')
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
        Predicts performance for a new feature vector using the fitted model.

        Parameters:
            X_new (np.ndarray): A 1D array (or a single-row 2D array) of features.

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
        Extrapolates performance to a new data scale n given performances at n0 and n1.

        Formula:
            performance_proj = (1 / ln(n1/n0)) * [ln(n/n0) * perf1 - ln(n/n1) * perf0]

        Parameters:
            n (float): Target data size.
            n0 (float): Baseline data size 0.
            n1 (float): Baseline data size 1.
            perf0 (float): Performance at n0.
            perf1 (float): Performance at n1.

        Returns:
            float: The extrapolated performance.
        """
        if n0 <= 0 or n1 <= 0 or n <= 0:
            raise ValueError("Data sizes must be positive.")
        return (1.0 / np.log(n1 / n0)) * (np.log(n / n0) * perf1 - np.log(n / n1) * perf0)

# -----------------------------------------
# Helper: Projection onto the Simplex
# -----------------------------------------
def project_onto_simplex(v: np.ndarray) -> np.ndarray:
    """
    Projects a vector v onto the probability simplex (nonnegative entries summing to 1)
    using the algorithm of Duchi et al. (2008).

    Parameters:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: The projected vector.
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
            decay (float): Learning rate decay factor.
        """
        self.lr = lr
        self.iterations = iterations
        self.decay = decay

    def optimize(self, n: float, n0: float, n1: float,
                 data_n0: dict, data_n1: dict,
                 q_init: np.ndarray,
                 params0: list = None, params1: list = None) -> np.ndarray:
        """
        Optimizes mixing ratios q = [q_1, ..., q_k] for k data sources via gradient descent
        with projection onto the probability simplex.

        For each data source i, a gradient is computed using dual solutions and OT distances
        from two experiments (at data sizes n0 and n1). The gradient for source i is:

            grad_i = (1/ln(n1/n0)) * [ ln(n/n0)*g_i^(1) - ln(n/n1)*g_i^(0) ],

        where g_i^(0) and g_i^(1) are computed with user-supplied parameters.

        Parameters:
            n (float): Target data size.
            n0 (float): Base data size for experiment 0.
            n1 (float): Base data size for experiment 1.
            data_n0 (dict): For experiment at n0; must contain:
                'dual': concatenated dual solution vector (1D array),
                'ds_lengths': list of lengths for each source,
                'full_len': total number of samples,
                'ot_dist': OT distance (scalar).
            data_n1 (dict): Same as data_n0 but for experiment at n1.
            q_init (np.ndarray): Initial mixing ratios (shape (k,)) that sum to 1.
            params0 (list): List of parameter dictionaries for experiment 0 (one per source), each with keys:
                           'b2', 'b1', 'b0', 'c2', 'c1'. Defaults to zeros if None.
            params1 (list): Same as params0 for experiment 1.

        Returns:
            np.ndarray: Optimized mixing ratios q of shape (k,).
        """
        k = q_init.shape[0]

        def default_params():
            return {'b2': 0.0, 'b1': 0.0, 'b0': 0.0, 'c2': 0.0, 'c1': 0.0}

        if params0 is None:
            params0 = [default_params() for _ in range(k)]
        if params1 is None:
            params1 = [default_params() for _ in range(k)]

        q = q_init.copy()

        # Unpack experiment 0 data.
        dual0 = data_n0['dual']
        lengths0 = np.array(data_n0['ds_lengths'])
        full_len0 = data_n0['full_len']
        ot0 = data_n0.get('ot_dist', 0.0)

        # Unpack experiment 1 data.
        dual1 = data_n1['dual']
        lengths1 = np.array(data_n1['ds_lengths'])
        full_len1 = data_n1['full_len']
        ot1 = data_n1.get('ot_dist', 0.0)

        # Compute calibrated gradient for each source.
        def calib(dual, start, L, full_len):
            if L <= 0 or full_len - L <= 0:
                return 0.0
            part = np.sum(dual[start: start + L])
            return (part - (np.sum(dual) - part) * (L / (full_len - L))) / L

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

        for it in range(self.iterations):
            step = self.lr * (self.decay ** it)
            grad = np.zeros_like(q)
            for i in range(k):
                g0 = grad_component(q[i], calib0[i], ot0, params0[i])
                g1 = grad_component(q[i], calib1[i], ot1, params1[i])
                grad[i] = (1.0 / np.log(n1 / n0)) * (np.log(n / n0) * g1 - np.log(n / n1) * g0)
            q = q + step * grad
            q = project_onto_simplex(q)
        return q

