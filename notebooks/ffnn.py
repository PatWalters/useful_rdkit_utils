import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class _MLP(nn.Module):
    """Internal multi-layer perceptron used by :class:`FFNNRegressor`.

    Builds a stack of ``Linear -> ReLU -> (optional Dropout)`` blocks followed
    by a single-output linear head. Not intended to be used directly.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_layers : sequence of int
        Width of each hidden layer, in order.
    dropout : float
        Dropout probability applied after each hidden activation. Set to 0
        to disable.
    """

    def __init__(self, n_features, hidden_layers, dropout):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
            Input batch.

        Returns
        -------
        torch.Tensor of shape (n_samples,)
            Predicted values with the trailing singleton dimension squeezed.
        """
        return self.net(x).squeeze(-1)


class FFNNRegressor(BaseEstimator, RegressorMixin):
    """Feed-forward neural network regressor for molecular descriptor data.

    A scikit-learn compatible wrapper around a PyTorch MLP. Optionally
    standardizes both features and target, trains with Adam + MSE loss, and
    supports early stopping on a held-out validation split.

    Suitable for QSAR / QSPR style problems where the inputs are a fixed-width
    matrix of molecular descriptors (RDKit, Mordred, fingerprint counts, etc.)
    and the target is a continuous activity value (pIC50, logP, ΔG, ...).

    Parameters
    ----------
    hidden_layers : tuple of int, default=(256, 128, 64)
        Widths of the hidden layers. The depth of the network equals
        ``len(hidden_layers)``.
    dropout : float, default=0.2
        Dropout probability applied after each hidden activation. Use a value
        in ``[0, 1)``; 0 disables dropout.
    learning_rate : float, default=1e-3
        Adam learning rate.
    batch_size : int, default=64
        Mini-batch size for SGD.
    epochs : int, default=200
        Maximum number of training epochs. Training may stop earlier if
        ``early_stopping_patience`` is triggered.
    weight_decay : float, default=1e-5
        L2 regularization coefficient passed to the Adam optimizer.
    early_stopping_patience : int, default=20
        Number of consecutive epochs without validation improvement before
        training is halted. Ignored when no validation split is used. Set to
        0 or None to disable.
    validation_fraction : float, default=0.1
        Fraction of the training data held out as a validation set for early
        stopping. Set to 0 or None to train on all data without validation.
    scale_features : bool, default=True
        If True, fit a :class:`~sklearn.preprocessing.StandardScaler` on the
        input matrix during ``fit`` and apply it during ``predict``.
    scale_target : bool, default=True
        If True, standardize the target during ``fit`` and invert the
        transformation in ``predict``. Helpful when activities span several
        orders of magnitude.
    device : str or torch.device, optional
        Device to run training and inference on (e.g. ``"cpu"``, ``"cuda"``,
        ``"mps"``). If None, the best available device is auto-selected.
    random_state : int, optional
        Seed for PyTorch and NumPy. Controls weight initialization, the
        train/validation split, and shuffling.
    verbose : bool, default=False
        If True, print per-epoch training (and validation) loss.

    Attributes
    ----------
    model_ : torch.nn.Module
        The trained PyTorch network.
    x_scaler_ : sklearn.preprocessing.StandardScaler
        Feature scaler. Present only when ``scale_features=True``.
    y_scaler_ : sklearn.preprocessing.StandardScaler
        Target scaler. Present only when ``scale_target=True``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    loss_history_ : list of float
        Mean training loss per epoch.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, n_features=20, random_state=0)
    >>> model = FFNNRegressor(hidden_layers=(64, 32), epochs=50, random_state=0)
    >>> _ = model.fit(X, y)
    >>> preds = model.predict(X[:5])
    """

    def __init__(
        self,
        hidden_layers=(256, 128, 64),
        dropout=0.2,
        learning_rate=1e-4,
        batch_size=64,
        epochs=200,
        weight_decay=1e-5,
        early_stopping_patience=20,
        validation_fraction=0.1,
        scale_features=True,
        scale_target=True,
        device=None,
        random_state=None,
        verbose=False,
    ):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.validation_fraction = validation_fraction
        self.scale_features = scale_features
        self.scale_target = scale_target
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _resolve_device(self):
        """Pick the torch device to use.

        Honors an explicit ``self.device`` if provided; otherwise chooses
        CUDA, then Apple MPS, then CPU.

        Returns
        -------
        torch.device
        """
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(self, X, y):
        """Train the network on a descriptor matrix and activity vector.

        Validates the inputs, optionally standardizes ``X`` and ``y``, splits
        off a validation set for early stopping, and trains the MLP with Adam
        and MSE loss. When early stopping triggers (or training completes),
        the weights from the best-performing epoch on the validation set are
        restored.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Molecular descriptor matrix. Must be 2D and numeric.
        y : array-like of shape (n_samples,)
            Continuous activity values (e.g. pIC50).

        Returns
        -------
        self : FFNNRegressor
            The fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        y = y.astype(np.float32)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self._device = self._resolve_device()
        self.n_features_in_ = X.shape[1]

        if self.scale_features:
            self.x_scaler_ = StandardScaler().fit(X)
            X_proc = np.clip(self.x_scaler_.transform(X), -10.0, 10.0)
        else:
            X_proc = X

        if self.scale_target:
            self.y_scaler_ = StandardScaler().fit(y.reshape(-1, 1))
            y_proc = self.y_scaler_.transform(y.reshape(-1, 1)).ravel()
        else:
            y_proc = y

        n = X_proc.shape[0]
        use_val = self.validation_fraction and 0 < self.validation_fraction < 1 and n > 10
        if use_val:
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(n)
            n_val = max(1, int(n * self.validation_fraction))
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
            X_tr, y_tr = X_proc[tr_idx], y_proc[tr_idx]
            X_val, y_val = X_proc[val_idx], y_proc[val_idx]
        else:
            X_tr, y_tr = X_proc, y_proc
            X_val = y_val = None

        train_ds = TensorDataset(
            torch.from_numpy(X_tr.astype(np.float32)),
            torch.from_numpy(y_tr.astype(np.float32)),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        self.model_ = _MLP(self.n_features_in_, list(self.hidden_layers), self.dropout).to(self._device)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        patience = 0
        self.loss_history_ = []

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_ds)
            self.loss_history_.append(epoch_loss)

            if use_val:
                self.model_.eval()
                with torch.no_grad():
                    xv = torch.from_numpy(X_val.astype(np.float32)).to(self._device)
                    yv = torch.from_numpy(y_val.astype(np.float32)).to(self._device)
                    val_loss = loss_fn(self.model_(xv), yv).item()

                if self.verbose:
                    print(f"epoch {epoch + 1:4d}  train={epoch_loss:.4f}  val={val_loss:.4f}")

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if self.early_stopping_patience and patience >= self.early_stopping_patience:
                        if self.verbose:
                            print(f"early stopping at epoch {epoch + 1}")
                        break
            elif self.verbose:
                print(f"epoch {epoch + 1:4d}  train={epoch_loss:.4f}")

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(self, X):
        """Predict activity values for new molecules.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Descriptor matrix. Must have the same number of columns as the
            matrix used during ``fit``.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted activity values, returned on the original (unscaled)
            scale of ``y``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If ``fit`` has not been called.
        ValueError
            If the number of features in ``X`` does not match the training
            matrix.
        """
        check_is_fitted(self, "model_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        X_proc = np.clip(self.x_scaler_.transform(X), -10.0, 10.0) if self.scale_features else X
        self.model_.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X_proc.astype(np.float32)).to(self._device)
            y_pred = self.model_(xt).cpu().numpy()
        if self.scale_target:
            y_pred = self.y_scaler_.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return y_pred


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    X, y = make_regression(n_samples=500, n_features=50, noise=10.0, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = FFNNRegressor(
        hidden_layers=(128, 64),
        epochs=300,
        random_state=42,
        verbose=False,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"R²   = {r2_score(y_te, pred):.3f}")
    print(f"RMSE = {np.sqrt(mean_squared_error(y_te, pred)):.3f}")
