"""
Flood Susceptibility Framework — Model Implementations
======================================================

M01-M05: Traditional machine learning
M06-M10: Hybrid machine learning

Legacy get_models() API is preserved for 03_train_models.py.
"""

import warnings
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    XGBClassifier = None
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    LGBMClassifier = None
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    CatBoostClassifier = None
    HAS_CAT = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


for name, available in [
    ("xgboost", HAS_XGB),
    ("lightgbm", HAS_LGBM),
    ("catboost", HAS_CAT),
    ("torch", HAS_TORCH),
]:
    if not available:
        print(f"[models] {name} is not installed.")


# ============================================================
# M01 — LOGISTIC REGRESSION
# ============================================================

def make_lr(seed=SEED):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=seed,
        ),
    )


# ============================================================
# M02 — RANDOM FOREST
# ============================================================

def make_rf(seed=SEED):
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )


# ============================================================
# M03 — XGBOOST
# ============================================================

def make_xgb(seed=SEED, **params):
    defaults = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
    }

    defaults.update(params)

    if HAS_XGB:
        return XGBClassifier(
            **defaults,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )

    warnings.warn(
        "XGBoost unavailable; using HistGradientBoostingClassifier "
        "as a fallback. Record this substitution in experiment results."
    )

    return HistGradientBoostingClassifier(
        max_iter=defaults["n_estimators"],
        max_depth=defaults["max_depth"],
        learning_rate=defaults["learning_rate"],
        max_features=defaults["colsample_bytree"],
        min_samples_leaf=max(
            10,
            int(defaults["min_child_weight"]) * 10,
        ),
        random_state=seed,
    )


# ============================================================
# M04 — LIGHTGBM
# ============================================================

def make_lgbm(seed=SEED):
    if HAS_LGBM:
        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )

    warnings.warn(
        "LightGBM unavailable; using HistGradientBoostingClassifier "
        "as a fallback."
    )

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        random_state=seed,
    )


# ============================================================
# M05 — MULTILAYER PERCEPTRON
# ============================================================

def make_mlp(seed=SEED):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            max_iter=300,
            random_state=seed,
        ),
    )


# ============================================================
# M06 — STACKING ENSEMBLE
# ============================================================

def make_stacking(seed=SEED):
    estimators = [
        ("rf", make_rf(seed)),
        ("xgb", make_xgb(seed)),
        ("lgbm", make_lgbm(seed)),
    ]

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=2000,
            random_state=seed,
        ),
        stack_method="predict_proba",
        cv=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        ),
        n_jobs=1,
    )


# ============================================================
# M07 — SOFT-VOTING ENSEMBLE
# ============================================================

def make_soft_voting(seed=SEED):
    return VotingClassifier(
        estimators=[
            ("rf", make_rf(seed)),
            ("xgb", make_xgb(seed)),
            ("mlp", make_mlp(seed)),
        ],
        voting="soft",
        n_jobs=1,
    )


# ============================================================
# M08 — GENETIC ALGORITHM + XGBOOST
# ============================================================

class GAXGB(ClassifierMixin, BaseEstimator):
    """Search XGBoost hyperparameters using a genetic algorithm."""

    SPACE = [
        ("n_estimators", 100, 600, int),
        ("max_depth", 3, 10, int),
        ("learning_rate", 0.01, 0.30, float),
        ("subsample", 0.50, 1.00, float),
        ("colsample_bytree", 0.50, 1.00, float),
        ("min_child_weight", 1, 10, int),
    ]

    def __init__(
        self,
        pop=10,
        gens=6,
        cv=3,
        sample=8000,
        seed=SEED,
    ):
        self.pop = pop
        self.gens = gens
        self.cv = cv
        self.sample = sample
        self.seed = seed

    def _decode(self, genome):
        result = {}

        for value, (name, low, high, kind) in zip(
            genome,
            self.SPACE,
        ):
            result[name] = kind(
                low + value * (high - low)
            )

        return result

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)

        if len(np.unique(y)) != 2:
            raise ValueError(
                "GAXGB requires both classes."
            )

        rng = np.random.default_rng(self.seed)

        # Stratified subset for hyperparameter search.
        if self.sample and len(y) > self.sample:
            from sklearn.model_selection import train_test_split

            indices = np.arange(len(y))

            selected, _ = train_test_split(
                indices,
                train_size=self.sample,
                stratify=y,
                random_state=self.seed,
            )

            X_search = X[selected]
            y_search = y[selected]
        else:
            X_search = X
            y_search = y

        smallest_class = int(
            np.bincount(y_search, minlength=2).min()
        )

        if smallest_class < 2:
            raise ValueError(
                "Too few samples of one class for GA cross-validation."
            )

        splitter = StratifiedKFold(
            n_splits=min(self.cv, smallest_class),
            shuffle=True,
            random_state=self.seed,
        )

        def fitness(genome):
            estimator = make_xgb(
                seed=self.seed,
                **self._decode(genome),
            )

            scores = cross_val_score(
                estimator,
                X_search,
                y_search,
                cv=splitter,
                scoring="roc_auc",
                n_jobs=1,
                error_score=np.nan,
            )

            scores = scores[np.isfinite(scores)]

            return float(scores.mean()) if len(scores) else -np.inf

        dimension = len(self.SPACE)

        population_size = max(4, int(self.pop))

        population = rng.random(
            (population_size, dimension)
        )

        scores = np.array([
            fitness(genome)
            for genome in population
        ])

        self.history_ = []

        for generation in range(self.gens):
            order = np.argsort(-scores)

            population = population[order]
            scores = scores[order]

            self.history_.append(float(scores[0]))

            print(
                f"    GA generation {generation + 1}/"
                f"{self.gens}: search AUC={scores[0]:.4f}"
            )

            children = []

            while len(children) < population_size - 2:
                first = population[
                    int(rng.integers(0, max(2, population_size // 2)))
                ]

                second = population[
                    int(rng.integers(0, max(2, population_size // 2)))
                ]

                child = np.where(
                    rng.random(dimension) < 0.5,
                    first,
                    second,
                )

                mutation_mask = rng.random(dimension) < 0.2

                child = np.clip(
                    child + mutation_mask * rng.normal(
                        0,
                        0.15,
                        dimension,
                    ),
                    0,
                    1,
                )

                children.append(child)

            children = np.asarray(children)

            population = np.vstack([
                population[:2],
                children,
            ])

            scores = np.concatenate([
                scores[:2],
                np.array([
                    fitness(child)
                    for child in children
                ]),
            ])

        best_index = int(np.argmax(scores))

        self.best_params_ = self._decode(
            population[best_index]
        )

        self.best_search_auc_ = float(scores[best_index])

        # Final estimator is trained on the complete training fold.
        self.model_ = make_xgb(
            seed=self.seed,
            **self.best_params_,
        )

        self.model_.fit(X, y)

        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]

        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(
            np.asarray(X)
        )

    def predict(self, X):
        return self.model_.predict(
            np.asarray(X)
        )


# ============================================================
# M09 — RF FEATURE SELECTION + LIGHTGBM
# ============================================================

def make_rf_feature_selection_lgbm(seed=SEED):
    return Pipeline([
        (
            "selector",
            SelectFromModel(
                RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=seed,
                ),
                threshold="median",
            ),
        ),
        (
            "classifier",
            make_lgbm(seed),
        ),
    ])


# ============================================================
# M10 — CATBOOST STACKING
# ============================================================

def make_catboost_stacking(seed=SEED):
    if not HAS_CAT:
        raise ImportError(
            "M10 requires CatBoost. Install it with "
            "'python -m pip install catboost'."
        )

    return StackingClassifier(
        estimators=[
            ("rf", make_rf(seed)),
            (
                "catboost",
                CatBoostClassifier(
                    iterations=300,
                    depth=6,
                    learning_rate=0.05,
                    verbose=False,
                    random_seed=seed,
                    allow_writing_files=False,
                ),
            ),
            ("lgbm", make_lgbm(seed)),
        ],
        final_estimator=LogisticRegression(
            max_iter=2000,
            random_state=seed,
        ),
        stack_method="predict_proba",
        cv=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        ),
        n_jobs=1,
    )


# ============================================================
# LEGACY MLP EMBEDDING + XGBOOST
# ============================================================

class MLPEmbedXGB(ClassifierMixin, BaseEstimator):
    """
    Learn hidden-layer features with an MLP, then classify the
    original features and learned representation with XGBoost.

    This implementation is retained for compatibility with the
    original hybrid model. Out-of-fold embeddings would be preferable
    for a more rigorous final experiment.
    """

    def __init__(self, hidden=(64, 32), seed=SEED):
        self.hidden = hidden
        self.seed = seed

    def _embed(self, X_scaled):
        activation = X_scaled

        for weights, bias in zip(
            self.mlp_.coefs_[:-1],
            self.mlp_.intercepts_[:-1],
        ):
            activation = np.maximum(
                activation @ weights + bias,
                0,
            )

        return activation

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)

        self.scaler_ = StandardScaler()

        X_scaled = self.scaler_.fit_transform(X)

        self.mlp_ = MLPClassifier(
            hidden_layer_sizes=self.hidden,
            early_stopping=True,
            max_iter=300,
            random_state=self.seed,
        )

        self.mlp_.fit(X_scaled, y)

        embedding = self._embed(X_scaled)

        combined = np.hstack([
            X_scaled,
            embedding,
        ])

        self.gbm_ = make_xgb(
            seed=self.seed
        )

        self.gbm_.fit(combined, y)

        self.classes_ = self.gbm_.classes_
        self.n_features_in_ = X.shape[1]

        return self

    def _transform(self, X):
        X_scaled = self.scaler_.transform(
            np.asarray(X, dtype=np.float32)
        )

        return np.hstack([
            X_scaled,
            self._embed(X_scaled),
        ])

    def predict_proba(self, X):
        return self.gbm_.predict_proba(
            self._transform(X)
        )

    def predict(self, X):
        return self.gbm_.predict(
            self._transform(X)
        )


# ============================================================
# LEGACY CNN-BILSTM
# ============================================================

if HAS_TORCH:

    class _Net(nn.Module):
        def __init__(self, c=32, h=32):
            super().__init__()

            self.conv = nn.Sequential(
                nn.Conv1d(1, c, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(c, c, 3, padding=1),
                nn.ReLU(),
            )

            self.lstm = nn.LSTM(
                c,
                h,
                batch_first=True,
                bidirectional=True,
            )

            self.head = nn.Sequential(
                nn.Linear(2 * h, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 1),
            )

        def forward(self, X):
            X = self.conv(X.unsqueeze(1))

            output, _ = self.lstm(
                X.permute(0, 2, 1)
            )

            return self.head(
                output.mean(1)
            ).squeeze(-1)


    class CNNBiLSTM(ClassifierMixin, BaseEstimator):
        """
        Experimental CNN-BiLSTM classifier.

        This consumes an ordered static feature vector. It is not a
        valid temporal forecasting model without genuine time series.
        """

        def __init__(
            self,
            epochs=40,
            batch=256,
            lr=1e-3,
            patience=6,
            seed=SEED,
        ):
            self.epochs = epochs
            self.batch = batch
            self.lr = lr
            self.patience = patience
            self.seed = seed

        def fit(self, X, y):
            torch.manual_seed(self.seed)

            rng = np.random.default_rng(self.seed)

            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)

            if len(np.unique(y)) != 2:
                raise ValueError(
                    "CNNBiLSTM requires both classes."
                )

            self.scaler_ = StandardScaler()

            X_scaled = self.scaler_.fit_transform(
                X
            ).astype(np.float32)

            indices = rng.permutation(len(y))

            n_val = max(
                1,
                int(0.1 * len(y)),
            )

            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            device = "cpu"

            self.dev_ = device

            net = _Net().to(device)

            optimizer = torch.optim.Adam(
                net.parameters(),
                lr=self.lr,
            )

            loss_function = nn.BCEWithLogitsLoss()

            X_train = torch.tensor(
                X_scaled[train_idx]
            ).to(device)

            y_train = torch.tensor(
                y[train_idx]
            ).to(device)

            X_val = torch.tensor(
                X_scaled[val_idx]
            ).to(device)

            y_val = torch.tensor(
                y[val_idx]
            ).to(device)

            best_loss = float("inf")
            best_state = None
            bad_epochs = 0

            self.history_ = []

            for epoch in range(self.epochs):
                net.train()

                permutation = torch.randperm(
                    len(train_idx),
                    device=device,
                )

                for start in range(
                    0,
                    len(train_idx),
                    self.batch,
                ):
                    batch_indices = permutation[
                        start:start + self.batch
                    ]

                    optimizer.zero_grad(set_to_none=True)

                    logits = net(
                        X_train[batch_indices]
                    )

                    loss = loss_function(
                        logits,
                        y_train[batch_indices],
                    )

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(),
                        max_norm=1.0,
                    )

                    optimizer.step()

                net.eval()

                with torch.no_grad():
                    val_loss = loss_function(
                        net(X_val),
                        y_val,
                    ).item()

                self.history_.append({
                    "epoch": epoch + 1,
                    "validation_loss": val_loss,
                })

                if val_loss < best_loss - 1e-4:
                    best_loss = val_loss
                    bad_epochs = 0

                    best_state = {
                        key: value.detach().clone()
                        for key, value in net.state_dict().items()
                    }

                else:
                    bad_epochs += 1

                    if bad_epochs >= self.patience:
                        break

            if best_state is not None:
                net.load_state_dict(best_state)

            self.net_ = net.eval()
            self.classes_ = np.array([0, 1])
            self.n_features_in_ = X.shape[1]

            return self

        def predict_proba(self, X):
            X_scaled = self.scaler_.transform(
                np.asarray(X, dtype=np.float32)
            ).astype(np.float32)

            tensor = torch.tensor(
                X_scaled,
                dtype=torch.float32,
            ).to(self.dev_)

            self.net_.eval()

            with torch.no_grad():
                probabilities = torch.sigmoid(
                    self.net_(tensor)
                ).cpu().numpy()

            return np.column_stack([
                1 - probabilities,
                probabilities,
            ])

        def predict(self, X):
            return (
                self.predict_proba(X)[:, 1] >= 0.5
            ).astype(int)

else:

    class CNNBiLSTM(ClassifierMixin, BaseEstimator):
        def __init__(
            self,
            epochs=40,
            batch=256,
            lr=1e-3,
            patience=6,
            seed=SEED,
        ):
            self.epochs = epochs
            self.batch = batch
            self.lr = lr
            self.patience = patience
            self.seed = seed

        def fit(self, X, y):
            raise ImportError(
                "CNNBiLSTM requires PyTorch."
            )



# ============================================================
# M12-M14 — TABULAR DEEP LEARNING
# ============================================================

if HAS_TORCH:

    class _TabularTransformer(nn.Module):
        def __init__(self, n_features, d_model=32, nhead=4):
            super().__init__()

            self.weight = nn.Parameter(
                torch.randn(n_features, d_model) * 0.02
            )
            self.bias = nn.Parameter(
                torch.zeros(n_features, d_model)
            )

            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=64,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )

            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=2,
            )

            self.projection = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.GELU(),
                nn.Dropout(0.1),
            )

            self.head = nn.Linear(32, 1)

        def encode(self, X):
            tokens = (
                X.unsqueeze(-1) * self.weight.unsqueeze(0)
                + self.bias.unsqueeze(0)
            )
            encoded = self.encoder(tokens)
            pooled = encoded.mean(dim=1)
            return self.projection(pooled)

        def forward(self, X):
            return self.head(self.encode(X)).squeeze(-1)


    class _Autoencoder(nn.Module):
        def __init__(self, n_features, latent_dim=16):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim),
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, n_features),
            )

        def forward(self, X):
            return self.decoder(self.encoder(X))


    class TorchTabularClassifier(ClassifierMixin, BaseEstimator):
        """Fold-local autoencoder/transformer feature learning."""

        def __init__(
            self,
            architecture="autoencoder_xgb",
            epochs=20,
            batch=256,
            lr=1e-3,
            latent_dim=16,
            d_model=32,
            seed=SEED,
        ):
            self.architecture = architecture
            self.epochs = epochs
            self.batch = batch
            self.lr = lr
            self.latent_dim = latent_dim
            self.d_model = d_model
            self.seed = seed

        def fit(self, X, y):
            if not HAS_XGB:
                raise ImportError("M12/M13 require XGBoost.")
            if not HAS_TORCH:
                raise ImportError("M12-M14 require PyTorch.")

            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)

            if len(np.unique(y)) != 2:
                raise ValueError("Both target classes are required.")

            self.scaler_ = StandardScaler()
            Xs = self.scaler_.fit_transform(X).astype(np.float32)

            if torch.cuda.is_available():
                self.device_ = "cuda"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self.device_ = "mps"
            else:
                self.device_ = "cpu"

            device = self.device_
            Xt = torch.tensor(Xs, dtype=torch.float32, device=device)
            yt = torch.tensor(y, dtype=torch.float32, device=device)

            if self.architecture == "autoencoder_xgb":
                net = _Autoencoder(
                    X.shape[1], self.latent_dim
                ).to(device)

                optimizer = torch.optim.Adam(
                    net.parameters(), lr=self.lr
                )
                loss_fn = nn.MSELoss()

                for _ in range(self.epochs):
                    net.train()
                    order = torch.randperm(len(Xt), device=device)

                    for start in range(0, len(Xt), self.batch):
                        idx = order[start:start + self.batch]
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_fn(net(Xt[idx]), Xt[idx])
                        loss.backward()
                        optimizer.step()

                net.eval()
                with torch.no_grad():
                    embedding = net.encoder(Xt).cpu().numpy()

                self.net_ = net
                features = np.hstack([Xs, embedding])

                self.gbm_ = make_xgb(seed=self.seed)
                self.gbm_.fit(features, y.astype(int))
                self.classes_ = self.gbm_.classes_

            elif self.architecture == "transformer_xgb":
                net = _TabularTransformer(
                    X.shape[1], self.d_model
                ).to(device)

                optimizer = torch.optim.AdamW(
                    net.parameters(), lr=self.lr
                )
                loss_fn = nn.BCEWithLogitsLoss()

                for _ in range(self.epochs):
                    net.train()
                    order = torch.randperm(len(Xt), device=device)

                    for start in range(0, len(Xt), self.batch):
                        idx = order[start:start + self.batch]
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_fn(net(Xt[idx]), yt[idx])
                        loss.backward()
                        optimizer.step()

                net.eval()
                with torch.no_grad():
                    embedding = net.encode(Xt).cpu().numpy()

                self.net_ = net
                features = np.hstack([Xs, embedding])

                self.gbm_ = make_xgb(seed=self.seed)
                self.gbm_.fit(features, y.astype(int))
                self.classes_ = self.gbm_.classes_

            elif self.architecture == "ft_transformer":
                net = _TabularTransformer(
                    X.shape[1], self.d_model
                ).to(device)

                optimizer = torch.optim.AdamW(
                    net.parameters(), lr=self.lr
                )
                loss_fn = nn.BCEWithLogitsLoss()

                for _ in range(self.epochs):
                    net.train()
                    order = torch.randperm(len(Xt), device=device)

                    for start in range(0, len(Xt), self.batch):
                        idx = order[start:start + self.batch]
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_fn(net(Xt[idx]), yt[idx])
                        loss.backward()
                        optimizer.step()

                self.net_ = net.eval()
                self.classes_ = np.array([0, 1])

            else:
                raise ValueError(
                    f"Unknown architecture: {self.architecture}"
                )

            self.n_features_in_ = X.shape[1]
            return self

        def predict_proba(self, X):
            Xs = self.scaler_.transform(
                np.asarray(X, dtype=np.float32)
            ).astype(np.float32)

            if self.architecture == "ft_transformer":
                Xt = torch.tensor(
                    Xs, dtype=torch.float32, device=self.device_
                )
                self.net_.eval()
                with torch.no_grad():
                    logits = self.net_(Xt)
                    p = torch.sigmoid(logits).cpu().numpy()

                return np.column_stack([1 - p, p])

            Xt = torch.tensor(
                Xs, dtype=torch.float32, device=self.device_
            )
            self.net_.eval()

            with torch.no_grad():
                if self.architecture == "autoencoder_xgb":
                    embedding = self.net_.encoder(Xt).cpu().numpy()
                else:
                    embedding = self.net_.encode(Xt).cpu().numpy()

            return self.gbm_.predict_proba(
                np.hstack([Xs, embedding])
            )

        def predict(self, X):
            return (
                self.predict_proba(X)[:, 1] >= 0.5
            ).astype(int)


else:

    class TorchTabularClassifier(ClassifierMixin, BaseEstimator):
        def __init__(
            self,
            architecture="autoencoder_xgb",
            epochs=20,
            batch=256,
            lr=1e-3,
            latent_dim=16,
            d_model=32,
            seed=SEED,
        ):
            self.architecture = architecture
            self.epochs = epochs
            self.batch = batch
            self.lr = lr
            self.latent_dim = latent_dim
            self.d_model = d_model
            self.seed = seed

        def fit(self, X, y):
            raise ImportError("M12-M14 require PyTorch.")


def make_autoencoder_xgb(seed=SEED, **kwargs):
    return TorchTabularClassifier(
        architecture="autoencoder_xgb",
        seed=seed,
        **kwargs,
    )


def make_transformer_xgb(seed=SEED, **kwargs):
    return TorchTabularClassifier(
        architecture="transformer_xgb",
        seed=seed,
        **kwargs,
    )


def make_ft_transformer(seed=SEED, **kwargs):
    return TorchTabularClassifier(
        architecture="ft_transformer",
        seed=seed,
        **kwargs,
    )



# ============================================================
# M15-M19 — CNN, DNN ENSEMBLE AND SPATIAL GRAPH MODELS
# ============================================================

from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors


class DNNEnsembleClassifier(ClassifierMixin, BaseEstimator):
    """Average predictions from independently initialized MLPs."""

    def __init__(self, n_members=5, max_iter=200, seed=SEED):
        self.n_members = n_members
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.imputer_ = SimpleImputer(strategy="median")
        X = self.imputer_.fit_transform(X)
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X)

        self.members_ = []
        for i in range(self.n_members):
            model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                early_stopping=True,
                max_iter=self.max_iter,
                random_state=self.seed + i,
            )
            model.fit(X, y)
            self.members_.append(model)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        X = self.scaler_.transform(
            self.imputer_.transform(np.asarray(X, dtype=float))
        )
        return np.mean(
            [model.predict_proba(X) for model in self.members_],
            axis=0,
        )

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


if HAS_TORCH:

    class _CNNAttentionNet(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.attention = nn.Conv1d(32, 1, kernel_size=1)
            self.head = nn.Sequential(
                nn.Linear(32, 32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            z = self.conv(x.unsqueeze(1))
            weights = torch.softmax(self.attention(z), dim=2)
            pooled = (z * weights).sum(dim=2)
            return self.head(pooled).squeeze(-1)


    class CNNAttentionClassifier(ClassifierMixin, BaseEstimator):
        """Experimental CNN-attention model for ordered tabular features."""

        def __init__(self, epochs=15, batch=256, lr=1e-3, seed=SEED):
            self.epochs = epochs
            self.batch = batch
            self.lr = lr
            self.seed = seed

        def fit(self, X, y):
            torch.manual_seed(self.seed)
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)

            self.imputer_ = SimpleImputer(strategy="median")
            X = self.imputer_.fit_transform(X).astype(np.float32)
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X).astype(np.float32)

            if torch.cuda.is_available():
                self.device_ = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device_ = "mps"
            else:
                self.device_ = "cpu"

            device = self.device_
            xt = torch.tensor(X, dtype=torch.float32, device=device)
            yt = torch.tensor(y, dtype=torch.float32, device=device)

            self.net_ = _CNNAttentionNet(X.shape[1]).to(device)
            optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr)
            loss_fn = nn.BCEWithLogitsLoss()

            rng = np.random.default_rng(self.seed)
            for _ in range(self.epochs):
                order = rng.permutation(len(X))
                self.net_.train()
                for start in range(0, len(X), self.batch):
                    idx = order[start:start + self.batch]
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(self.net_(xt[idx]), yt[idx])
                    loss.backward()
                    optimizer.step()

            self.net_.eval()
            self.classes_ = np.array([0, 1])
            self.n_features_in_ = X.shape[1]
            return self

        def predict_proba(self, X):
            X = self.scaler_.transform(
                self.imputer_.transform(np.asarray(X, dtype=np.float32))
            ).astype(np.float32)
            xt = torch.tensor(X, dtype=torch.float32, device=self.device_)
            self.net_.eval()
            with torch.no_grad():
                p = torch.sigmoid(self.net_(xt)).cpu().numpy()
            return np.column_stack([1 - p, p])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


    class _GraphNet(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(2 * n_features, 64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            self.head = nn.Linear(32, 1)

        def forward(self, own, neighbour):
            z = self.encoder(torch.cat([own, neighbour], dim=1))
            return self.head(z).squeeze(-1)


    class SpatialGraphClassifier(ClassifierMixin, BaseEstimator):
        """
        One-hop spatial graph model.

        Input columns must be environmental predictors followed by lon, lat.
        Graphs are built from coordinates; labels are used only for training
        nodes. Held-out predictions aggregate predictor features from training
        neighbours only.
        """

        def __init__(
            self, architecture="gnn_xgb", k=8, epochs=12,
            batch=512, lr=1e-3, seed=SEED
        ):
            self.architecture = architecture
            self.k = k
            self.epochs = epochs
            self.batch = batch
            self.lr = lr
            self.seed = seed

        @staticmethod
        def _dist(a, b):
            lat0 = np.deg2rad(np.mean(a[:, 1]))
            ax = a[:, 0] * np.cos(lat0)
            bx = b[:, 0] * np.cos(lat0)
            return (ax[:, None] - bx[None, :]) ** 2 + (
                a[:, 1, None] - b[None, :, 1]
            ) ** 2

        def _aggregate(self, own, coords, reference, ref_coords, training):
            distances = self._dist(coords, ref_coords)
            if training:
                np.fill_diagonal(distances, np.inf)

            k = min(self.k, len(ref_coords) - (1 if training else 0))
            if k < 1:
                return np.zeros_like(own)

            neighbours = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
            output = np.empty_like(own)

            for i, ids in enumerate(neighbours):
                values = reference[ids]
                if self.architecture == "gat":
                    # Feature-similarity attention over geographic neighbours.
                    q = own[i]
                    scores = values @ q / np.sqrt(max(1, own.shape[1]))
                    scores = scores - scores.max()
                    weights = np.exp(np.clip(scores, -30, 30))
                    weights /= weights.sum()
                    output[i] = weights @ values
                elif self.architecture == "sage_xgb":
                    # GraphSAGE-style max-pooling aggregation.
                    output[i] = values.max(axis=0)
                else:
                    output[i] = values.mean(axis=0)
            return output

        def fit(self, X, y):
            if not HAS_TORCH:
                raise ImportError("Spatial graph models require PyTorch.")
            if self.architecture in {"gnn_xgb", "sage_xgb"} and not HAS_XGB:
                raise ImportError("M17 and M19 require XGBoost.")

            torch.manual_seed(self.seed)
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=int)
            if X.shape[1] < 3:
                raise ValueError("Graph models require predictors plus lon and lat.")

            raw_features = X[:, :-2]
            self.train_coords_ = X[:, -2:]
            self.imputer_ = SimpleImputer(strategy="median")
            raw_features = self.imputer_.fit_transform(raw_features)
            self.scaler_ = StandardScaler()
            self.train_features_ = self.scaler_.fit_transform(raw_features).astype(np.float32)

            neighbour = self._aggregate(
                self.train_features_, self.train_coords_,
                self.train_features_, self.train_coords_, True
            )

            if torch.cuda.is_available():
                self.device_ = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device_ = "mps"
            else:
                self.device_ = "cpu"

            device = self.device_
            own = torch.tensor(self.train_features_, dtype=torch.float32, device=device)
            agg = torch.tensor(neighbour, dtype=torch.float32, device=device)
            target = torch.tensor(y, dtype=torch.float32, device=device)

            self.net_ = _GraphNet(self.train_features_.shape[1]).to(device)
            optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr)
            loss_fn = nn.BCEWithLogitsLoss()
            rng = np.random.default_rng(self.seed)

            for _ in range(self.epochs):
                order = rng.permutation(len(y))
                self.net_.train()
                for start in range(0, len(y), self.batch):
                    ids = order[start:start + self.batch]
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.net_(own[ids], agg[ids])
                    loss = loss_fn(logits, target[ids])
                    loss.backward()
                    optimizer.step()

            self.net_.eval()
            with torch.no_grad():
                train_embedding = self.net_.encoder(
                    torch.cat([own, agg], dim=1)
                ).cpu().numpy()

            if self.architecture in {"gnn_xgb", "sage_xgb"}:
                self.gbm_ = make_xgb(seed=self.seed)
                self.gbm_.fit(
                    np.hstack([self.train_features_, train_embedding]), y
                )
                self.classes_ = self.gbm_.classes_
            else:
                self.classes_ = np.array([0, 1])

            self.n_features_in_ = X.shape[1]
            return self

        def predict_proba(self, X):
            X = np.asarray(X, dtype=np.float32)
            raw_features = self.imputer_.transform(X[:, :-2])
            features = self.scaler_.transform(raw_features).astype(np.float32)
            coords = X[:, -2:]

            agg = self._aggregate(
                features, coords,
                self.train_features_, self.train_coords_, False
            )

            own_t = torch.tensor(features, dtype=torch.float32, device=self.device_)
            agg_t = torch.tensor(agg, dtype=torch.float32, device=self.device_)
            self.net_.eval()
            with torch.no_grad():
                if self.architecture == "gat":
                    logits = self.net_(own_t, agg_t)
                    p = torch.sigmoid(logits).cpu().numpy()
                    return np.column_stack([1 - p, p])

                embedding = self.net_.encoder(
                    torch.cat([own_t, agg_t], dim=1)
                ).cpu().numpy()

            return self.gbm_.predict_proba(np.hstack([features, embedding]))

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


else:

    class CNNAttentionClassifier(ClassifierMixin, BaseEstimator):
        def __init__(self, epochs=15, batch=256, lr=1e-3, seed=SEED):
            self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed
        def fit(self, X, y):
            raise ImportError("M15 requires PyTorch.")

    class SpatialGraphClassifier(ClassifierMixin, BaseEstimator):
        def __init__(self, architecture="gnn_xgb", k=8, epochs=12, batch=512, lr=1e-3, seed=SEED):
            self.architecture, self.k, self.epochs, self.batch, self.lr, self.seed = architecture, k, epochs, batch, lr, seed
        def fit(self, X, y):
            raise ImportError("M17-M19 require PyTorch.")


def make_cnn_attention(seed=SEED, **kwargs):
    return CNNAttentionClassifier(seed=seed, **kwargs)


def make_dnn_ensemble(seed=SEED, **kwargs):
    return DNNEnsembleClassifier(seed=seed, **kwargs)


def make_gnn_xgb(seed=SEED, **kwargs):
    return SpatialGraphClassifier(architecture="gnn_xgb", seed=seed, **kwargs)


def make_gat(seed=SEED, **kwargs):
    return SpatialGraphClassifier(architecture="gat", seed=seed, **kwargs)


def make_graphsage_xgb(seed=SEED, **kwargs):
    return SpatialGraphClassifier(architecture="sage_xgb", seed=seed, **kwargs)


# ============================================================

# ============================================================
# M20-M25: ADDITIONAL FLOOD SUSCEPTIBILITY MODELS
# ============================================================

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold


def make_svm_rbf(seed=42):
    """M20: RBF support vector machine with fold-local preprocessing."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=seed,
        )),
    ])


def make_extra_trees(seed=42, n_estimators=400):
    """M21: Extra Trees ensemble."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def make_catboost_standalone(seed=42, iterations=400):
    """M22: Standalone CatBoost classifier."""
    from catboost import CatBoostClassifier

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", CatBoostClassifier(
            iterations=iterations,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        )),
    ])


def make_ebm(seed=42, max_bins=128):
    """M23: Explainable Boosting Machine."""
    from interpret.glassbox import ExplainableBoostingClassifier

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", ExplainableBoostingClassifier(
            max_bins=max_bins,
            interactions=10,
            outer_bags=4,
            inner_bags=0,
            learning_rate=0.02,
            random_state=seed,
        )),
    ])


class TabNetFloodClassifier(BaseEstimator, ClassifierMixin):
    """M24: TabNet wrapper with a scikit-learn-compatible interface."""

    def __init__(self, max_epochs=40, patience=8, seed=42):
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    def fit(self, X, y):
        import numpy as np
        from pytorch_tabnet.tab_model import TabNetClassifier

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.imputer_ = SimpleImputer(strategy="median")
        X = self.imputer_.fit_transform(X).astype(np.float32)
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("TabNetFloodClassifier requires two classes.")

        self.model_ = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=4,
            gamma=1.3,
            seed=self.seed,
            verbose=0,
            device_name="cpu",
        )

        self.model_.fit(
            X_train=X,
            y_train=y,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )
        return self

    def predict_proba(self, X):
        import numpy as np
        X = np.asarray(X, dtype=np.float32)
        X = self.imputer_.transform(X).astype(np.float32)
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def make_tabnet(seed=42, max_epochs=40, patience=8):
    return TabNetFloodClassifier(
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
    )


class CalibratedFloodEnsemble(BaseEstimator, ClassifierMixin):
    """
    M25: Mean of internally calibrated RF, XGBoost, and CatBoost
    probabilities. Calibration uses only the data passed to fit().
    """

    def __init__(self, seed=42, cv=3, n_estimators=250):
        self.seed = seed
        self.cv = cv
        self.n_estimators = n_estimators

    def fit(self, X, y):
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.imputer_ = SimpleImputer(strategy="median")
        X = self.imputer_.fit_transform(X)
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("CalibratedFloodEnsemble requires two classes.")

        counts = np.bincount(y)
        if counts.min() < self.cv:
            raise ValueError(
                f"Not enough examples per class for {self.cv}-fold calibration."
            )

        inner_cv = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.seed,
        )

        estimators = [
            RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_features="sqrt",
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1,
            ),
            XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.seed,
                n_jobs=4,
            ),
            CatBoostClassifier(
                iterations=self.n_estimators,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                random_seed=self.seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=4,
            ),
        ]

        self.calibrated_models_ = []
        for estimator in estimators:
            calibrated = CalibratedClassifierCV(
                estimator=clone(estimator),
                method="sigmoid",
                cv=inner_cv,
            )
            calibrated.fit(X, y)
            self.calibrated_models_.append(calibrated)

        return self

    def predict_proba(self, X):
        import numpy as np
        X = np.asarray(X, dtype=float)
        X = self.imputer_.transform(X)

        probabilities = [
            model.predict_proba(X)[:, 1]
            for model in self.calibrated_models_
        ]
        positive = np.mean(probabilities, axis=0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def make_calibrated_ensemble(seed=42, cv=3, n_estimators=250):
    return CalibratedFloodEnsemble(
        seed=seed,
        cv=cv,
        n_estimators=n_estimators,
    )



# STABLE MODEL REGISTRY
# ============================================================

MODEL_BUILDERS = {
    "M01": make_lr,
    "M02": make_rf,
    "M03": make_xgb,
    "M04": make_lgbm,
    "M05": make_mlp,
    "M06": make_stacking,
    "M07": make_soft_voting,
    "M08": lambda seed=SEED, **kwargs: GAXGB(
        seed=seed,
        **kwargs,
    ),
    "M09": make_rf_feature_selection_lgbm,
    "M10": make_catboost_stacking,
    "M11": lambda seed=SEED, **kwargs: MLPEmbedXGB(seed=seed, **kwargs),
    "M12": make_autoencoder_xgb,
    "M13": make_transformer_xgb,
    "M14": make_ft_transformer,
    "M15": make_cnn_attention,
    "M16": make_dnn_ensemble,
    "M17": make_gnn_xgb,
    "M18": make_gat,
    "M19": make_graphsage_xgb,
    "M20": make_svm_rbf,
    "M21": make_extra_trees,
    "M22": make_catboost_standalone,
    "M23": make_ebm,
    "M24": make_tabnet,
    "M25": make_calibrated_ensemble,
}


MODEL_NAMES = {
    "M01": "Logistic Regression",
    "M02": "Random Forest",
    "M03": "XGBoost",
    "M04": "LightGBM",
    "M05": "Multilayer Perceptron",
    "M06": "Stacking: RF + XGBoost + LightGBM",
    "M07": "Soft Voting: RF + XGBoost + MLP",
    "M08": "Genetic Algorithm Optimised XGBoost",
    "M09": "RF Feature Selection + LightGBM",
    "M10": "CatBoost Stacking",
    "M11": "MLP Embedding + XGBoost",
    "M12": "Autoencoder + XGBoost",
    "M13": "Transformer + XGBoost",
    "M14": "FT-Transformer",
    "M15": "CNN + Attention (tabular experiment)",
    "M16": "Deep Neural Network Ensemble",
    "M17": "GNN + XGBoost",
    "M18": "Graph Attention Network",
    "M19": "GraphSAGE + XGBoost",
    "M20": "RBF Support Vector Machine",
    "M21": "Extra Trees",
    "M22": "Standalone CatBoost",
    "M23": "Explainable Boosting Machine",
    "M24": "TabNet",
    "M25": "Calibrated Ensemble (RF + XGBoost + CatBoost)",
}


def get_model_by_id(model_id, seed=SEED, **kwargs):
    """Construct a model by stable model ID."""

    model_id = str(model_id).strip().upper()

    if model_id not in MODEL_NAMES:
        raise KeyError(
            f"Unknown or not-yet-implemented model ID: {model_id}"
        )

    if model_id not in MODEL_BUILDERS:
        raise NotImplementedError(
            f"{model_id} is registered but not implemented."
        )

    return MODEL_BUILDERS[model_id](
        seed=seed,
        **kwargs,
    )


def get_implemented_model_ids():
    return list(MODEL_BUILDERS.keys())


def get_model_metadata(model_id):
    model_id = str(model_id).strip().upper()

    if model_id not in MODEL_NAMES:
        raise KeyError(model_id)

    return {
        "id": model_id,
        "name": MODEL_NAMES[model_id],
        "implemented": model_id in MODEL_BUILDERS,
    }


# ============================================================
# LEGACY API — IMPORTANT FOR YOUR EXISTING TRAINING SCRIPT
# ============================================================

def get_models(fast=False):
    """
    Return the original baseline and hybrid dictionaries.

    This preserves compatibility with the original
    03_train_models.py.
    """

    baselines = {
        "LR": make_lr(),
        "RF": make_rf(),
        "XGB": make_xgb(),
        "LGBM": make_lgbm(),
        "MLP": make_mlp(),
    }

    hybrids = {
        "H1 Stacking (RF+XGB+LGBM -> LR)": make_stacking(),

        "H2 Soft-Voting (RF+XGB+MLP)": make_soft_voting(),

        "H3 GA-XGBoost": GAXGB(
            pop=6 if fast else 10,
            gens=2 if fast else 6,
            cv=3,
            sample=3000 if fast else 8000,
        ),

        "H5 MLP-embedding + XGBoost": MLPEmbedXGB(),

        "H6 RF-feature-selection + LGBM":
            make_rf_feature_selection_lgbm(),

        "M10 CatBoost Stacking": make_catboost_stacking(),
        "M12 Autoencoder + XGBoost": make_autoencoder_xgb(),
        "M13 Transformer + XGBoost": make_transformer_xgb(),
        "M14 FT-Transformer": make_ft_transformer(),
        "M15 CNN + Attention": make_cnn_attention(epochs=3 if fast else 15),
        "M16 DNN Ensemble": make_dnn_ensemble(n_members=2 if fast else 5, max_iter=100 if fast else 200),
        "M17 GNN + XGBoost": make_gnn_xgb(epochs=3 if fast else 12),
        "M18 Graph Attention Network": make_gat(epochs=3 if fast else 12),
        "M19 GraphSAGE + XGBoost": make_graphsage_xgb(epochs=3 if fast else 12),
        "M20 RBF-SVM": make_svm_rbf(),
        "M21 Extra Trees": make_extra_trees(
            n_estimators=150 if fast else 400
        ),
        "M22 Standalone CatBoost": make_catboost_standalone(
            iterations=100 if fast else 400
        ),
        "M23 Explainable Boosting Machine": make_ebm(),
        "M24 TabNet": make_tabnet(
            max_epochs=5 if fast else 40,
            patience=2 if fast else 8,
        ),
        "M25 Calibrated Ensemble": make_calibrated_ensemble(
            cv=3,
            n_estimators=100 if fast else 250,
        ),
    }

    if HAS_TORCH:
        hybrids["H4 CNN-BiLSTM"] = CNNBiLSTM(
            epochs=2 if fast else 40,
            batch=256,
            patience=2 if fast else 6,
        )

    return baselines, dict(sorted(hybrids.items()))


# ============================================================
# DEPENDENCY STATUS
# ============================================================

def dependency_status():
    return {
        "xgboost": HAS_XGB,
        "lightgbm": HAS_LGBM,
        "catboost": HAS_CAT,
        "torch": HAS_TORCH,
    }


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score

    print("Dependency status:")
    for name, installed in dependency_status().items():
        print(f"  {name}: {'available' if installed else 'unavailable'}")

    print("\nImplemented IDs:", get_implemented_model_ids())

    X_test, y_test = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        random_state=SEED,
    )

    # Small smoke tests. GA and deep learning are excluded here
    # because their full training can take considerably longer.
    test_ids = [
        "M01",
        "M02",
        "M03",
        "M04",
        "M05",
        "M06",
        "M07",
        "M09",
    ]

    if HAS_CAT:
        test_ids.append("M10")

    failures = []

    for model_id in test_ids:
        print(f"\nTesting {model_id}: {MODEL_NAMES[model_id]}")

        try:
            model = clone(
                get_model_by_id(model_id)
            )

            model.fit(X_test, y_test)

            probabilities = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(
                y_test,
                probabilities,
            )

            print(f"  PASS — ROC-AUC: {auc:.4f}")

        except Exception as exc:
            failures.append((model_id, str(exc)))
            print(f"  FAIL — {type(exc).__name__}: {exc}")

    print("\nLegacy API test:")

    baselines, hybrids = get_models(fast=True)

    print("Baselines:", list(baselines))
    print("Hybrids:", list(hybrids))

    if failures:
        print("\nFailures:")
        for model_id, error in failures:
            print(f"  {model_id}: {error}")

        raise SystemExit(1)

    print("\nSelf-test completed.")



# """
# All models. Baselines = single models. Hybrids = combinations of models / optimisers.
# If xgboost / lightgbm / catboost / torch are not installed, the code falls back
# (or skips the model) and tells you.
# """
# import warnings
# import numpy as np
# from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.ensemble import (RandomForestClassifier, StackingClassifier, VotingClassifier,
#                               HistGradientBoostingClassifier)
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.preprocessing import StandardScaler

# try:
#     from xgboost import XGBClassifier; HAS_XGB = True
# except Exception:
#     HAS_XGB = False
# try:
#     from lightgbm import LGBMClassifier; HAS_LGBM = True
# except Exception:
#     HAS_LGBM = False
# try:
#     from catboost import CatBoostClassifier; HAS_CAT = True
# except Exception:
#     HAS_CAT = False
# try:
#     import torch, torch.nn as nn; HAS_TORCH = True
# except Exception:
#     HAS_TORCH = False

# SEED = 42
# for name, ok in [("xgboost", HAS_XGB), ("lightgbm", HAS_LGBM), ("catboost", HAS_CAT), ("torch", HAS_TORCH)]:
#     if not ok:
#         print(f"[models] {name} not installed -> " + ("using scikit-learn fallback" if name in ("xgboost", "lightgbm") else "skipping related parts"))


# # --------------------------------------------------------------- base learners
# def make_xgb(seed=SEED, **p):
#     d = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=1)
#     d.update(p)
#     if HAS_XGB:
#         return XGBClassifier(**d, eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=seed)
#     return HistGradientBoostingClassifier(max_iter=d["n_estimators"], max_depth=d["max_depth"],
#                                           learning_rate=d["learning_rate"], max_features=d["colsample_bytree"],
#                                           min_samples_leaf=int(d["min_child_weight"]) * 10, random_state=seed)


# def make_lgbm(seed=SEED):
#     if HAS_LGBM:
#         return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
#                               colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)
#     return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=seed)


# def make_rf(seed=SEED):
#     return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed)


# def make_mlp(seed=SEED):
#     return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
#                                                          max_iter=300, random_state=seed))


# # ------------------------------------------------------------------ HYBRID 3
# class GAXGB(BaseEstimator, ClassifierMixin):
#     """Genetic Algorithm searches the best XGBoost settings, then trains the final model."""
#     SPACE = [("n_estimators", 100, 600, int), ("max_depth", 3, 10, int), ("learning_rate", 0.01, 0.3, float),
#              ("subsample", 0.5, 1.0, float), ("colsample_bytree", 0.5, 1.0, float), ("min_child_weight", 1, 10, int)]

#     def __init__(self, pop=10, gens=6, cv=3, sample=8000, seed=SEED):
#         self.pop, self.gens, self.cv, self.sample, self.seed = pop, gens, cv, sample, seed

#     def _decode(self, g):
#         return {n: t(lo + x * (hi - lo)) for x, (n, lo, hi, t) in zip(g, self.SPACE)}

#     def fit(self, X, y):
#         X, y = np.asarray(X), np.asarray(y)
#         rng = np.random.default_rng(self.seed)
#         idx = rng.choice(len(y), min(self.sample, len(y)), replace=False)
#         Xs, ys = X[idx], y[idx]
#         folds = StratifiedKFold(self.cv, shuffle=True, random_state=self.seed)
#         def fitness(g):
#             return cross_val_score(make_xgb(**self._decode(g)), Xs, ys, cv=folds, scoring="roc_auc").mean()
#         dim = len(self.SPACE)
#         pop = rng.random((self.pop, dim)); scores = np.array([fitness(g) for g in pop])
#         self.history_ = []
#         for _ in range(self.gens):
#             order = np.argsort(-scores); pop, scores = pop[order], scores[order]
#             self.history_.append(float(scores[0]))
#             kids = []
#             while len(kids) < self.pop - 2:
#                 pick = lambda: pop[min(rng.integers(0, self.pop, 2))]            # tournament (lower index = fitter)
#                 a, b = pick(), pick()
#                 child = np.where(rng.random(dim) < 0.5, a, b)                      # crossover
#                 child = np.clip(child + (rng.random(dim) < 0.2) * rng.normal(0, 0.15, dim), 0, 1)   # mutation
#                 kids.append(child)
#             kids = np.array(kids)
#             pop = np.vstack([pop[:2], kids]); scores = np.concatenate([scores[:2], [fitness(k) for k in kids]])
#         best = pop[int(np.argmax(scores))]
#         self.best_params_ = self._decode(best)
#         self.model_ = make_xgb(**self.best_params_).fit(X, y)
#         self.classes_ = self.model_.classes_
#         return self

#     def predict_proba(self, X): return self.model_.predict_proba(np.asarray(X))
#     def predict(self, X): return self.model_.predict(np.asarray(X))


# # ------------------------------------------------------------------ HYBRID 5
# class MLPEmbedXGB(BaseEstimator, ClassifierMixin):
#     """Neural network learns a compact representation; XGBoost classifies raw features + that representation."""
#     def __init__(self, hidden=(64, 32), seed=SEED):
#         self.hidden, self.seed = hidden, seed

#     def _embed(self, Xs):
#         a = Xs
#         for W, b in zip(self.mlp_.coefs_[:-1], self.mlp_.intercepts_[:-1]):
#             a = np.maximum(a @ W + b, 0)
#         return a

#     def fit(self, X, y):
#         X = np.asarray(X)
#         self.scaler_ = StandardScaler().fit(X); Xs = self.scaler_.transform(X)
#         self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden, early_stopping=True, max_iter=300,
#                                   random_state=self.seed).fit(Xs, y)
#         self.gbm_ = make_xgb(self.seed).fit(np.hstack([Xs, self._embed(Xs)]), y)
#         self.classes_ = self.gbm_.classes_
#         return self

#     def _z(self, X):
#         Xs = self.scaler_.transform(np.asarray(X)); return np.hstack([Xs, self._embed(Xs)])
#     def predict_proba(self, X): return self.gbm_.predict_proba(self._z(X))
#     def predict(self, X): return self.gbm_.predict(self._z(X))


# # ------------------------------------------------------------------ HYBRID 4
# if HAS_TORCH:
#     class _Net(nn.Module):
#         def __init__(self, c=32, h=32):
#             super().__init__()
#             self.conv = nn.Sequential(nn.Conv1d(1, c, 3, padding=1), nn.ReLU(), nn.Conv1d(c, c, 3, padding=1), nn.ReLU())
#             self.lstm = nn.LSTM(c, h, batch_first=True, bidirectional=True)
#             self.head = nn.Sequential(nn.Linear(2 * h, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

#         def forward(self, x):                       # x: (batch, n_features)
#             x = self.conv(x.unsqueeze(1))           # (batch, c, n_features)
#             out, _ = self.lstm(x.permute(0, 2, 1))  # (batch, n_features, 2h)
#             return self.head(out.mean(1)).squeeze(-1)

#     class CNNBiLSTM(BaseEstimator, ClassifierMixin):
#         """1-D CNN reads local feature patterns, BiLSTM reads them as a sequence, dense head classifies."""
#         def __init__(self, epochs=40, batch=256, lr=1e-3, patience=6, seed=SEED):
#             self.epochs, self.batch, self.lr, self.patience, self.seed = epochs, batch, lr, patience, seed

#         def fit(self, X, y):
#             torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
#             X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
#             self.scaler_ = StandardScaler().fit(X); Xs = self.scaler_.transform(X).astype(np.float32)
#             idx = rng.permutation(len(y)); nv = max(int(0.1 * len(y)), 1)
#             va, tr = idx[:nv], idx[nv:]
#             dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
#             self.dev_ = dev; net = _Net().to(dev)
#             opt = torch.optim.Adam(net.parameters(), lr=self.lr); lossf = nn.BCEWithLogitsLoss()
#             Xt, yt = torch.tensor(Xs[tr]).to(dev), torch.tensor(y[tr]).to(dev)
#             Xv, yv = torch.tensor(Xs[va]).to(dev), torch.tensor(y[va]).to(dev)
#             best, bad, state = 1e9, 0, None
#             for _ in range(self.epochs):
#                 net.train(); perm = torch.randperm(len(tr), device=dev)
#                 for i in range(0, len(tr), self.batch):
#                     b = perm[i:i + self.batch]; opt.zero_grad()
#                     lossf(net(Xt[b]), yt[b]).backward(); opt.step()
#                 net.eval()
#                 with torch.no_grad(): vl = lossf(net(Xv), yv).item()
#                 if vl < best - 1e-4:
#                     best, bad = vl, 0; state = {k: v.detach().clone() for k, v in net.state_dict().items()}
#                 else:
#                     bad += 1
#                     if bad >= self.patience: break
#             net.load_state_dict(state); self.net_ = net.eval(); self.classes_ = np.array([0, 1])
#             return self

#         def predict_proba(self, X):
#             Xs = self.scaler_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
#             with torch.no_grad():
#                 p = torch.sigmoid(self.net_(torch.tensor(Xs).to(self.dev_))).cpu().numpy()
#             return np.column_stack([1 - p, p])
#         def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# # -------------------------------------------------------------------- registry
# def get_models(fast=False):
#     baselines = {
#         "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
#         "RF": make_rf(), "XGB": make_xgb(), "LGBM": make_lgbm(), "MLP": make_mlp(),
#     }
#     stack_est = [("rf", make_rf()), ("xgb", make_xgb()), ("lgbm", make_lgbm())]
#     if HAS_CAT:
#         stack_est.append(("cat", CatBoostClassifier(iterations=300, verbose=0, random_seed=SEED)))
#     hybrids = {
#         "H1 Stacking (RF+XGB+LGBM -> LR)": StackingClassifier(stack_est, final_estimator=LogisticRegression(max_iter=2000),
#                                                                stack_method="predict_proba", cv=5),
#         "H2 Soft-Voting (RF+XGB+MLP)": VotingClassifier([("rf", make_rf()), ("xgb", make_xgb()), ("mlp", make_mlp())], voting="soft"),
#         "H3 GA-XGBoost": GAXGB(pop=6, gens=3, sample=3000) if fast else GAXGB(),
#         "H5 MLP-embedding + XGBoost": MLPEmbedXGB(),
#         "H6 RF-feature-selection + LGBM": Pipeline([("sel", SelectFromModel(RandomForestClassifier(300, n_jobs=-1, random_state=SEED), threshold="median")),
#                                                     ("clf", make_lgbm())]),
#     }
#     if HAS_TORCH:
#         hybrids["H4 CNN-BiLSTM"] = CNNBiLSTM(epochs=10 if fast else 40)
#     return baselines, dict(sorted(hybrids.items()))
