"""
Air Quality Trend Predictor - Beijing Multi-Site Air Quality Data
Predicts future PM2.5 with Linear Regression and Random Forest, using a
chronological split. Compares both against a persistence baseline.

Usage:
    python air_quality_predictor.py --data data/PRSA_Data_20130301-20170228 --station Aotizhongxin
Outputs go to ./figures and ./results.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
POLLUTANTS_WEATHER = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NUMERIC = ["PM2.5"] + POLLUTANTS_WEATHER
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

TRAIN_END = "2015-02-28"
VAL_START, VAL_END = "2015-03-01", "2016-02-29"
TEST_START = "2016-03-01"


def load_and_clean(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index("datetime").sort_index()
    raw_missing = df[NUMERIC].isna().mean().round(4)

    # Fill only short gaps (<= 6 h). Longer gaps stay missing and are dropped later.
    df[NUMERIC] = df[NUMERIC].interpolate(method="time", limit=6, limit_area="inside")

    # Wind direction -> angle -> sin/cos (circular variable)
    ang = np.deg2rad(df["wd"].map({d: i * 22.5 for i, d in enumerate(COMPASS)}))
    df["wd_sin"] = np.sin(ang).ffill(limit=6)
    df["wd_cos"] = np.cos(ang).ffill(limit=6)
    return df, raw_missing


def make_features(df, horizon):
    X = pd.DataFrame(index=df.index)
    # Only information available at time t (no leakage from the future)
    for lag in [0, 1, 2, 5, 11, 23, 47]:
        X[f"pm25_lag{lag}"] = df["PM2.5"].shift(lag)
    X["pm25_roll24"] = df["PM2.5"].rolling(24, min_periods=12).mean()
    for c in POLLUTANTS_WEATHER + ["wd_sin", "wd_cos"]:
        X[c] = df[c]
    X["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    X["mon_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    X["mon_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    y = df["PM2.5"].shift(-horizon).rename("y")
    return X.join(y).dropna()


def metrics(y, p):
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": mean_squared_error(y, p) ** 0.5,
        "R2": r2_score(y, p),
    }


def eda_plots(df, station, outdir):
    fig, ax = plt.subplots(figsize=(14, 4))
    df["PM2.5"].resample("D").mean().plot(ax=ax, lw=0.8)
    ax.set(title=f"{station}: daily mean PM2.5", ylabel="ug/m3")
    fig.tight_layout(); fig.savefig(f"{outdir}/01_pm25_over_time.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df.groupby(df.index.month)["PM2.5"].mean().plot(kind="bar", ax=axes[0], title="Mean PM2.5 by month")
    df.groupby(df.index.hour)["PM2.5"].mean().plot(ax=axes[1], title="Mean PM2.5 by hour of day")
    fig.tight_layout(); fig.savefig(f"{outdir}/02_seasonal_daily_patterns.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[NUMERIC].corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    fig.colorbar(im); ax.set_title("Correlation heatmap")
    fig.tight_layout(); fig.savefig(f"{outdir}/03_correlation.png", dpi=130); plt.close(fig)


def run_horizon(df, horizon, station, figdir):
    data = make_features(df, horizon)
    train = data[:TRAIN_END]
    val = data[VAL_START:VAL_END]
    test = data[TEST_START:]
    feats = [c for c in data.columns if c != "y"]

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": None,  # tuned below on the validation set
    }
    models["Linear Regression"].fit(train[feats], train.y)

    # Small hyperparameter search for RF, chosen on VALIDATION only
    best, best_mae = None, np.inf
    for leaf in [5, 20]:
        for depth in [10, None]:
            rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=leaf, max_depth=depth,
                                       n_jobs=-1, random_state=SEED)
            rf.fit(train[feats], train.y)
            m = mean_absolute_error(val.y, rf.predict(val[feats]))
            if m < best_mae:
                best, best_mae, best_params = rf, m, (leaf, depth)
    models["Random Forest"] = best

    preds = {"Persistence": test["pm25_lag0"].values}
    for name, mdl in models.items():
        preds[name] = mdl.predict(test[feats])

    rows = []
    for name, p in preds.items():
        rows.append({"station": station, "horizon_h": horizon, "model": name, **metrics(test.y, p)})
    res = pd.DataFrame(rows)

    # Plots: actual vs predicted (full test year + 2-week zoom)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(test.index, test.y, label="Actual", lw=0.7)
    axes[0].plot(test.index, preds["Random Forest"], label="Random Forest", lw=0.7, alpha=0.8)
    axes[0].plot(test.index, preds["Linear Regression"], label="Linear Regression", lw=0.7, alpha=0.8)
    axes[0].set(title=f"{station}: actual vs predicted PM2.5 (+{horizon}h), test year", ylabel="ug/m3")
    axes[0].legend()
    z = slice(0, 24 * 14)
    axes[1].plot(test.index[z], test.y.values[z], label="Actual")
    axes[1].plot(test.index[z], preds["Random Forest"][z], label="Random Forest")
    axes[1].plot(test.index[z], preds["Linear Regression"][z], label="Linear Regression")
    axes[1].set(title="First two weeks of test period (zoom)", ylabel="ug/m3"); axes[1].legend()
    fig.tight_layout(); fig.savefig(f"{figdir}/04_actual_vs_pred_h{horizon}.png", dpi=130); plt.close(fig)

    # Failure analysis: error by pollution level (using RF)
    err = pd.DataFrame({"y": test.y.values, "abs_err": np.abs(test.y.values - preds["Random Forest"]),
                        "signed": preds["Random Forest"] - test.y.values}, index=test.index)
    bins = pd.cut(err.y, [0, 35, 75, 150, 250, np.inf], labels=["0-35", "35-75", "75-150", "150-250", "250+"])
    by_level = err.groupby(bins, observed=True).agg(n=("y", "size"), MAE=("abs_err", "mean"),
                                                   mean_signed_error=("signed", "mean")).round(2)
    by_season = err.groupby(err.index.month.map(
        lambda m: "winter" if m in (12, 1, 2) else "spring" if m in (3, 4, 5)
        else "summer" if m in (6, 7, 8) else "autumn")).abs_err.mean().round(2)

    if horizon == 1:
        imp = pd.Series(models["Random Forest"].feature_importances_, index=feats).sort_values()
        fig, ax = plt.subplots(figsize=(7, 6)); imp.tail(12).plot(kind="barh", ax=ax)
        ax.set_title("Random Forest feature importance (top 12)")
        fig.tight_layout(); fig.savefig(f"{figdir}/05_feature_importance.png", dpi=130); plt.close(fig)

    info = {"rf_params(min_samples_leaf,max_depth)": best_params,
            "n_train": len(train), "n_val": len(val), "n_test": len(test)}
    return res, by_level, by_season, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing the PRSA_Data_*.csv files")
    ap.add_argument("--station", default="Aotizhongxin")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24])
    a = ap.parse_args()

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    path = os.path.join(a.data, f"PRSA_Data_{a.station}_20130301-20170228.csv")
    df, raw_missing = load_and_clean(path)
    print(f"Station: {a.station}, rows: {len(df)}")
    print("Raw missing fraction per column:\n", raw_missing.to_string(), "\n")

    eda_plots(df, a.station, "figures")

    all_res = []
    for h in a.horizons:
        res, by_level, by_season, info = run_horizon(df, h, a.station, "figures")
        all_res.append(res)
        print(f"=== Horizon +{h}h  {info}")
        print(res.round(2).to_string(index=False))
        print("\nRF error by true PM2.5 level (negative mean_signed_error = under-prediction):")
        print(by_level.to_string())
        print("\nRF MAE by season:", by_season.to_dict(), "\n")

    pd.concat(all_res).round(3).to_csv("results/metrics.csv", index=False)
    print("Saved results/metrics.csv and figures/*.png")


if __name__ == "__main__":
    main()
    