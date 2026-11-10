# Beijing Multi-Zone PM2.5 Forecasting (2013–2017)

## 📌 Executive Summary
Air pollution in Beijing exhibits extreme seasonal and spatial variability across urban, suburban, and rural zones. This project builds an end-to-end Machine Learning forecasting pipeline using **105,192 hourly sensor records** (March 2013 – February 2017) across three monitoring stations (**Aotizhongxin**, **Changping**, and **Dingling**) to predict **1-hour ahead PM2.5 concentrations**.

By combining domain-specific meteorological feature engineering (dew-point depression / humidity proxy, cyclical diurnal encoding, and PACF-guided momentum lags) with a tuned **Histogram-Based Gradient Boosting Regressor (`HistGradientBoostingRegressor`)**, the model achieved an **$R^2$ of 0.9477** on the unseen 2016–2017 winter smog test set—reducing Root Mean Squared Error (RMSE) by **5.90%** and Mean Absolute Error (MAE) by **5.46%** over a strong Naive Persistence Baseline.

---

## 🔍 Key Exploratory Data Analysis (EDA) Findings
* **Urban-to-Rural Pollution Gradient:** Urban `Aotizhongxin` experiences **25.4% higher average PM2.5** ($82.77\text{ }\mu\text{g/m}^3$) than rural `Dingling` ($65.99\text{ }\mu\text{g/m}^3$). Furthermore, `Aotizhongxin` peaks around **midnight (00:00–01:00)** due to nighttime heavy-duty freight traffic and shallow boundary layers, whereas suburban `Changping` exhibits a double-rush-hour peak (10:00 and 21:00).
* **Geographical Wind Trap:** Easterly and southeasterly winds (`ESE`, `E`, `ENE`) transport industrial emissions from the North China Plain against Beijing's surrounding mountains, averaging **$95.9\text{–}99.7\text{ }\mu\text{g/m}^3$ PM2.5**—more than **double** the concentration of clean northwesterly mountain winds (`WNW` averages **$43.1\text{ }\mu\text{g/m}^3$**).
* **Two-Stage Missing Data Strategy:** Gap-length distribution analysis revealed that **82%–89% of missing PM2.5 streaks are $\le 3$ hours**, alongside rare multi-day sensor outages (up to 343 hours). Short gaps ($\le 6$ hours) were imputed via station-grouped linear interpolation prior to lag creation, while multi-day outages were dropped post-feature-engineering to prevent time-shift leakage.

---

## 🛠️ Feature Engineering Highlights
* **Humidity Proxy (`humid`):** Derived from temperature and dew point (`TEMP - DEWP`), ranking as the **#2 most important feature** via permutation importance despite weak standalone linear correlations.
* **PACF-Guided Lags & Momentum:** Partial Autocorrelation (PACF) analysis showed direct signal cutoff after Lag 2 (`Lag 1 = +0.969`, `Lag 2 = -0.246`). Created compact lag features (`pm25_lag_1h`, `pm25_lag_2h`), hourly rate-of-change (`pm25_diff_1h`), and 24-hour rolling means (`pm25_roll_mean_24h`).
* **Cyclical Time Encoding:** Transformed `hour` and `month` using sine/cosine functions (`hour_cos` ranked in the Top 10 features).

---

## 📊 Model Performance (Chronological Split)
* **Train Set:** Mar 2013 – Dec 2015 | **Validation Set:** Jan 2016 – Oct 2016 | **Test Set:** Nov 2016 – Feb 2017 (Winter Smog Season)

| Model | Split | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Persistence Baseline** | Validation | 8.23 | 17.42 | 0.9207 |
| **Linear / Ridge Regression** | Validation | 8.50 | 16.83 | 0.9260 |
| **Random Forest Regressor** | Validation | 8.08 | 16.89 | 0.9255 |
| **Tuned HistGradientBoosting** | Validation | **7.84 (-4.74%)** | **16.56 (-4.94%)** | **0.9283** |
| **Naive Persistence Baseline** | **Test (Unseen Winter)** | 12.45 | 23.72 | 0.9410 |
| **Tuned HistGradientBoosting** | **Test (Unseen Winter)** | **11.77 (-5.46%)** | **22.32 (-5.90%)** | **0.9477** |