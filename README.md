# Time Series Regression Statsmodels
This repository contains Python code for time series forecasting using statistical models from the statsmodels library. The project demonstrates three different modeling approaches for temperature data from three farms.
**A Nowa Analytics Machine Learning Project**  

## **Project Overview**  
This project demonstrates **time series forecasting** using statistical models implemented with the `statsmodels` library. We analyze temperature data from three farms and apply different autoregressive models to generate accurate predictions.  

### **Key Objectives**  
✔ **Study time series patterns** to select the best forecasting model  
✔ **Apply mathematical models** (AR, ARMA, ARIMA, SARIMA) using `statsmodels`  
✔ **Optimize model parameters** for improved accuracy  
✔ **Forecast unseen values** and evaluate performance  
✔ **Generate professional reports** using AI-powered analytics  

---

## **Models Implemented**  

| **Farm**  | **Model** | **Description** |  
|-----------|----------|----------------|  
| **Farm 1** | `AR(34)` | Autoregressive model with 34 lags |  
| **Farm 2** | `ARMA(24,2)` | Autoregressive Moving Average model |  
| **Farm 3** | `SARIMA(11,1,1)(2,1,1,12)` | Seasonal ARIMA with yearly seasonality |  

### **Code Implementation**  
```python
# Model fitting
mod_f1 = AutoReg(df_f1, lags=34, old_names=False).fit()
mod_f2 = ARIMA(df_f2, order=(24, 0, 2)).fit()
mod_f3 = SARIMAX(df_f3, order=(11, 1, 1), seasonal_order=(2, 1, 1, 12)).fit()

# Forecasting
forecast_f1 = mod_f1.predict(start=len(df_f1), end=len(df_f1) + 35)
forecast_f2 = mod_f2.predict(start=len(df_f2), end=len(df_f2) + 35)
forecast_f3 = mod_f3.predict(start=len(df_f3), end=len(df_f3) + 35)
```

---

## **Key Features**  
📊 **Interactive Forecast Visualization**  
- Historical vs. predicted values  
- Confidence intervals for uncertainty estimation  
- Comparative analysis across farms  

🔍 **Model Optimization**  
- ACF/PACF analysis for lag selection  
- AIC/BIC criteria for model comparison  
- Residual diagnostics for validation  

📑 **AI-Powered Reporting**  
- Automated insights using **Nowa Analytics' AI (Gamma)**  
- Professional styling for business presentations  

---

## **Requirements**  
- Python 3.8+  
- `statsmodels` (for time series modeling)  
- `pandas` (data manipulation)  
- `matplotlib` & `seaborn` (visualization)  

Install dependencies:  
```bash
pip install statsmodels pandas matplotlib seaborn
```

---

## **License**  
Developed by **Nowa Analytics** under the **MIT License**.  

---

**Nowa Analytics** – *Data-Driven Decisions, Powered by AI*
