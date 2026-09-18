# Submission summary: Restaurant monthly revenue

## Required model

- **Target:** `Monthly_Revenue` (INR).
- **Predictors:** seven numeric operating variables plus nominal `Cuisine_Type` and `Location_Type`.
- **Excluded:** `Outlet_ID`; it identifies a record and has no legitimate operational interpretation as a predictor.
- **Split:** 2,000 training rows and 500 test rows (80/20, seed 42).
- **Final workflow:** cap invalid ratings to the 1-5 scale, standardize numeric predictors, one-hot encode nominal predictors with one reference category, and fit multiple ordinary least squares regression.

## Data quality and corrective action

There are no missing values and no duplicate rows. Five `Customer_Rating` values exceed 5; they were capped to 5.0, with no records removed. The correction is explicitly recorded in `data_corrections.json`. All remaining checked ranges are valid.

## Assumption evidence

| Check | Evidence / decision |
|---|---|
| Linearity | `figures/linearity_daily_customers.svg` and residual-versus-fitted plot provide visual checks. The final model has high out-of-sample explanatory power. |
| Multicollinearity | Maximum VIF is 1.2226: no concerning redundancy. |
| Independence | Durbin-Watson is 2.057, consistent with no material serial correlation in this non-time-series generated data. |
| Homoscedasticity | Inspect `figures/residuals_vs_fitted.svg`; use it to discuss any remaining spread pattern honestly. |
| Normality | Inspect `figures/qq_plot_residuals.svg`; regression relies on residuals, not normal input features. |
| Outliers/influence | 21 standardized residuals exceed 3 and 111 Cook's-distance values exceed 4/n. They are valid generated observations, so they were retained and disclosed rather than silently deleted. |
| Re-check | The model is refit after capping ratings and excluding the identifier. The same diagnostics are generated from the final fitted workflow. |

## Held-out test performance

- R-squared: **0.9745**
- Adjusted training R-squared: **0.9773**
- MAE: **INR 8,707.18**
- MSE: **125,707,078.88**
- RMSE: **INR 11,211.92**

For a manager, the app's result is an estimated monthly INR revenue for a specified outlet scenario. It is useful for comparing operational plans; it is not a guarantee of future sales.
