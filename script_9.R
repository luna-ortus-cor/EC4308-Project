# set working directory
setwd("C:/Users/russn/Downloads/")

library(readr)
library(dplyr)
library(zoo)
library(pls)
library(ggplot2)
library(forecast)
library(tidyr)
library(tseries)
library(xgboost)
library(caret)
library(lmtest)

raw_data <- read_csv("2025-09-MD.csv")

# Convert to monthly growth (%)
data <- raw_data %>%
  mutate(INDPRO_growth = 100 * (INDPRO / lag(INDPRO) - 1)) %>%
  filter(!is.na(INDPRO_growth)) 

# Calculate % of missing values per column
na_share <- colMeans(is.na(data))
na_share[na_share > 0]

# na_share[na_share > 0]
# CMRMTSPLx        PERMIT      PERMITNE      PERMITMW       PERMITS       PERMITW        ACOGNO 
# 0.00125       0.01500       0.01500       0.01500       0.01500       0.01500       0.49750 
# ANDENOx       BUSINVx      ISRATIOx      NONREVSL        CONSPI S&P div yield  S&P PE ratio 
# 0.13625       0.00125       0.00125       0.00125       0.00125       0.00250       0.00125 
# CP3Mx     COMPAPFFx TWEXAFEGSMTHx      UMCSENTx   DTCOLNVHFNM      DTCTHFNM       VIXCLSx 
# 0.00125       0.00125       0.21000       0.19250       0.00125       0.00125       0.05250 

data$sasdate <- as.Date(data$sasdate, format = "%m/%d/%Y")

data <- data %>% 
  dplyr::select(-ACOGNO)%>%
  dplyr::filter(sasdate >= as.Date("1978-01-01"))%>%
  dplyr::filter(!(sasdate %in% as.Date(c("2025-07-01", "2025-08-01"))))

# impute CP3Mx and COMPAPFFx using linear interpolation
data <- data %>%
  mutate(
    CP3Mx = na.approx(CP3Mx, na.rm = FALSE),
    COMPAPFFx = na.approx(COMPAPFFx, na.rm = FALSE)
  )

# check that CP3Mx and COMPAPFFx is filled (should give 0, 0)
colSums(is.na(data[c("CP3Mx", "COMPAPFFx")]))

train_data <- data %>% filter(sasdate < as.Date("2016-01-01"))
test_data  <- data %>% filter(sasdate >= as.Date("2016-01-01"))

# --- Define response and predictors ---
y_train <- train_data$INDPRO_growth
y_test  <- test_data$INDPRO_growth

X_train <- train_data %>% dplyr::select(-sasdate, -INDPRO, -INDPRO_growth)
X_test  <- test_data %>% dplyr::select(-sasdate, -INDPRO, -INDPRO_growth)

# --- Standardize predictors using training statistics ---
X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test,
                       center = attr(X_train_scaled, "scaled:center"),
                       scale  = attr(X_train_scaled, "scaled:scale"))

# --- Create lagged features ---
create_lagged_features <- function(data, y_col, n_lags = 4) {
  y <- data[[y_col]]
  
  y_lagged <- data.frame(matrix(NA, nrow = nrow(data), ncol = n_lags))
  colnames(y_lagged) <- paste0(y_col, "_lag", 1:n_lags)
  
  for (i in 1:n_lags) {
    y_lagged[, i] <- dplyr::lag(y, i)
  }
  
  X_cols <- setdiff(names(data), c("sasdate", "INDPRO", y_col))
  X_lagged_list <- list()
  
  for (col in X_cols) {
    for (lag_val in 1:n_lags) {
      lag_col_name <- paste0(col, "_lag", lag_val)
      X_lagged_list[[lag_col_name]] <- dplyr::lag(data[[col]], lag_val)
    }
  }
  
  X_lagged <- as.data.frame(X_lagged_list)
  
  all_features <- cbind(data[, setdiff(names(data), c("sasdate", "INDPRO"))],
                        y_lagged,
                        X_lagged)
  
  return(all_features)
}

# Apply lagging to train and test data
n_lags <- 4
train_data_lagged <- create_lagged_features(train_data, "INDPRO_growth", n_lags)
test_data_lagged <- create_lagged_features(test_data, "INDPRO_growth", n_lags)

train_data_lagged <- train_data_lagged %>% drop_na()
test_data_lagged <- test_data_lagged %>% drop_na()

y_train_boost <- train_data_lagged$INDPRO_growth
X_train_boost <- train_data_lagged %>% dplyr::select(-INDPRO_growth)

y_test_boost <- test_data_lagged$INDPRO_growth
X_test_boost <- test_data_lagged %>% dplyr::select(-INDPRO_growth)

# Standardize predictors
X_train_boost_scaled <- scale(X_train_boost)
X_test_boost_scaled <- scale(X_test_boost,
                             center = attr(X_train_boost_scaled, "scaled:center"),
                             scale  = attr(X_train_boost_scaled, "scaled:scale"))

dtrain <- xgb.DMatrix(data = as.matrix(X_train_boost_scaled), label = y_train_boost)
dtest <- xgb.DMatrix(data = as.matrix(X_test_boost_scaled), label = y_test_boost)

# ===== HYPERPARAMETER OPTIMIZATION =====
cat("===== HYPERPARAMETER OPTIMIZATION =====\n\n")

# Grid search for optimal hyperparameters
param_grid <- expand.grid(
  max_depth = c(3, 4, 5),
  eta = c(0.05, 0.1, 0.15),
  subsample = c(0.7, 0.8),
  colsample_bytree = c(0.7, 0.8)
)

best_test_rmse <- Inf
best_params <- NULL
results_grid <- list()

set.seed(42)
for (i in 1:nrow(param_grid)) {
  params <- list(
    objective = "reg:squarederror",
    max_depth = param_grid$max_depth[i],
    eta = param_grid$eta[i],
    subsample = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    min_child_weight = 1,
    gamma = 0,
    lambda = 1,
    alpha = 0.5
  )
  
  xgb_cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 500,
    nfold = 5,
    early_stopping_rounds = 20,
    verbose = 0,
    metrics = "rmse"
  )
  
  test_rmse <- min(xgb_cv$evaluation_log$test_rmse_mean)
  
  results_grid[[i]] <- data.frame(
    max_depth = param_grid$max_depth[i],
    eta = param_grid$eta[i],
    subsample = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    test_rmse = test_rmse,
    best_nrounds = xgb_cv$best_iteration
  )
  
  if (test_rmse < best_test_rmse) {
    best_test_rmse <- test_rmse
    best_params <- params
    best_nrounds_opt <- xgb_cv$best_iteration
  }
  
  cat(sprintf("Iteration %d/%d | Depth: %d, Eta: %.2f, RMSE: %.4f\n",
              i, nrow(param_grid), params$max_depth, params$eta, test_rmse))
}

results_df_grid <- do.call(rbind, results_grid)
cat("\n===== TOP 5 PARAMETER COMBINATIONS =====\n")
print(head(results_df_grid[order(results_df_grid$test_rmse), ], 5))

# ===== TRAIN FINAL MODEL WITH BEST PARAMETERS =====
cat("\n===== TRAINING FINAL MODEL =====\n")
set.seed(42)
xgb_model <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = best_nrounds_opt,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 1,
  print_every_n = 50
)

# ===== PREDICTIONS AND ERROR METRICS =====
y_train_pred <- predict(xgb_model, dtrain)
y_test_pred <- predict(xgb_model, dtest)

# Calculate all error metrics
train_mse <- mean((y_train_boost - y_train_pred)^2)
test_mse <- mean((y_test_boost - y_test_pred)^2)
train_rmse <- sqrt(train_mse)
test_rmse <- sqrt(test_mse)
train_mae <- mean(abs(y_train_boost - y_train_pred))
test_mae <- mean(abs(y_test_boost - y_test_pred))
train_r2 <- 1 - sum((y_train_boost - y_train_pred)^2) / sum((y_train_boost - mean(y_train_boost))^2)
test_r2 <- 1 - sum((y_test_boost - y_test_pred)^2) / sum((y_test_boost - mean(y_test_boost))^2)

# Mean Absolute Percentage Error
train_mape <- mean(abs((y_train_boost - y_train_pred) / y_train_boost)) * 100
test_mape <- mean(abs((y_test_boost - y_test_pred) / y_test_boost)) * 100

cat("\n===== MODEL PERFORMANCE =====\n")
cat("TRAINING SET:\n")
cat("  MSE:  ", round(train_mse, 6), "\n")
cat("  RMSE: ", round(train_rmse, 4), "\n")
cat("  MAE:  ", round(train_mae, 4), "\n")
cat("  MAPE: ", round(train_mape, 2), "%\n")
cat("  R²:   ", round(train_r2, 4), "\n\n")
cat("TEST SET:\n")
cat("  MSE:  ", round(test_mse, 6), "\n")
cat("  RMSE: ", round(test_rmse, 4), "\n")
cat("  MAE:  ", round(test_mae, 4), "\n")
cat("  MAPE: ", round(test_mape, 2), "%\n")
cat("  R²:   ", round(test_r2, 4), "\n")

# ===== STATISTICAL SIGNIFICANCE TESTS =====
cat("\n===== STATISTICAL SIGNIFICANCE TESTS =====\n\n")

residuals_test <- y_test_boost - y_test_pred

# 1. Residuals normality test (Shapiro-Wilk)
sw_test <- shapiro.test(residuals_test)
cat("1. SHAPIRO-WILK NORMALITY TEST (H0: Residuals are normal)\n")
cat("   Statistic:", round(sw_test$statistic, 4), "\n")
cat("   P-value:  ", round(sw_test$p.value, 4), "\n")
cat("   Result:   ", ifelse(sw_test$p.value < 0.05, "REJECT H0 - Not normal", "FAIL TO REJECT H0 - Normal"), "\n\n")

# 2. Test for autocorrelation (Durbin-Watson)
dw_test <- dwtest(lm(residuals_test ~ 1))
cat("2. DURBIN-WATSON TEST (H0: No autocorrelation)\n")
cat("   Statistic:", round(dw_test$statistic, 4), "\n")
cat("   P-value:  ", round(dw_test$p.value, 4), "\n")
cat("   Result:   ", ifelse(dw_test$p.value < 0.05, "REJECT H0 - Autocorrelated", "FAIL TO REJECT H0 - No autocorr."), "\n\n")

# 3. Test for heteroscedasticity (manual test)
fitted_vals <- y_test_pred
residuals_squared <- residuals_test^2
het_lm <- lm(residuals_squared ~ fitted_vals)
het_test_stat <- summary(het_lm)$r.squared * length(residuals_test)
het_pval <- pchisq(het_test_stat, df = 1, lower.tail = FALSE)
cat("3. HETEROSCEDASTICITY TEST (H0: Constant variance)\n")
cat("   Test Stat:", round(het_test_stat, 4), "\n")
cat("   P-value:  ", round(het_pval, 4), "\n")
cat("   Result:   ", ifelse(het_pval < 0.05, "REJECT H0 - Heteroscedastic", "FAIL TO REJECT H0 - Homoscedastic"), "\n\n")

# 4. Diebold-Mariano test for forecast accuracy (vs. benchmark: AR(4) model)
# Fit AR(4) on training data using Arima for more stability
library(forecast)
ar4_model <- auto.arima(y_train_boost, max.p = 4, max.d = 0, max.q = 0, 
                        stepwise = FALSE, approximation = FALSE, allowdrift = TRUE)

# Forecast for test set using AR(4)
ar4_fcst <- forecast(ar4_model, h = length(y_test_boost))
benchmark_pred <- ar4_fcst$mean

dm_errors <- (y_test_boost - y_test_pred)^2 - (y_test_boost - benchmark_pred)^2
dm_mean <- mean(dm_errors)
dm_se <- sd(dm_errors) / sqrt(length(dm_errors))
dm_stat <- dm_mean / dm_se
dm_pval <- 2 * pnorm(-abs(dm_stat))
cat("4. DIEBOLD-MARIANO TEST (XGBoost vs. Naive Forecast)\n")
cat("   Statistic:", round(dm_stat, 4), "\n")
cat("   P-value:  ", round(dm_pval, 4), "\n")
cat("   Result:   ", ifelse(dm_pval < 0.05, "XGBoost is SIGNIFICANTLY better", "No significant difference"), "\n\n")

# 5. Mean Forecast Error (bias test)
mfe <- mean(residuals_test)
mfe_se <- sd(residuals_test) / sqrt(length(residuals_test))
mfe_tstat <- mfe / mfe_se
mfe_pval <- 2 * pt(-abs(mfe_tstat), df = length(residuals_test) - 1)
cat("5. BIAS TEST (H0: Mean error = 0)\n")
cat("   Mean Error:", round(mfe, 4), "\n")
cat("   Std. Error:", round(mfe_se, 4), "\n")
cat("   T-statistic:", round(mfe_tstat, 4), "\n")
cat("   P-value:  ", round(mfe_pval, 4), "\n")
cat("   Result:   ", ifelse(mfe_pval < 0.05, "Model is BIASED", "Model is UNBIASED"), "\n\n")

# ===== FEATURE IMPORTANCE =====
cat("===== FEATURE IMPORTANCE (Top 15) =====\n")
importance_matrix <- xgb.importance(
  feature_names = colnames(X_train_boost_scaled),
  model = xgb_model
)
print(importance_matrix[1:15])

# ===== INTERPRETABILITY: PARTIAL DEPENDENCE =====
cat("\n===== PARTIAL DEPENDENCE ANALYSIS =====\n")

# Get top 6 features
top_features <- importance_matrix$Feature[1:6]

par(mfrow = c(2, 3), mar = c(4, 4, 2, 1))
for (feat in top_features) {
  feat_idx <- which(colnames(X_train_boost_scaled) == feat)
  
  # Create range of values
  feat_range <- seq(min(X_train_boost_scaled[, feat_idx], na.rm = TRUE),
                    max(X_train_boost_scaled[, feat_idx], na.rm = TRUE),
                    length.out = 50)
  
  # Average predictions across feature range
  X_temp <- X_train_boost_scaled
  pd_preds <- numeric(50)
  
  for (i in 1:50) {
    X_temp[, feat_idx] <- feat_range[i]
    d_temp <- xgb.DMatrix(data = as.matrix(X_temp))
    pd_preds[i] <- mean(predict(xgb_model, d_temp))
  }
  
  plot(feat_range, pd_preds, type = "l", lwd = 2,
       main = paste("Partial Dependence:", feat),
       xlab = paste(feat, "(standardized)"),
       ylab = "Predicted INDPRO Growth (%)")
  grid()
}
par(mfrow = c(1, 1))

# ===== VISUALIZATION OF PREDICTIONS =====
results_df_final <- data.frame(
  actual = y_test_boost,
  predicted = y_test_pred,
  residuals = y_test_boost - y_test_pred,
  time_index = 1:length(y_test_boost)
)

p1 <- ggplot(results_df_final, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Actual vs Predicted (Test Set)",
       x = "Actual INDPRO Growth (%)",
       y = "Predicted INDPRO Growth (%)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

p2 <- ggplot(results_df_final, aes(x = time_index, y = actual)) +
  geom_line(aes(color = "Actual"), size = 0.7) +
  geom_line(aes(y = predicted, color = "Predicted"), size = 0.7) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "steelblue")) +
  labs(title = "Time Series: Actual vs Predicted",
       x = "Time Index",
       y = "INDPRO Growth (%)") +
  theme_minimal() +
  theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

p3 <- ggplot(results_df_final, aes(x = predicted, y = residuals)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals vs Predicted",
       x = "Predicted INDPRO Growth (%)",
       y = "Residuals") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

p4 <- ggplot(results_df_final, aes(x = residuals)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Residuals",
       x = "Residuals",
       y = "Frequency") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

combined_plot <- gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 2)
print(combined_plot)
