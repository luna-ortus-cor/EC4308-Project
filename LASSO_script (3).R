#===============================================================================
# LASSO-based Forecasting for Industrial Production Growth
#===============================================================================

# Load libraries ---------------------------------------------------------------
library(readr)
library(dplyr)
library(zoo)
library(glmnet)
library(lubridate)

# Helper functions -------------------------------------------------------------

# Transformation function
apply_tcode <- function(x, tcode, scale100 = TRUE) {
  safe_log <- function(v) { v[v <= 0] <- NA_real_; log(v) }
  L <- function(x, k = 1) dplyr::lag(x, k)
  
  out <- switch(as.character(tcode),
                "1" = x,
                "2" = x - L(x, 1),
                "3" = (x - L(x, 1)) - (L(x, 1) - L(x, 2)),
                "4" = safe_log(x),
                "5" = { y <- safe_log(x) - safe_log(L(x, 1)); if (scale100) 100*y else y },
                "6" = { y <- (safe_log(x) - safe_log(L(x, 1))) - (safe_log(L(x, 1)) - safe_log(L(x, 2))); 
                if (scale100) 100*y else y },
                "7" = { g <- (x / L(x, 1)) - 1; d <- g - L(g, 1); if (scale100) 100*d else d },
                x
  )
  return(out)
}

# Create lagged features
create_lagged_data <- function(data, predictor_vars, max_lags, target_var, skip_vars = NULL) {
  lagged_data <- data %>% select(sasdate, all_of(target_var))
  
  for (var in predictor_vars) {
    if (var %in% skip_vars) {
      lagged_data[[var]] <- data[[var]]
    } else {
      for (lag in 1:max_lags) {
        lag_name <- paste0(var, "_lag", lag)
        lagged_data[[lag_name]] <- dplyr::lag(data[[var]], lag)
      }
    }
  }
  
  lagged_data <- lagged_data %>% drop_na()
  return(lagged_data)
}

# BIC calculation for glmnet
get_bic <- function(fit, X, y) {
  n <- length(y)
  preds <- predict(fit, X)
  rss <- colSums((y - preds)^2)
  df <- fit$df
  bic <- n * log(rss / n) + df * log(n)
  return(bic)
}

# Evaluation metrics
evaluate_model <- function(y_true, y_pred, model_name = "") {
  rmse <- sqrt(mean((y_true - y_pred)^2))
  mae <- mean(abs(y_true - y_pred))
  r2 <- 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
  
  cat(sprintf("\n=== %s PERFORMANCE ===\n", toupper(model_name)))
  cat(sprintf("RMSE: %.4f | MAE: %.4f | R²: %.4f\n", rmse, mae, r2))
  
  return(list(rmse = rmse, mae = mae, r2 = r2))
}

# Data preparation -------------------------------------------------------------

cat("Loading and transforming data...\n")

# Load raw data
raw_file <- read_csv("2025-09-MD.csv")
tcode_row <- raw_file[1, ]
data_raw <- raw_file[-1, ]

# Convert dates and handle missing values
data_raw <- data_raw %>%
  mutate(
    sasdate = mdy(sasdate),
    UMCSENTx = na.approx(UMCSENTx, na.rm = FALSE)
  )

# Extract transformation codes
num_cols <- setdiff(names(data_raw), "sasdate")
tcodes <- suppressWarnings(as.integer(tcode_row[num_cols] %>% unlist()))
names(tcodes) <- num_cols

# Apply transformations
stationary <- data_raw %>%
  mutate(across(all_of(num_cols),
                ~ apply_tcode(.x, tcodes[cur_column()]),
                .names = "{.col}")) %>%
  rename(INDPRO_growth = INDPRO)

# Final data preparation
data <- stationary %>% 
  select(-ACOGNO) %>%
  mutate(
    CP3Mx = na.approx(CP3Mx, na.rm = FALSE),
    COMPAPFFx = na.approx(COMPAPFFx, na.rm = FALSE),
    BAA_AAA = BAA - AAA,
    T10Y3M = GS10 - TB3MS,
    COVID_dummy = ifelse(sasdate >= as.Date("2020-03-01") & 
                           sasdate <= as.Date("2020-06-01"), 1, 0)
  ) %>%
  filter(
    sasdate >= as.Date("1978-01-01"),
    !(sasdate %in% as.Date(c("2025-07-01", "2025-08-01")))
  )

# Define variables and splits
target_var <- "INDPRO_growth"
exclude_vars <- c("sasdate", "INDPRO_growth")
predictor_vars <- setdiff(names(data), exclude_vars)

train_end <- as.Date("2015-12-01")
oos_start <- as.Date("2016-01-01")

# Add autoregressive lags
for (lag in 1:3) {
  lag_name <- paste0("INDPRO_growth_lag", lag)
  data[[lag_name]] <- dplyr::lag(data[[target_var]], lag)
}
base_ar_cols <- paste0("INDPRO_growth_lag", 1:3)
predictor_vars <- c(predictor_vars, base_ar_cols)

# Create lagged features
cat("Creating lagged features...\n")
max_lags <- 12
lagged_dataset <- create_lagged_data(data, predictor_vars, max_lags, target_var, 
                                     skip_vars = base_ar_cols)

cat(sprintf("Total features: %d\n", ncol(lagged_dataset) - 2))

# Train/test split
train_data <- lagged_dataset %>% filter(sasdate <= train_end)
test_data <- lagged_dataset %>% filter(sasdate >= oos_start)

X_train <- train_data %>% select(-sasdate, -INDPRO_growth) %>% as.matrix()
y_train <- train_data$INDPRO_growth
X_test <- test_data %>% select(-sasdate, -INDPRO_growth) %>% as.matrix()
y_test <- test_data$INDPRO_growth

# Set penalty factors (don't penalize AR terms)
base_ar_indices <- which(colnames(X_train) %in% base_ar_cols)
penalty <- rep(1, ncol(X_train))
penalty[base_ar_indices] <- 0

# Data quality checks
cat("\n=== DATA QUALITY CHECKS ===\n")
cat(sprintf("Train obs: %d | Test obs: %d\n", nrow(X_train), nrow(X_test)))
cat(sprintf("NAs in train: %d | NAs in test: %d\n", 
            sum(is.na(X_train)), sum(is.na(X_test))))

# Model 1: LASSO with CV ---------------------------------------------------

cat("\n\n=== FITTING LASSO (CV) ===\n")
set.seed(1)
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1, standardize = TRUE, 
                      penalty.factor = penalty)

lambda_cv <- cv_lasso$lambda.min
cat(sprintf("Optimal lambda (CV): %.6f\n", lambda_cv))

# Predictions
pred_train_cv <- predict(cv_lasso, newx = X_train, s = "lambda.min")
pred_test_cv <- predict(cv_lasso, newx = X_test, s = "lambda.min")

# Evaluate
evaluate_model(y_train, pred_train_cv, "LASSO-CV In-Sample")
results_cv <- evaluate_model(y_test, pred_test_cv, "LASSO-CV Out-of-Sample")

# Model 2: LASSO with BIC --------------------------------------------------

cat("\n\n=== FITTING LASSO (BIC) ===\n")
fit_lasso <- glmnet(X_train, y_train, alpha = 1, standardize = TRUE, 
                    penalty.factor = penalty)

bic_values <- get_bic(fit_lasso, X_train, y_train)
lambda_bic <- fit_lasso$lambda[which.min(bic_values)]
cat(sprintf("Optimal lambda (BIC): %.6f\n", lambda_bic))

# Selected variables
coef_bic <- coef(fit_lasso, s = lambda_bic)
selected_vars_bic <- rownames(coef_bic)[which(coef_bic != 0)]
selected_vars_bic <- selected_vars_bic[selected_vars_bic != "(Intercept)"]
cat(sprintf("Variables selected: %d\n", length(selected_vars_bic)))

# Predictions
pred_train_bic <- predict(fit_lasso, newx = X_train, s = lambda_bic)
pred_test_bic <- predict(fit_lasso, newx = X_test, s = lambda_bic)

# Evaluate
evaluate_model(y_train, pred_train_bic, "LASSO-BIC In-Sample")
results_bic <- evaluate_model(y_test, pred_test_bic, "LASSO-BIC Out-of-Sample")

# Model 3: Post-LASSO OLS (BIC) --------------------------------------------

cat("\n\n=== FITTING POST-LASSO (BIC) ===\n")

# Use BIC-selected variables
X_train_post_bic <- as.data.frame(X_train[, selected_vars_bic, drop = FALSE])
X_test_post_bic <- as.data.frame(X_test[, selected_vars_bic, drop = FALSE])

# Fit OLS on selected variables
postlasso_fit_bic <- lm(y_train ~ ., data = X_train_post_bic)
cat(sprintf("Post-LASSO (BIC) selected %d variables\n", length(selected_vars_bic)))

# Predictions
pred_train_post_bic <- predict(postlasso_fit_bic, newdata = X_train_post_bic)
pred_test_post_bic <- predict(postlasso_fit_bic, newdata = X_test_post_bic)

# Evaluate
evaluate_model(y_train, pred_train_post_bic, "Post-LASSO (BIC) In-Sample")
results_post_bic <- evaluate_model(y_test, pred_test_post_bic, "Post-LASSO (BIC) Out-of-Sample")

# Model 4: Post-LASSO OLS (CV) ---------------------------------------------

cat("\n\n=== FITTING POST-LASSO (CV) ===\n")

# Use CV-selected variables
coef_cv <- coef(cv_lasso, s = "lambda.min")
selected_vars_cv <- rownames(coef_cv)[which(coef_cv != 0)]
selected_vars_cv <- selected_vars_cv[selected_vars_cv != "(Intercept)"]

X_train_post_cv <- as.data.frame(X_train[, selected_vars_cv, drop = FALSE])
X_test_post_cv <- as.data.frame(X_test[, selected_vars_cv, drop = FALSE])

# Fit OLS on selected variables
postlasso_fit_cv <- lm(y_train ~ ., data = X_train_post_cv)
cat(sprintf("Post-LASSO (CV) selected %d variables\n", length(selected_vars_cv)))

# Predictions
pred_train_post_cv <- predict(postlasso_fit_cv, newdata = X_train_post_cv)
pred_test_post_cv <- predict(postlasso_fit_cv, newdata = X_test_post_cv)

# Evaluate
evaluate_model(y_train, pred_train_post_cv, "Post-LASSO (CV) In-Sample")
results_post_cv <- evaluate_model(y_test, pred_test_post_cv, "Post-LASSO (CV) Out-of-Sample")

# Model 5: Adaptive LASSO --------------------------------------------------

cat("\n\n=== FITTING ADAPTIVE LASSO ===\n")

# Ridge for initial weights
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0)
coefs_ini <- as.vector(coef(cv_ridge, s = cv_ridge$lambda.min))[-1]

# Adaptive weights
w <- 1 / abs(coefs_ini)^1
w[is.infinite(w)] <- 1e6

# Fit adaptive LASSO
cv_adalasso <- cv.glmnet(X_train, y_train, alpha = 1, penalty.factor = w, 
                         standardize = TRUE)

# Predictions
pred_train_ada <- predict(cv_adalasso, newx = X_train, s = "lambda.min")
pred_test_ada <- predict(cv_adalasso, newx = X_test, s = "lambda.min")

# Evaluate
evaluate_model(y_train, pred_train_ada, "Adaptive LASSO In-Sample")
results_ada <- evaluate_model(y_test, pred_test_ada, "Adaptive LASSO Out-of-Sample")

# Diagnostics --------------------------------------------------------------

cat("\n\n=== DIAGNOSTICS (LASSO-CV) ===\n")

test_resids <- y_test - pred_test_cv
cat(sprintf("Mean residual: %.6f\n", mean(test_resids)))
cat(sprintf("SD residuals: %.4f | SD y_test: %.4f\n", 
            sd(test_resids), sd(y_test)))

# Durbin-Watson statistic
dw_stat <- sum(diff(test_resids)^2) / sum(test_resids^2)
cat(sprintf("Durbin-Watson: %.4f\n", dw_stat))

# Naive benchmark
naive_forecast <- c(NA, y_test[-length(y_test)])
naive_cor <- cor(naive_forecast[-1], y_test[-1], use = "complete.obs")
model_cor <- cor(pred_test_cv, y_test, use = "complete.obs")
cat(sprintf("Naive correlation: %.4f | Model correlation: %.4f\n", 
            naive_cor, model_cor))

# Plots --------------------------------------------------------------------

par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))

# 1. Actual vs Predicted
plot(y_test, pred_test_cv, 
     main = 'OOS: Actual vs Predicted (LASSO-CV)',
     xlab = 'Actual', ylab = 'Predicted',
     pch = 20, col = 'blue')
abline(0, 1, col = 'red', lty = 2)

# 2. Residuals over time
plot(test_data$sasdate, test_resids,
     main = 'OOS Residuals Over Time',
     xlab = 'Date', ylab = 'Residual',
     type = 'l', col = 'blue')
abline(h = 0, col = 'red', lty = 2)

# 3. ACF of residuals
acf(test_resids, main = 'ACF of OOS Residuals')

# 4. Residual histogram
hist(test_resids, breaks = 20, 
     main = 'Distribution of OOS Residuals',
     xlab = 'Residual', col = 'lightblue')

par(mfrow = c(1, 1))

# Summary comparison -------------------------------------------------------

cat("\n\n=== MODEL COMPARISON (OOS) ===\n")
comparison <- data.frame(
  Model = c("LASSO-CV", "LASSO-BIC", "Post-LASSO (BIC)", "Post-LASSO (CV)", "Adaptive LASSO"),
  RMSE = round(c(results_cv$rmse, results_bic$rmse, results_post_bic$rmse, 
                 results_post_cv$rmse, results_ada$rmse), 4),
  MAE = round(c(results_cv$mae, results_bic$mae, results_post_bic$mae, 
                results_post_cv$mae, results_ada$mae), 4),
  R2 = round(c(results_cv$r2, results_bic$r2, results_post_bic$r2, 
               results_post_cv$r2, results_ada$r2), 4)
)
print(comparison)
