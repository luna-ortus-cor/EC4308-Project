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
    T10Y3M = GS10 - TB3MS
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
max_lags <- 4
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
cat(sprintf("Train obs: %d | Test obs: %d\n", nrow(X_train), nrow(X_test)))
cat(sprintf("NAs in train: %d | NAs in test: %d\n", 
            sum(is.na(X_train)), sum(is.na(X_test))))

# Model 1: LASSO with Rolling CV -------------------------------------------

# Helper function to perform rolling CV with glmnet and get 1se lambda
rolling_cv_lasso <- function(X, y, initial_window, horizon, alpha=1, penalty.factor=NULL) {
  n <- nrow(X)
  fold_results <- list()
  lambdas <- NULL
  
  for (start in seq(initial_window, n - horizon, by = horizon)) {  # FIX: Add by = horizon
    train_index <- 1:start
    test_index <- (start + 1):min(start + horizon, n)  # FIX: Ensure we don't exceed n
    
    X_train <- X[train_index, , drop=FALSE]
    y_train <- y[train_index]
    
    X_test <- X[test_index, , drop=FALSE]
    y_test <- y[test_index]
    
    # Fit glmnet on training
    fit <- glmnet(X_train, y_train, alpha=alpha, standardize=TRUE, penalty.factor=penalty.factor)
    
    # If first fold, store lambda sequence
    if (is.null(lambdas)) lambdas <- fit$lambda
    
    # Compute predictions on test for each lambda
    preds <- predict(fit, newx=X_test, s=lambdas)
    mse <- colMeans((matrix(y_test, nrow=length(y_test), ncol=length(lambdas)) - preds)^2)
    
    fold_results[[length(fold_results)+1]] <- mse
  }
  
  # Average MSE per lambda over all folds
  avg_mse <- Reduce("+", fold_results) / length(fold_results)
  
  # Identify lambda with minimum avg MSE and lambda at 1se rule
  lambda_min_idx <- which.min(avg_mse)
  lambda_min <- lambdas[lambda_min_idx]
  
  # Find lambda_1se: largest lambda with mse within 1 se of the minimum
  se <- sd(sapply(fold_results, function(x) x[lambda_min_idx])) / sqrt(length(fold_results))  # FIX: SE of min lambda
  mse_min <- min(avg_mse)
  lambda_1se <- max(lambdas[avg_mse <= mse_min + se])
  
  return(list(lambda_min=lambda_min, lambda_1se=lambda_1se, avg_mse=avg_mse, lambdas=lambdas))
}

set.seed(1)
initial_window <- floor(0.6 * nrow(X_train))
horizon <- 1  # One-step ahead forecasting

cv_results <- rolling_cv_lasso(X_train, y_train, initial_window, horizon, alpha=1, penalty.factor=penalty)

# Use 1se lambda for final model
lambda_1se <- cv_results$lambda_1se
cat(sprintf("Optimal lambda (1se rolling CV): %.6f\n", lambda_1se))

# Fit final model on the whole training set using lambda_1se
final_lasso <- glmnet(X_train, y_train, alpha=1, standardize=TRUE, penalty.factor=penalty)
pred_train_rolling <- predict(final_lasso, newx=X_train, s=lambda_1se)
pred_test_rolling <- predict(final_lasso, newx=X_test, s=lambda_1se)

# Evaluate
evaluate_model(y_train, pred_train_rolling, "LASSO Rolling CV In-Sample")
results_rolling <- evaluate_model(y_test, pred_test_rolling, "LASSO Rolling CV Out-of-Sample")

# Coefficients at lambda_1se
coef_rolling <- coef(final_lasso, s=lambda_1se)
selected_vars_rolling <- rownames(coef_rolling)[which(coef_rolling != 0)]
selected_vars_rolling <- selected_vars_rolling[selected_vars_rolling != "(Intercept)"]

cat("Variables selected by LASSO (Rolling CV 1se lambda):\n")
print(selected_vars_rolling)


# Model 2: LASSO with BIC --------------------------------------------------

fit_lasso <- glmnet(X_train, y_train, alpha = 1, standardize = TRUE, 
                    penalty.factor = penalty)

bic_values <- get_bic(fit_lasso, X_train, y_train)
lambda_bic <- fit_lasso$lambda[which.min(bic_values)]
cat(sprintf("Optimal lambda (BIC): %.6f\n", lambda_bic))

# Predictions
pred_train_bic <- predict(fit_lasso, newx = X_train, s = lambda_bic)
pred_test_bic <- predict(fit_lasso, newx = X_test, s = lambda_bic)

# Evaluate
evaluate_model(y_train, pred_train_bic, "LASSO-BIC In-Sample")
results_bic <- evaluate_model(y_test, pred_test_bic, "LASSO-BIC Out-of-Sample")

# Coefficients at lambda selected by BIC
coef_bic <- coef(fit_lasso, s = lambda_bic)
selected_vars_bic <- rownames(coef_bic)[which(coef_bic != 0)]
selected_vars_bic <- selected_vars_bic[selected_vars_bic != "(Intercept)"]

cat("Variables selected by LASSO (BIC):\n")
print(selected_vars_bic)


# Model 3: Post-LASSO OLS (BIC) -------------------

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


# Model 4: Post-LASSO OLS (Rolling CV) -------------------------------------

# Use Rolling CV (1se) selected variables from Model 1
selected_vars_cv <- selected_vars_rolling

X_train_post_cv <- as.data.frame(X_train[, selected_vars_cv, drop = FALSE])
X_test_post_cv <- as.data.frame(X_test[, selected_vars_cv, drop = FALSE])

# Fit OLS on selected variables
postlasso_fit_cv <- lm(y_train ~ ., data = X_train_post_cv)
cat(sprintf("\nPost-LASSO (Rolling CV) selected %d variables\n", length(selected_vars_cv)))

# Predictions
pred_train_post_cv <- predict(postlasso_fit_cv, newdata = X_train_post_cv)
pred_test_post_cv <- predict(postlasso_fit_cv, newdata = X_test_post_cv)

# Evaluate
evaluate_model(y_train, pred_train_post_cv, "Post-LASSO (Rolling CV) In-Sample")
results_post_cv <- evaluate_model(y_test, pred_test_post_cv, "Post-LASSO (Rolling CV) Out-of-Sample")


# Model 5: Adaptive LASSO with Rolling CV ----------------------------------

# Step 1: Get initial weights from Ridge (using rolling CV to avoid leakage)
set.seed(1)
cv_results_ridge <- rolling_cv_lasso(X_train, y_train, initial_window, horizon, 
                                     alpha=0, penalty.factor=penalty)
lambda_ridge <- cv_results_ridge$lambda_1se

# Fit ridge with selected lambda
ridge_fit <- glmnet(X_train, y_train, alpha=0, standardize=TRUE)
coefs_ini <- as.vector(coef(ridge_fit, s=lambda_ridge))[-1]

# Step 2: Adaptive weights
w <- 1 / abs(coefs_ini)^1
w[is.infinite(w)] <- 1e6

# Step 3: Fit adaptive LASSO using rolling CV
set.seed(1)
cv_results_ada <- rolling_cv_lasso(X_train, y_train, initial_window, horizon, 
                                   alpha=1, penalty.factor=w)
lambda_ada <- cv_results_ada$lambda_1se
cat(sprintf("Adaptive LASSO lambda (1se rolling CV): %.6f\n", lambda_ada))

# Fit final adaptive LASSO
ada_fit <- glmnet(X_train, y_train, alpha=1, penalty.factor=w, standardize=TRUE)

# Predictions
pred_train_ada <- predict(ada_fit, newx=X_train, s=lambda_ada)
pred_test_ada <- predict(ada_fit, newx=X_test, s=lambda_ada)

# Evaluate
evaluate_model(y_train, pred_train_ada, "Adaptive LASSO In-Sample")
results_ada <- evaluate_model(y_test, pred_test_ada, "Adaptive LASSO Out-of-Sample")


# Diagnostics --------------------------------------------------------------

test_resids <- y_test - pred_test_rolling  # Using Model 1 predictions
cat(sprintf("\nMean residual: %.6f\n", mean(test_resids)))
cat(sprintf("SD residuals: %.4f | SD y_test: %.4f\n", 
            sd(test_resids), sd(y_test)))

# Durbin-Watson statistic
dw_stat <- sum(diff(test_resids)^2) / sum(test_resids^2)
cat(sprintf("Durbin-Watson: %.4f\n", dw_stat))

# Naive benchmark
naive_forecast <- c(NA, y_test[-length(y_test)])
naive_cor <- cor(naive_forecast[-1], y_test[-1], use = "complete.obs")
model_cor <- cor(pred_test_rolling, y_test, use = "complete.obs")
cat(sprintf("Naive correlation: %.4f | Model correlation: %.4f\n", 
            naive_cor, model_cor))


# Plots --------------------------------------------------------------------

par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))

# 1. Actual vs Predicted
plot(y_test, pred_test_rolling, 
     main = 'OOS: Actual vs Predicted (LASSO Rolling CV)',
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

comparison <- data.frame(
  Model = c("LASSO-BIC", "LASSO-Rolling-CV (1se)", 
            "Post-LASSO (BIC)", "Post-LASSO (Rolling CV)", "Adaptive LASSO"),
  RMSE = round(c(results_bic$rmse, results_cv$rmse, 
                 results_post_bic$rmse, results_post_cv$rmse, results_ada$rmse), 4),
  MAE = round(c(results_bic$mae, results_cv$mae, 
                results_post_bic$mae, results_post_cv$mae, results_ada$mae), 4),
  R2 = round(c(results_bic$r2, results_cv$r2, 
               results_post_bic$r2, results_post_cv$r2, results_ada$r2), 4)
)

cat("\n")
print(comparison)