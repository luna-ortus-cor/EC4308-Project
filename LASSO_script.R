#----------------------------
# Load libraries
#----------------------------
library(readr)
library(dplyr)
library(zoo)
library(glmnet)
library(stringr)
library(tibble)
library(tidyr)

#############################
# LASSO
#############################

#----------------------------
# Load & clean data
#----------------------------
raw_data <- read_csv("2025-09-MD.csv")
data <- raw_data %>%
  mutate(
    INDPRO_growth = 100 * (INDPRO / lag(INDPRO) - 1),
    sasdate = as.Date(sasdate, format = "%m/%d/%Y")
  ) %>%
  select(-ACOGNO) %>%
  filter(!is.na(INDPRO_growth), sasdate >= as.Date("1978-01-01")) %>%
  filter(!(sasdate %in% as.Date(c("2025-07-01", "2025-08-01"))))

# Define variables
target_var <- "INDPRO_growth"
exclude_vars <- c("sasdate", "INDPRO", "INDPRO_growth")  # exclude target
predictor_vars <- setdiff(names(data), exclude_vars)

# Interpolate missing values in predictors only
data <- data %>% 
  arrange(sasdate) %>%
  mutate(across(all_of(predictor_vars), ~ na.approx(., na.rm = FALSE))) %>%
  drop_na()

#----------------------------
# define train/test split dates
#----------------------------
train_end <- as.Date("2015-12-01")
oos_start <- as.Date("2016-01-01")

#----------------------------
# create additional variables
#----------------------------
data <- data %>%
  mutate(
    BAA_AAA = BAA - AAA,
    T10Y3M = GS10 - TB3MS,
    COVID_dummy = ifelse(sasdate >= as.Date("2020-03-01") & sasdate <= as.Date("2020-06-01"), 1, 0)
  )

# Update predictor list to include new variables
predictor_vars <- setdiff(names(data), exclude_vars)

cat("Number of predictors before differencing:", length(predictor_vars), "\n")
cat("Checking if target is in predictors:", "INDPRO_growth" %in% predictor_vars, "\n")
cat("Checking if INDPRO is in predictors:", "INDPRO" %in% predictor_vars, "\n")

#----------------------------
# Variable transformation for stationarity
#----------------------------
diff_pad <- function(x) c(NA, diff(x))

# Variables to NOT difference
exclude_from_diff <- c("COVID_dummy", "INDPRO_growth", "INDPRO")
to_diff_vars <- setdiff(predictor_vars, exclude_from_diff)

cat("\nVariables to be differenced:", length(to_diff_vars), "\n")

# Apply differencing only to predictor variables
data <- data %>% 
  mutate(across(all_of(to_diff_vars), diff_pad, .names = "{.col}")) %>% 
  drop_na()

# Refresh predictor list after differencing
predictor_vars <- setdiff(names(data), exclude_vars)

cat("Number of predictors after differencing:", length(predictor_vars), "\n")

#----------------------------
# Create lagged predictors
#----------------------------
# Add INDPRO_growth_lag1, INDPRO_growth_lag2, INDPRO_growth_lag3 
for (lag in 1:3) {
  lag_name <- paste0("INDPRO_growth_lag", lag)
  data[[lag_name]] <- dplyr::lag(data[["INDPRO_growth"]], lag)
}
base_ar_cols <- paste0("INDPRO_growth_lag", 1:3)
# Add those to predictor_vars
predictor_vars <- c(predictor_vars, base_ar_cols)


max_lags <- 12

create_lagged_data <- function(data, predictor_vars, max_lags, target_var, skip_vars = base_ar_cols) {
  lagged_data <- data %>% select(sasdate, all_of(target_var))
  for (var in predictor_vars) {
    if (var %in% skip_vars) {
      lagged_data[[var]] <- data[[var]]
      # Do NOT create further lags
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


lagged_dataset <- create_lagged_data(data, predictor_vars, max_lags, target_var)
#because of date and target
cat("\nTotal lagged features created:", ncol(lagged_dataset) - 2, "\n") 

#----------------------------
# Split train/test
#----------------------------
train_data <- lagged_dataset %>% filter(sasdate <= train_end)
test_data  <- lagged_dataset %>% filter(sasdate >= oos_start)

X_train <- train_data %>% select(-sasdate, -INDPRO_growth) %>% as.matrix()
y_train <- train_data$INDPRO_growth

X_test  <- test_data %>% select(-sasdate, -INDPRO_growth) %>% as.matrix()
y_test  <- test_data$INDPRO_growth

base_ar_cols <- c("INDPRO_growth_lag1", "INDPRO_growth_lag2", "INDPRO_growth_lag3")
base_ar_indices <- which(colnames(X_train) %in% base_ar_cols)
penalty <- rep(1, ncol(X_train))
penalty[base_ar_indices] <- 0    




# Verification checks
stopifnot(nrow(X_train) == length(y_train))
stopifnot(nrow(X_test) == length(y_test))
stopifnot(identical(colnames(X_train), colnames(X_test)))

cat("\nData quality checks:\n")
cat("NAs in X_train:", sum(is.na(X_train)), "\n")
cat("NAs in y_train:", sum(is.na(y_train)), "\n")
cat("NAs in X_test:", sum(is.na(X_test)), "\n")
cat("NAs in y_test:", sum(is.na(y_test)), "\n")
cat("Train obs:", nrow(X_train), "Test obs:", nrow(X_test), "\n")

# Check for INDPRO leakage
indpro_cols <- grep("INDPRO", colnames(X_train), value = TRUE)
cat("\nINDPRO-related columns in predictors:", paste(indpro_cols, collapse = ", "), "\n")

#----------------------------
# Fit LASSO with BIC selection
#----------------------------
fit_lasso <- glmnet(X_train, y_train, alpha = 1, standardize = TRUE, penalty.factor = penalty)

# BIC calculation
get_bic <- function(fit, X, y) {
  n <- length(y)
  preds <- predict(fit, X)
  rss <- colSums((y - preds)^2)
  df <- fit$df  # Number of nonzero coefficients
  bic <- n * log(rss / n) + df * log(n)
  return(bic)
}

bic_values <- get_bic(fit_lasso, X_train, y_train)
lambda_bic <- fit_lasso$lambda[which.min(bic_values)]

cat("\nOptimal lambda (BIC):", lambda_bic, "\n")

#----------------------------
# Extract selected variables and coefficients
#----------------------------
coef_bic <- coef(fit_lasso, s = lambda_bic)
selected_vars_bic <- rownames(coef_bic)[which(coef_bic != 0)]
selected_vars_bic <- selected_vars_bic[selected_vars_bic != "(Intercept)"]

cat("Number of variables selected by BIC:", length(selected_vars_bic), "\n")
cat("\nSelected variables:\n")
print(selected_vars_bic)

#----------------------------
# In-sample performance
#----------------------------
pred_train <- predict(fit_lasso, newx = X_train, s = lambda_bic)
train_rmse <- sqrt(mean((y_train - pred_train)^2))
train_r2 <- 1 - sum((y_train - pred_train)^2) / sum((y_train - mean(y_train))^2)

cat("\n=== IN-SAMPLE PERFORMANCE ===\n")
cat("R-squared:", round(train_r2, 4), "\n")
cat("RMSE:", round(train_rmse, 4), "\n")

#----------------------------
# OOS forecast
#----------------------------
pred_test <- predict(fit_lasso, newx = X_test, s = lambda_bic)
test_rmse <- sqrt(mean((y_test - pred_test)^2))
test_r2 <- 1 - sum((y_test - pred_test)^2) / sum((y_test - mean(y_test))^2)

cat("\n=== OOS PERFORMANCE ===\n")
cat("R-squared:", round(test_r2, 4), "\n")
cat("RMSE:", round(test_rmse, 4), "\n")

#----------------------------
# Leakage tests
#----------------------------
cat("\n=== LEAKAGE TESTS ===\n")

# Test 1: Naive forecast benchmark
naive_forecast <- c(NA, y_test[-length(y_test)])
naive_cor <- cor(naive_forecast[-1], y_test[-1], use = "complete.obs")
cat("Naive forecast correlation (should be moderate):", round(naive_cor, 4), "\n")

# Test 2: Model prediction correlation
model_cor <- cor(pred_test, y_test, use = "complete.obs")
cat("Model prediction correlation:", round(model_cor, 4), "\n")

# Test 3: model_cor >> naive_cor it will be a problem
model_cor > 0.9 && test_r2 > 0.9

#----------------------------
# Diagnostic plots
#----------------------------
par(mfrow = c(2, 2))

# Plot 1: Actual vs Predicted (test)
plot(y_test, pred_test, 
     main = 'Out-of-Sample: Actual vs Predicted',
     xlab = 'Actual', ylab = 'Predicted',
     pch = 20, col = 'blue')
abline(0, 1, col = 'red', lty = 2)

# Plot 2: Residuals over time (test)
test_resids <- y_test - pred_test
plot(test_data$sasdate, test_resids,
     main = 'Out-of-Sample Residuals Over Time',
     xlab = 'Date', ylab = 'Residual',
     type = 'l', col = 'blue')
abline(h = 0, col = 'red', lty = 2)

# Plot 3: ACF of test residuals
acf(test_resids, main = 'ACF of Out-of-Sample Residuals')

# Plot 4: Histogram of residuals
hist(test_resids, breaks = 20, 
     main = 'Distribution of Out-of-Sample Residuals',
     xlab = 'Residual', col = 'lightblue')

par(mfrow = c(1, 1))

#----------------------------
# Additional diagnostics
#----------------------------
cat("Mean of test residuals:", round(mean(test_resids), 6), "\n")
cat("SD of test residuals:", round(sd(test_resids), 4), "\n")
cat("SD of y_test:", round(sd(y_test), 4), "\n")

# Durbin-Watson test for autocorrelation (informal)
dw_stat <- sum(diff(test_resids)^2) / sum(test_resids^2)
round(dw_stat, 4)



#############################
# Post-LASSO
#############################


#----------------------------
# Select non-zero coefficients
#----------------------------


# Get the coefficient vector at your chosen lambda
coef_bic <- coef(fit_lasso, s = lambda_bic)

# The coef_bic object is a sparse matrix. To extract the predictor names:
selected_vars <- rownames(coef_bic)[which(coef_bic != 0)] # include intercept
selected_vars <- selected_vars[selected_vars != "(Intercept)"] # drop intercept
print(selected_vars)

X_train_post <- X_train[, selected_vars, drop = FALSE]
X_test_post  <- X_test[, selected_vars, drop = FALSE]

#----------------------------
# In-sample OLS
#----------------------------

X_train_post_df <- as.data.frame(X_train_post)
postlasso_fit <- lm(y_train ~ ., data = X_train_post_df)
summary(postlasso_fit)
#----------------------------
# OOS
#----------------------------

X_test_post_df <- as.data.frame(X_test_post)
postlasso_pred <- predict(postlasso_fit, newdata = X_test_post_df)
#rmse
postlasso_rmse <- sqrt(mean((y_test - postlasso_pred)^2))
round(postlasso_rmse, 4)
#r-square
postlasso_r2 <- 1 - sum((y_test - postlasso_pred)^2) / sum((y_test - mean(y_test))^2)
round(postlasso_r2, 4)

plot(y_test, postlasso_pred, 
     main='Post-LASSO: OOS Actual vs Predicted', 
     xlab='Actual', ylab='Predicted', 
     pch=20, col='blue')
abline(0, 1, col='red', lty=2)



#############################
# Adaptive LASSO
#############################

#----------------------------
# Ridge as initial estimator
#----------------------------

# X_train and y_train are your predictors and target
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0) 
coefs_ini <- as.vector(coef(cv_ridge, s = cv_ridge$lambda.min))[-1] # Drop intercept

#----------------------------
# Adaptive weights
#----------------------------
w <- 1 / abs(coefs_ini)^1 # gamma = 1
w[is.infinite(w)] <- 1e6

#----------------------------
# Run LASSO
#----------------------------
cv_adalasso <- cv.glmnet(
  X_train, y_train,
  alpha = 1,
  penalty.factor = w,
  standardize = TRUE
)

#----------------------------
# OOS
#----------------------------

postlasso_pred <- predict(cv_adalasso, newx = X_test, s = "lambda.min")

#----------------------------
# Evaluate results
#----------------------------

summary(postlasso_pred)        
head(postlasso_pred)           
# RMSE
rmse <- sqrt(mean((y_test - postlasso_pred)^2))
round(rmse, 4)

# OOS R-squared
r2 <- 1 - sum((y_test - postlasso_pred)^2) / sum((y_test - mean(y_test))^2)
round(r2, 4)




