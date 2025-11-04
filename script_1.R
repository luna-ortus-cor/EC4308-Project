# set working directory
setwd("C:/Users/russn/Downloads/")

library(readr)
library(dplyr)
library(zoo)

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



library(pls)
library(ggplot2)
# =============================
# DEFINE IN-/OUT-SAMPLE SPLIT
# =============================
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

# =============================
# 1️⃣ PRINCIPAL COMPONENT ANALYSIS (PCA)
# =============================
pca_result <- prcomp(X_train_scaled, scale. = FALSE)
summary(pca_result)

# Scree plot (variance explained)
var_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
p <- ggplot(data.frame(PC = 1:length(var_explained),
                  Variance = var_explained),
       aes(x = PC, y = Variance)) +
  geom_line() + geom_point() +
  theme_minimal() +
  ggtitle("Scree Plot of PCA") +
  ylab("Proportion of Variance Explained")
print(p)

# =============================
# 2️⃣ PRINCIPAL COMPONENT REGRESSION (PCR)
# =============================
set.seed(123)
pcr_model <- pcr(y_train ~ X_train_scaled, validation = "CV")

# Optimal number of components (min RMSEP)
best_ncomp_pcr <- which.min(RMSEP(pcr_model)$val[1, , -1])
cat("Optimal number of components for PCR:", best_ncomp_pcr, "\n")

# Cross-validation plot
validationplot(pcr_model, val.type = "MSEP")

# Predictions
pcr_pred_in  <- predict(pcr_model, ncomp = best_ncomp_pcr)
pcr_pred_out <- predict(pcr_model, newdata = data.frame(X_train_scaled = X_test_scaled),
                        ncomp = best_ncomp_pcr)

# =============================
# 3️⃣ PARTIAL LEAST SQUARES (PLS)
# =============================
set.seed(123)
pls_model <- plsr(y_train ~ X_train_scaled, validation = "CV")

best_ncomp_pls <- which.min(RMSEP(pls_model)$val[1, , -1])
cat("Optimal number of components for PLS:", best_ncomp_pls, "\n")

validationplot(pls_model, val.type = "MSEP")

pls_pred_in  <- predict(pls_model, ncomp = best_ncomp_pls)
pls_pred_out <- predict(pls_model, newdata = data.frame(X_train_scaled = X_test_scaled),
                        ncomp = best_ncomp_pls)

# =============================
# 4️⃣ PERFORMANCE EVALUATION
# =============================

# Helper functions
mse  <- function(actual, pred) mean((actual - pred)^2)
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
mae  <- function(actual, pred) mean(abs(actual - pred))

# --- PCR metrics ---
mse_pcr_in  <- mse(y_train, pcr_pred_in)
mse_pcr_out <- mse(y_test,  pcr_pred_out)
rmse_pcr_in <- rmse(y_train, pcr_pred_in)
rmse_pcr_out <- rmse(y_test, pcr_pred_out)

# --- PLS metrics ---
mse_pls_in  <- mse(y_train, pls_pred_in)
mse_pls_out <- mse(y_test,  pls_pred_out)
rmse_pls_in <- rmse(y_train, pls_pred_in)
rmse_pls_out <- rmse(y_test, pls_pred_out)

# --- Combine results ---
results <- data.frame(
  Model = c("PCR", "PLS"),
  ncomp = c(best_ncomp_pcr, best_ncomp_pls),
  MSE_in = c(mse_pcr_in, mse_pls_in),
  MSE_out = c(mse_pcr_out, mse_pls_out),
  RMSE_in = c(rmse_pcr_in, rmse_pls_in),
  RMSE_out = c(rmse_pcr_out, rmse_pls_out)
)
print(results)

# =============================
# 5️⃣ (OPTIONAL) ADDITIONAL METRICS
# =============================

# Mean Absolute Error (MAE)
MAE_in  <- c(mae(y_train, pcr_pred_in), mae(y_train, pls_pred_in))
MAE_out <- c(mae(y_test, pcr_pred_out), mae(y_test, pls_pred_out))

results$MAE_in  <- MAE_in
results$MAE_out <- MAE_out

# R-squared (in-sample only)
r2_in <- c(
  1 - sum((y_train - pcr_pred_in)^2) / sum((y_train - mean(y_train))^2),
  1 - sum((y_train - pls_pred_in)^2) / sum((y_train - mean(y_train))^2)
)
results$R2_in <- r2_in

print(results)

# =============================
# 6️⃣ NAÏVE BENCHMARK & FORECAST COMPARISON (FIXED)
# =============================

library(forecast)

# --- Naïve forecast: previous month's growth ---
y_full <- data$INDPRO_growth
y_test_naive <- lag(y_full, 1)[data$sasdate >= as.Date("2016-01-01")]

# Remove first NA
valid_idx <- !is.na(y_test_naive)
y_test_valid <- y_test[valid_idx]
y_test_naive <- y_test_naive[valid_idx]

# Fix predictions (ensure same length)
pcr_pred_out_valid <- as.numeric(pcr_pred_out[valid_idx])
pls_pred_out_valid <- as.numeric(pls_pred_out[valid_idx])

# --- Benchmark MSE ---
mse_naive_out <- mean((y_test_valid - y_test_naive)^2)
cat("Naïve forecast MSE (out-of-sample):", mse_naive_out, "\n")

# --- Theil’s U ---
TheilsU_pcr <- sqrt(mse_pcr_out / mse_naive_out)
TheilsU_pls <- sqrt(mse_pls_out / mse_naive_out)

cat("Theil's U (PCR):", TheilsU_pcr, "\n")
cat("Theil's U (PLS):", TheilsU_pls, "\n")

# =============================
# 7️⃣ DIEBOLD–MARIANO TEST (FIXED)
# =============================

# Helper wrapper to safely compute both statistic and p-value
dm_test_safe <- function(e1, e2, h = 1, alternative = "two.sided") {
  out <- tryCatch({
    res <- forecast::dm.test(e1, e2, h = h, power = 2, alternative = alternative)
    list(statistic = res$statistic, p.value = res$p.value)
  }, error = function(e) list(statistic = NA, p.value = NA))
  return(out)
}

# Forecast errors
err_pcr <- y_test_valid - pcr_pred_out_valid
err_pls <- y_test_valid - pls_pred_out_valid
err_naive <- y_test_valid - y_test_naive

# Run DM tests
dm_pcr_vs_naive <- dm_test_safe(err_pcr, err_naive)
dm_pls_vs_naive <- dm_test_safe(err_pls, err_naive)
dm_pcr_vs_pls   <- dm_test_safe(err_pcr, err_pls)

cat("\n--- Diebold–Mariano test results ---\n")
cat("PCR vs Naïve: p-value =", dm_pcr_vs_naive$p.value, "\n")
cat("PLS vs Naïve: p-value =", dm_pls_vs_naive$p.value, "\n")
cat("PCR vs PLS:   p-value =", dm_pcr_vs_pls$p.value, "\n")

# =============================
# 8️⃣ FINAL SUMMARY TABLE
# =============================

final_results <- results %>%
  mutate(
    TheilU = c(TheilsU_pcr, TheilsU_pls),
    DM_vs_Naive_p = c(dm_pcr_vs_naive$p.value, dm_pls_vs_naive$p.value)
  )

print(final_results)

# =====================================================
# 1️⃣ ROLLING (EXPANDING-WINDOW) ONE-STEP-AHEAD FORECAST
# =====================================================

library(pls)

# --- Settings ---
start_year <- 2015   # initial training ends here
forecast_horizon <- 1  # 1-step-ahead forecast
dates <- data$sasdate

# --- Storage for forecasts ---
rolling_forecasts_pcr <- c()
rolling_forecasts_pls <- c()
rolling_actuals <- c()
rolling_dates <- c()

# --- Prepare predictor matrices ---
X_full <- data %>% dplyr::select(-sasdate, -INDPRO, -INDPRO_growth)
y_full <- data$INDPRO_growth
X_full_scaled <- scale(X_full)

# Loop through each forecast period from 2016 onward
for (t in which(format(dates, "%Y") >= "2016")) {
  # Define expanding window up to t-1
  train_idx <- 1:(t - forecast_horizon)
  test_idx <- t
  
  if (length(train_idx) < 60) next  # skip too-small samples early on
  
  X_train <- X_full_scaled[train_idx, , drop = FALSE]
  y_train <- y_full[train_idx]
  
  X_test <- X_full_scaled[test_idx, , drop = FALSE]
  y_test <- y_full[test_idx]
  
  # Fit PCR model (can fix number of components or reselect each time)
  pcr_fit <- pcr(y_train ~ X_train, validation = "none", ncomp = 5)
  pls_fit <- plsr(y_train ~ X_train, validation = "none", ncomp = 5)
  
  # Forecast 1-step ahead
  pred_pcr <- predict(pcr_fit, newdata = X_test, ncomp = 5)
  pred_pls <- predict(pls_fit, newdata = X_test, ncomp = 5)
  
  # Store results
  rolling_forecasts_pcr <- c(rolling_forecasts_pcr, pred_pcr)
  rolling_forecasts_pls <- c(rolling_forecasts_pls, pred_pls)
  rolling_actuals <- c(rolling_actuals, y_test)
  rolling_dates <- c(rolling_dates, dates[test_idx])
}

# --- Evaluate rolling performance ---
mse_roll_pcr <- mean((rolling_actuals - rolling_forecasts_pcr)^2)
mse_roll_pls <- mean((rolling_actuals - rolling_forecasts_pls)^2)

cat("\nRolling forecast MSE (PCR):", mse_roll_pcr,
    "\nRolling forecast MSE (PLS):", mse_roll_pls, "\n")

# --- Plot rolling forecasts vs actuals ---
plot_df <- data.frame(
  Date = rolling_dates,
  Actual = rolling_actuals,
  PCR = rolling_forecasts_pcr,
  PLS = rolling_forecasts_pls
)

p <- ggplot(plot_df, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = PCR, color = "PCR forecast"), linetype = "dashed") +
  geom_line(aes(y = PLS, color = "PLS forecast"), linetype = "dotted") +
  theme_minimal() +
  labs(title = "Rolling 1-step-ahead Forecasts (Expanding Window)",
       y = "INDPRO Growth (%)", color = "Series")
print(p)

# =====================================================
# 2️⃣ MULTI-STEP-AHEAD FORECASTS: STATIC & RECURSIVE
# =====================================================

library(lubridate)

# --- Settings ---
forecast_horizons <- c(1, 3, 6, 12)   # horizons in months
ncomp_use <- 5                        # number of components for PCR/PLS
forecast_start <- as.Date("2016-01-01")

# --- Prepare data ---
X_train <- X_full_scaled[dates < forecast_start, , drop = FALSE]
y_train <- y_full[dates < forecast_start]
X_test_full <- X_full_scaled[dates >= forecast_start, , drop = FALSE]
y_test_full <- y_full[dates >= forecast_start]
dates_test <- dates[dates >= forecast_start]

# --- Fit final models using in-sample data ---
pcr_final <- pcr(y_train ~ X_train, validation = "none", ncomp = ncomp_use)
pls_final <- plsr(y_train ~ X_train, validation = "none", ncomp = ncomp_use)

# -----------------------------------------------------
# A. STATIC (DIRECT) MULTI-STEP FORECASTS
# -----------------------------------------------------
multi_static <- list()

for (h in forecast_horizons) {
  start_date <- forecast_start %m+% months(h - 1)
  test_idx <- which(dates >= start_date)
  if (length(test_idx) == 0) next
  
  X_test <- X_full_scaled[test_idx, , drop = FALSE]
  y_test <- y_full[test_idx]
  
  pred_pcr <- predict(pcr_final, newdata = X_test, ncomp = ncomp_use)
  pred_pls <- predict(pls_final, newdata = X_test, ncomp = ncomp_use)
  
  mse_pcr <- mean((y_test - pred_pcr)^2)
  mse_pls <- mean((y_test - pred_pls)^2)
  
  multi_static[[paste0(h, "-month")]] <- data.frame(
    Horizon = h,
    Type = "Static",
    MSE_PCR = mse_pcr,
    MSE_PLS = mse_pls
  )
}

# -----------------------------------------------------
# B. RECURSIVE (DYNAMIC) MULTI-STEP FORECASTS
# -----------------------------------------------------
# Here, we forecast one step at a time, feeding predicted values forward.

multi_recursive <- list()

for (h in forecast_horizons) {
  y_preds_pcr <- numeric(length = h)
  y_preds_pls <- numeric(length = h)
  
  # Initialize model and predictors
  X_train_dyn <- X_train
  y_train_dyn <- y_train
  
  # Start recursive loop
  for (step in 1:h) {
    # Fit models (could update components or keep constant)
    pcr_fit <- pcr(y_train_dyn ~ X_train_dyn, validation = "none", ncomp = ncomp_use)
    pls_fit <- plsr(y_train_dyn ~ X_train_dyn, validation = "none", ncomp = ncomp_use)
    
    # Determine forecast target date
    forecast_date <- forecast_start %m+% months(step - 1)
    test_idx <- which(dates == forecast_date)
    if (length(test_idx) == 0) next
    
    X_future <- X_full_scaled[test_idx, , drop = FALSE]
    
    # Predict this step
    y_preds_pcr[step] <- predict(pcr_fit, newdata = X_future, ncomp = ncomp_use)
    y_preds_pls[step] <- predict(pls_fit, newdata = X_future, ncomp = ncomp_use)
    
    # Append predicted values to training (recursive updating)
    y_train_dyn <- c(y_train_dyn, y_preds_pcr[step])
    X_train_dyn <- rbind(X_train_dyn, X_future)
  }
  
  # Compare against actual values if available
  test_idx_all <- which(dates >= forecast_start & dates < forecast_start %m+% months(h))
  if (length(test_idx_all) > 0) {
    y_actual <- y_full[test_idx_all]
    mse_pcr <- mean((y_actual[seq_along(y_preds_pcr)] - y_preds_pcr)^2)
    mse_pls <- mean((y_actual[seq_along(y_preds_pls)] - y_preds_pls)^2)
    
    multi_recursive[[paste0(h, "-month")]] <- data.frame(
      Horizon = h,
      Type = "Recursive",
      MSE_PCR = mse_pcr,
      MSE_PLS = mse_pls
    )
  }
}

# -----------------------------------------------------
# Combine static + recursive results
# -----------------------------------------------------
multi_results <- do.call(rbind, c(multi_static, multi_recursive))
print(multi_results)

# --- Plot comparison ---
p <- ggplot(multi_results, aes(x = Horizon, y = MSE_PCR, color = Type)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(title = "PCR Multi-Step Forecast Comparison",
       y = "MSE", x = "Forecast Horizon (months)") +
  scale_color_manual(values = c("Static" = "steelblue", "Recursive" = "firebrick"))
print(p)

p <- ggplot(multi_results, aes(x = Horizon, y = MSE_PLS, color = Type)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(title = "PLS Multi-Step Forecast Comparison",
       y = "MSE", x = "Forecast Horizon (months)") +
  scale_color_manual(values = c("Static" = "steelblue", "Recursive" = "firebrick"))
print(p)

# =====================================================
# 3️⃣ ROLLING (EXPANDING-WINDOW) MULTI-STEP FORECASTS
# =====================================================

library(lubridate)
library(pls)
library(dplyr)
library(ggplot2)

# --- Settings ---
forecast_horizons <- c(1, 3, 6, 12)   # months ahead
ncomp_use <- 5                        # components for PCR/PLS
min_train_obs <- 60                   # minimum obs before forecasting
forecast_start <- as.Date("2016-01-01")

# --- Prepare full data ---
X_full <- data %>% dplyr::select(-sasdate, -INDPRO, -INDPRO_growth)
y_full <- data$INDPRO_growth
dates_full <- data$sasdate
X_full_scaled <- scale(X_full)

# --- Storage for forecast errors by horizon ---
rolling_errors_pcr <- list()
rolling_errors_pls <- list()

for (h in forecast_horizons) {
  rolling_errors_pcr[[as.character(h)]] <- c()
  rolling_errors_pls[[as.character(h)]] <- c()
}

# --- Rolling forecast loop ---
for (t in seq_len(nrow(data))) {
  current_date <- dates_full[t]
  if (current_date < forecast_start) next
  
  # Define training window: all data before current_date
  train_idx <- which(dates_full < current_date)
  if (length(train_idx) < min_train_obs) next
  
  X_train <- X_full_scaled[train_idx, , drop = FALSE]
  y_train <- y_full[train_idx]
  
  # Fit models on available data
  pcr_fit <- pcr(y_train ~ X_train, validation = "none", ncomp = ncomp_use)
  pls_fit <- plsr(y_train ~ X_train, validation = "none", ncomp = ncomp_use)
  
  # Generate forecasts for each horizon
  for (h in forecast_horizons) {
    forecast_date <- current_date %m+% months(h)
    test_idx <- which(dates_full == forecast_date)
    
    # Skip if forecast horizon exceeds available data
    if (length(test_idx) == 0) next
    
    X_test <- X_full_scaled[test_idx, , drop = FALSE]
    y_actual <- y_full[test_idx]
    
    # Forecast using fitted models
    y_pred_pcr <- predict(pcr_fit, newdata = X_test, ncomp = ncomp_use)
    y_pred_pls <- predict(pls_fit, newdata = X_test, ncomp = ncomp_use)
    
    # Store forecast errors
    rolling_errors_pcr[[as.character(h)]] <- c(
      rolling_errors_pcr[[as.character(h)]],
      as.numeric(y_actual - y_pred_pcr)
    )
    rolling_errors_pls[[as.character(h)]] <- c(
      rolling_errors_pls[[as.character(h)]],
      as.numeric(y_actual - y_pred_pls)
    )
  }
}

# --- Compute MSE by horizon ---
rolling_results <- data.frame(
  Horizon = forecast_horizons,
  MSE_PCR = sapply(rolling_errors_pcr, function(e) mean(e^2, na.rm = TRUE)),
  MSE_PLS = sapply(rolling_errors_pls, function(e) mean(e^2, na.rm = TRUE))
)
print(rolling_results)

# --- Plot results ---
p <- ggplot(rolling_results, aes(x = Horizon)) +
  geom_line(aes(y = MSE_PCR, color = "PCR")) +
  geom_line(aes(y = MSE_PLS, color = "PLS")) +
  geom_point(aes(y = MSE_PCR, color = "PCR")) +
  geom_point(aes(y = MSE_PLS, color = "PLS")) +
  theme_minimal() +
  labs(title = "Rolling Multi-Step Forecast MSEs (Expanding Window)",
       y = "Mean Squared Error",
       x = "Forecast Horizon (months)",
       color = "Model")
print(p)

# =====================================================
# 4️⃣ ROLLING MULTI-STEP FORECASTS VS. NAIVE BENCHMARK
# =====================================================

library(forecast)   # for dm.test()
library(lubridate)
library(dplyr)
library(ggplot2)
library(pls)

# --- Settings ---
forecast_horizons <- c(1, 3, 6, 12)
ncomp_use <- 5
min_train_obs <- 60
forecast_start <- as.Date("2016-01-01")

# --- Prepare full data ---
X_full <- data %>% dplyr::select(-sasdate, -INDPRO, -INDPRO_growth)
y_full <- data$INDPRO_growth
dates_full <- data$sasdate
X_full_scaled <- scale(X_full)

# --- Storage ---
rolling_errors_pcr <- list()
rolling_errors_pls <- list()
rolling_errors_naive <- list()

for (h in forecast_horizons) {
  rolling_errors_pcr[[as.character(h)]] <- c()
  rolling_errors_pls[[as.character(h)]] <- c()
  rolling_errors_naive[[as.character(h)]] <- c()
}

# --- Rolling forecast loop ---
for (t in seq_len(nrow(data))) {
  current_date <- dates_full[t]
  if (current_date < forecast_start) next
  
  # Training window (expanding)
  train_idx <- which(dates_full < current_date)
  # train_idx <- tail(which(dates_full < current_date), 120)  # last 10 years
  if (length(train_idx) < min_train_obs) next
  
  X_train <- X_full_scaled[train_idx, , drop = FALSE]
  y_train <- y_full[train_idx]
  
  # Fit models
  pcr_fit <- pcr(y_train ~ X_train, validation = "none", ncomp = ncomp_use)
  pls_fit <- plsr(y_train ~ X_train, validation = "none", ncomp = ncomp_use)
  
  # Forecast each horizon
  for (h in forecast_horizons) {
    forecast_date <- current_date %m+% months(h)
    test_idx <- which(dates_full == forecast_date)
    if (length(test_idx) == 0) next
    
    X_test <- X_full_scaled[test_idx, , drop = FALSE]
    y_actual <- y_full[test_idx]
    
    # PCR & PLS predictions
    y_pred_pcr <- predict(pcr_fit, newdata = X_test, ncomp = ncomp_use)
    y_pred_pls <- predict(pls_fit, newdata = X_test, ncomp = ncomp_use)
    
    # Naïve forecast = last observed value (random walk in growth rate)
    y_pred_naive <- y_full[t]  # previous observed growth rate
    
    # Store forecast errors
    rolling_errors_pcr[[as.character(h)]] <- c(
      rolling_errors_pcr[[as.character(h)]],
      y_actual - y_pred_pcr
    )
    rolling_errors_pls[[as.character(h)]] <- c(
      rolling_errors_pls[[as.character(h)]],
      y_actual - y_pred_pls
    )
    rolling_errors_naive[[as.character(h)]] <- c(
      rolling_errors_naive[[as.character(h)]],
      y_actual - y_pred_naive
    )
  }
}

# --- Compute MSE, Theil's U, and DM tests ---
results_list <- list()

for (h in forecast_horizons) {
  err_pcr <- rolling_errors_pcr[[as.character(h)]]
  err_pls <- rolling_errors_pls[[as.character(h)]]
  err_naive <- rolling_errors_naive[[as.character(h)]]
  
  mse_pcr <- mean(err_pcr^2, na.rm = TRUE)
  mse_pls <- mean(err_pls^2, na.rm = TRUE)
  mse_naive <- mean(err_naive^2, na.rm = TRUE)
  
  # Theil's U = sqrt(MSE_model / MSE_naive)
  theil_pcr <- sqrt(mse_pcr / mse_naive)
  theil_pls <- sqrt(mse_pls / mse_naive)
  
  # Diebold-Mariano test (two-sided)
  dm_pcr <- tryCatch(dm.test(err_pcr, err_naive, h = 1, alternative = "two.sided")$p.value,
                     error = function(e) NA)
  dm_pls <- tryCatch(dm.test(err_pls, err_naive, h = 1, alternative = "two.sided")$p.value,
                     error = function(e) NA)
  
  results_list[[as.character(h)]] <- data.frame(
    Horizon = h,
    MSE_PCR = mse_pcr,
    MSE_PLS = mse_pls,
    MSE_Naive = mse_naive,
    TheilU_PCR = theil_pcr,
    TheilU_PLS = theil_pls,
    DM_pval_PCR = dm_pcr,
    DM_pval_PLS = dm_pls
  )
}

# --- Combine results ---
benchmark_results <- do.call(rbind, results_list)
print(benchmark_results)

# --- Visualization of relative performance ---
p <- ggplot(benchmark_results, aes(x = Horizon)) +
  geom_line(aes(y = TheilU_PCR, color = "PCR")) +
  geom_line(aes(y = TheilU_PLS, color = "PLS")) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
  geom_point(aes(y = TheilU_PCR, color = "PCR")) +
  geom_point(aes(y = TheilU_PLS, color = "PLS")) +
  theme_minimal() +
  labs(title = "Theil’s U: PCR/PLS vs Naïve Forecast",
       y = "Theil’s U ( <1 = better than Naïve )",
       x = "Forecast Horizon (months)",
       color = "Model")
print(p)

# =====================================================
# 🧾 FINAL FORECAST SUMMARY TABLE (PCR, PLS vs Naïve)
# =====================================================

library(dplyr)
library(gt)
library(scales)

# --- Step 1: Combine results from all models ---

summary_static_recursive <- multi_results %>%
  mutate(Method = Type) %>%
  dplyr::select(Horizon, Method, MSE_PCR, MSE_PLS)

summary_rolling <- rolling_results %>%
  mutate(Method = "Rolling") %>%
  dplyr::select(Horizon, Method, MSE_PCR, MSE_PLS)

summary_benchmark <- benchmark_results %>%
  mutate(Method = "Rolling_vs_Naive") %>%
  dplyr::select(Horizon, Method, MSE_PCR, MSE_PLS, MSE_Naive,
         TheilU_PCR, TheilU_PLS, DM_pval_PCR, DM_pval_PLS)

# --- Step 2: Merge and format numeric precision ---
combined_table <- bind_rows(
  summary_static_recursive,
  summary_rolling,
  summary_benchmark
) %>%
  mutate(across(where(is.numeric), ~ round(., 4))) %>%
  arrange(Horizon, Method)

# --- Step 3: Add significance flags and stars ---
combined_table_sig <- combined_table %>%
  mutate(
    stars_PCR = case_when(
      DM_pval_PCR < 0.01 ~ "***",
      DM_pval_PCR < 0.05 ~ "**",
      DM_pval_PCR < 0.10 ~ "*",
      TRUE ~ ""
    ),
    stars_PLS = case_when(
      DM_pval_PLS < 0.01 ~ "***",
      DM_pval_PLS < 0.05 ~ "**",
      DM_pval_PLS < 0.10 ~ "*",
      TRUE ~ ""
    ),
    DM_pval_PCR_fmt = ifelse(!is.na(DM_pval_PCR),
                             sprintf("%.4f%s", DM_pval_PCR, stars_PCR),
                             NA),
    DM_pval_PLS_fmt = ifelse(!is.na(DM_pval_PLS),
                             sprintf("%.4f%s", DM_pval_PLS, stars_PLS),
                             NA)
  )

# --- Step 4: Build and format the gt table ---
gt_stars <- combined_table_sig %>%
  gt(rowname_col = "Horizon", groupname_col = "Method") %>%
  tab_header(
    title = md("**Forecast Performance Summary (PCR, PLS vs Naïve)**"),
    subtitle = md("_Includes Theil’s U, DM test p-values, and significance stars_")
  ) %>%
  fmt_number(
    columns = c(MSE_PCR, MSE_PLS, TheilU_PCR, TheilU_PLS, MSE_Naive),
    decimals = 4
  ) %>%
  cols_label(
    MSE_PCR = "MSE",
    TheilU_PCR = "Theil’s U",
    DM_pval_PCR_fmt = "DM p-val",
    MSE_PLS = "MSE",
    TheilU_PLS = "Theil’s U",
    DM_pval_PLS_fmt = "DM p-val",
    MSE_Naive = "Naïve MSE"
  ) %>%
  tab_spanner(label = "PCR", columns = c(MSE_PCR, TheilU_PCR, DM_pval_PCR_fmt)) %>%
  tab_spanner(label = "PLS", columns = c(MSE_PLS, TheilU_PLS, DM_pval_PLS_fmt)) %>%
  # --- Highlight significant p-values (<0.05) ---
  tab_style(
    style = list(cell_text(weight = "bold", color = "firebrick")),
    locations = cells_body(columns = c(DM_pval_PCR_fmt),
                           rows = DM_pval_PCR < 0.05)
  ) %>%
  tab_style(
    style = list(cell_text(weight = "bold", color = "steelblue")),
    locations = cells_body(columns = c(DM_pval_PLS_fmt),
                           rows = DM_pval_PLS < 0.05)
  ) %>%
  # --- Color-code Theil’s U for better-than-naïve forecasts ---
  data_color(
    columns = c(TheilU_PCR, TheilU_PLS),
    colors = col_bin(
      palette = c("#C7E9C0", "white", "#FDD0A2"),
      domain = c(0.8, 1.2)
    )
  ) %>%
  tab_source_note(
    md("_Stars: *** p < 0.01, ** p < 0.05, * p < 0.10.  
    Bold red/blue = statistically significant at 5%.  
    Green shading = Theil’s U < 1 (better than naïve forecast)._")
  )

# --- Step 5: Display and/or export ---
#print(gt_stars)
# Optionally save to file:
# gtsave(gt_stars, "Forecast_Summary_Final.html")
# gtsave(gt_stars, "Forecast_Summary_Final.pdf")


# =====================================================
# TIME SERIES ANALYSIS PIPELINE: INDPRO (1978–2025)
# =====================================================

# --- Load packages ---
library(forecast)
library(mFilter)
library(strucchange)
library(vars)
library(tseries)
library(ggplot2)
library(lubridate)
library(dplyr)
library(zoo)
library(future.apply)

# =====================================================
# 0️⃣ PREPARE DATA
# =====================================================

macro_vars <- data %>% dplyr::select(CP3Mx, COMPAPFFx, UMCSENTx, VIXCLSx)
indpro_growth <- data$INDPRO_growth

complete_cases <- complete.cases(indpro_growth, macro_vars)
ts_growth_clean <- ts(indpro_growth[complete_cases], start=c(1978,2), frequency=12)
xreg_clean <- macro_vars[complete_cases, ]

# Train/test split (1978–2015, 2016–2025)
train_ts <- window(ts_growth_clean, end=c(2015,12))
test_ts  <- window(ts_growth_clean, start=c(2016,1))
train_xreg <- xreg_clean[1:length(train_ts), ]
test_xreg  <- xreg_clean[(length(train_ts)+1):(length(train_ts)+length(test_ts)), ]

# =====================================================
# 1️⃣ DECOMPOSITION
# =====================================================

ts_indpro <- ts(data$INDPRO[complete_cases], start=c(1978,1), frequency=12)

# STL decomposition
autoplot(stl(ts_indpro, s.window="periodic")) + ggtitle("STL Decomposition of INDPRO")

# HP Filter
hp <- hpfilter(ts_indpro, freq=14400)
p <- autoplot(ts_indpro, series="Actual") +
  autolayer(hp$trend, series="HP Trend") +
  autolayer(hp$cycle, series="HP Cycle") +
  ggtitle("HP Filter: Trend and Cycle") +
  theme_minimal()
print(p)

# =====================================================
# 2️⃣ STATIONARITY & STRUCTURAL BREAKS
# =====================================================

adf_test <- adf.test(ts_growth_clean)
cat("ADF test p-value:", adf_test$p.value, "\n")

bp_test <- breakpoints(ts_growth_clean ~ 1)
summary(bp_test)
plot(bp_test, main="Structural Breaks in INDPRO Growth")
lines(fitted(bp_test, breaks=2), col="red", lwd=2)

# Chow test around breakpoint
bp_years <- time(ts_growth_clean)[breakpoints(bp_test)$breakpoints]
cat("Possible break dates:", bp_years, "\n")

# CUSUM test visualization
cusum_test <- efp(ts_growth_clean ~ 1, type="Rec-CUSUM")
plot(cusum_test, main="CUSUM Test for Structural Stability")

# =====================================================
# 3️⃣ UNIVARIATE ARIMA FORECASTS
# =====================================================

fit_arima <- auto.arima(train_ts)
summary(fit_arima)
fc_arima <- forecast(fit_arima, h=length(test_ts))

autoplot(fc_arima) +
  autolayer(test_ts, series="Actual") +
  ggtitle("ARIMA Forecast of INDPRO Growth") +
  theme_minimal()

mse_arima_out <- mean((test_ts - fc_arima$mean)^2)

# =====================================================
# 4️⃣ ARIMAX WITH PCA (Multivariate Regressors)
# =====================================================

pca_model <- prcomp(train_xreg, scale.=TRUE)
expl_var <- summary(pca_model)$importance[3,]
n_comp <- which(cumsum(expl_var) >= 0.9)[1]
train_xreg_pca <- pca_model$x[,1:n_comp]
test_xreg_scaled <- scale(test_xreg, center=pca_model$center, scale=pca_model$scale)
test_xreg_pca <- as.matrix(test_xreg_scaled) %*% pca_model$rotation[,1:n_comp]

fit_arimax <- auto.arima(train_ts, xreg=train_xreg_pca)
summary(fit_arimax)
order_fixed <- arimaorder(fit_arimax)[1:3]

fc_arimax <- forecast(fit_arimax, xreg=test_xreg_pca)
autoplot(fc_arimax) +
  autolayer(test_ts, series="Actual") +
  ggtitle("ARIMAX (PCA) Forecast") +
  theme_minimal()

mse_arimax_out <- mean((test_ts - fc_arimax$mean)^2)

# =====================================================
# 5️⃣ FAST ROLLING / RECURSIVE ARIMAX FORECAST
# =====================================================

n_forecast <- length(test_ts)
rolling_fc_arimax <- numeric(n_forecast)

plan(multisession, workers=4)  # parallel processing
rolling_fc_arimax <- future_sapply(1:n_forecast, function(i) {
  idx <- 1:(length(train_ts) + i - 1)
  train_y <- ts_growth_clean[idx]
  train_x <- as.matrix(xreg_clean[idx, ])
  train_x_pca <- scale(train_x, center=pca_model$center, scale=pca_model$scale) %*% pca_model$rotation[,1:n_comp]
  next_x <- as.matrix(test_xreg[i, ])
  next_x_pca <- scale(next_x, center=pca_model$center, scale=pca_model$scale) %*% pca_model$rotation[,1:n_comp]
  
  fit <- try(Arima(train_y, order=order_fixed, xreg=train_x_pca), silent=TRUE)
  if(inherits(fit, "try-error")) return(NA)
  fc <- forecast(fit, xreg=next_x_pca, h=1)
  return(fc$mean)
})

rolling_fc_arimax <- ts(rolling_fc_arimax, start=c(2016,1), frequency=12)
mse_roll_arimax <- mean((test_ts - rolling_fc_arimax)^2, na.rm=TRUE)
cat("Rolling ARIMAX MSE:", round(mse_roll_arimax,4), "\n")

p <- autoplot(window(ts_growth_clean, start=c(2010,1)), series="Actual") +
  autolayer(rolling_fc_arimax, series="Rolling ARIMAX") +
  ggtitle("Rolling Recursive ARIMAX Forecast") +
  theme_minimal()
print(p)

# =====================================================
# 6️⃣ MULTIVARIATE VAR FORECAST (STATIC + RECURSIVE)
# =====================================================

var_data <- cbind(INDPRO_growth=train_ts, train_xreg)
var_ts <- ts(var_data, start=c(1978,2), frequency=12)

lag_select <- VARselect(var_ts, lag.max=12, type="const")
p_var <- lag_select$selection["AIC(n)"]
fit_var <- VAR(var_ts, p=p_var, type="const")

# Static (multi-step) VAR forecast
fc_var <- predict(fit_var, n.ahead=length(test_ts))
fc_var_indpro <- ts(fc_var$fcst$INDPRO_growth[,"fcst"], start=c(2016,1), frequency=12)
mse_var_out <- mean((test_ts - fc_var_indpro)^2)

p <- autoplot(window(ts_growth_clean, start=c(2010,1)), series="Actual") +
  autolayer(fc_var_indpro, series="VAR Forecast") +
  ggtitle("VAR Multi-step Forecast") +
  theme_minimal()
print(p)

# Recursive VAR forecast
recursive_fc <- numeric(length(test_ts))
for(i in 1:length(test_ts)){
  sub_data <- cbind(INDPRO_growth=ts_growth_clean[1:(length(train_ts)+i-1)],
                    xreg_clean[1:(length(train_ts)+i-1),])
  fit_rec <- VAR(sub_data, p=p_var, type="const")
  fc_rec <- predict(fit_rec, n.ahead=1)
  recursive_fc[i] <- fc_rec$fcst$INDPRO_growth[,"fcst"]
}
recursive_fc <- ts(recursive_fc, start=c(2016,1), frequency=12)
mse_var_rec <- mean((test_ts - recursive_fc)^2)

p <- autoplot(window(ts_growth_clean, start=c(2010,1)), series="Actual") +
  autolayer(recursive_fc, series="Recursive VAR Forecast") +
  ggtitle("Recursive VAR Forecast") +
  theme_minimal()
print(p)

# =====================================================
# 7️⃣ FORECAST COMPARISON
# =====================================================

results_compare <- data.frame(
  Model = c("ARIMA","ARIMAX (PCA)","Rolling ARIMAX","VAR","Recursive VAR"),
  MSE_Out = c(mse_arima_out, mse_arimax_out, mse_roll_arimax, mse_var_out, mse_var_rec)
)
print(results_compare)

p <- ggplot(results_compare, aes(x=Model, y=MSE_Out, fill=Model)) +
  geom_col() +
  geom_text(aes(label=round(MSE_Out,4)), vjust=-0.3) +
  ggtitle("Out-of-Sample Forecast MSE Comparison") +
  theme_minimal()
print(p)

# =====================================================
# 8️⃣ FORECAST EVALUATION: DIEBOLD–MARIANO TEST
# =====================================================

library(forecast)
library(MSwM)  # for regime-switching models

# Define a small helper for DM test
dm_result <- function(e1, e2, h=1){
  test <- dm.test(e1, e2, h=h, power=2)
  return(data.frame(
    Statistic = round(test$statistic, 3),
    P_value   = round(test$p.value, 4)
  ))
}

# Align forecast errors for comparison
e_arima <- test_ts - fc_arima$mean
e_arimax <- test_ts - fc_arimax$mean
e_roll <- test_ts - rolling_fc_arimax
e_var <- test_ts - fc_var_indpro
e_var_rec <- test_ts - recursive_fc

# Compare models pairwise
dm_arimax_vs_arima <- dm_result(e_arimax, e_arima)
dm_var_vs_arimax <- dm_result(e_var, e_arimax)
dm_roll_vs_arimax <- dm_result(e_roll, e_arimax)
dm_varrec_vs_var <- dm_result(e_var_rec, e_var)

cat("\nDiebold–Mariano Test Results:\n")
dm_table <- data.frame(
  Comparison = c("ARIMAX vs ARIMA", "VAR vs ARIMAX", "Rolling ARIMAX vs ARIMAX", "Recursive VAR vs VAR"),
  Statistic  = c(dm_arimax_vs_arima$Statistic, dm_var_vs_arimax$Statistic, dm_roll_vs_arimax$Statistic, dm_varrec_vs_var$Statistic),
  P_value    = c(dm_arimax_vs_arima$P_value, dm_var_vs_arimax$P_value, dm_roll_vs_arimax$P_value, dm_varrec_vs_var$P_value)
)
print(dm_table)

# Interpretation:
# - p < 0.05: first model significantly better than second.
# - p > 0.05: no significant difference.

library(forecast)
library(ggplot2)
library(dplyr)

# --- Prepare time series ---
ts_indpro <- ts(data$INDPRO_growth, start = c(1978,2), frequency = 12)
train_ts <- window(ts_indpro, end = c(2015,12))
test_ts <- window(ts_indpro, start = c(2016,1))
n_forecast <- length(test_ts)

# -------------------------------
# 1a. Multi-step static forecast (ARIMA)
# -------------------------------
fit_arima <- auto.arima(train_ts)
fc_arima <- forecast(fit_arima, h = n_forecast)

autoplot(fc_arima) +
  autolayer(test_ts, series = "Actual") +
  ggtitle("ARIMA Multi-step Forecast of INDPRO Growth") +
  ylab("Monthly Growth (%)") +
  theme_minimal()

mse_out_arima <- mean((test_ts - fc_arima$mean)^2)
cat("ARIMA Multi-step Out-of-sample MSE:", round(mse_out_arima,4), "\n")

# -------------------------------
# 1b. Rolling (dynamic) forecast (ARIMA)
# -------------------------------
rolling_fc <- numeric(n_forecast)
train_roll <- train_ts

for(i in 1:n_forecast){
  fit_roll <- auto.arima(train_roll)
  fc_step <- forecast(fit_roll, h = 1)
  rolling_fc[i] <- fc_step$mean
  train_roll <- ts(c(train_roll, test_ts[i]), frequency = 12)
}

# Plot rolling forecast vs actual
plot(test_ts, type="l", col="black", lwd=2, ylab="INDPRO Growth", xlab="Time")
lines(rolling_fc, col="blue", lwd=2)
legend("topright", legend=c("Actual","Rolling Forecast"), col=c("black","blue"), lwd=2)

mse_rolling <- mean((test_ts - rolling_fc)^2)
cat("Rolling Forecast Out-of-sample MSE:", round(mse_rolling,4), "\n")

# -------------------------------
# 1c. ETS model (Exponential Smoothing)
# -------------------------------
fit_ets <- ets(train_ts)
fc_ets <- forecast(fit_ets, h = n_forecast)

autoplot(fc_ets) + autolayer(test_ts, series="Actual") +
  ggtitle("ETS Multi-step Forecast") +
  ylab("Monthly Growth (%)") + theme_minimal()

mse_ets <- mean((test_ts - fc_ets$mean)^2)
cat("ETS Forecast Out-of-sample MSE:", round(mse_ets,4), "\n")

library(glmnet)
library(caret)

# --- Create lagged features ---
lags <- 12
n <- length(ts_indpro)
X <- embed(ts_indpro, lags + 1)[, -1]  # predictors
y <- embed(ts_indpro, lags + 1)[, 1]   # target
dates <- tail(time(ts_indpro), nrow(X))

# Split train/test
train_idx <- which(dates <= 2015 + 12/12)
test_idx  <- which(dates > 2015 + 12/12)

X_train <- X[train_idx, ]
y_train <- y[train_idx]

X_test <- X[test_idx, ]
y_test <- y[test_idx]

# -------------------------------
# 2a. Lasso regression
# -------------------------------
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1)
lasso_pred <- predict(cv_lasso, X_test)

mse_lasso <- mean((y_test - lasso_pred)^2)
cat("Lasso Out-of-sample MSE:", round(mse_lasso,4), "\n")

plot(y_test, type="l", col="black", lwd=2, ylab="INDPRO Growth", xlab="Time")
lines(lasso_pred, col="red", lwd=2)
legend("topright", legend=c("Actual","Lasso Forecast"), col=c("black","red"), lwd=2)

# -------------------------------
# 2b. Random Forest
# -------------------------------
library(randomForest)
rf_fit <- randomForest(X_train, y_train)
rf_pred <- predict(rf_fit, X_test)

mse_rf <- mean((y_test - rf_pred)^2)
cat("Random Forest Out-of-sample MSE:", round(mse_rf,4), "\n")

plot(y_test, type="l", col="black", lwd=2)
lines(rf_pred, col="blue", lwd=2)
legend("topright", legend=c("Actual","RF Forecast"), col=c("black","blue"), lwd=2)

# =====================================================
# FULL TIME SERIES FORECAST PIPELINE: INDPRO
# =====================================================

# --- Load packages ---
library(forecast)
library(ggplot2)
library(dplyr)
library(vars)
library(glmnet)
library(caret)
library(randomForest)
library(stats)
library(tseries)

# =====================================================
# 0️⃣ Prepare data
# =====================================================
ts_indpro <- ts(data$INDPRO_growth, start = c(1978,2), frequency = 12)

# Example macro variables for VAR (replace with your actual columns)
var_data <- data %>% 
  dplyr::select(INDPRO_growth, CP3Mx, COMPAPFFx, UMCSENTx, VIXCLSx) %>%
  na.omit()

ts_var <- ts(var_data, start=c(1978,2), frequency=12)

# Split train/test
train_ts <- window(ts_indpro, end=c(2015,12))
test_ts  <- window(ts_indpro, start=c(2016,1))
n_forecast <- length(test_ts)

train_var <- window(ts_var, end=c(2015,12))
test_var  <- window(ts_var, start=c(2016,1))

# =====================================================
# 1️⃣ ARIMA Forecasts
# =====================================================
# Multi-step (static)
fit_arima <- auto.arima(train_ts)
fc_arima <- forecast(fit_arima, h=n_forecast)

# Rolling (dynamic)
rolling_fc <- numeric(n_forecast)
train_roll <- train_ts
for(i in 1:n_forecast){
  fit_roll <- auto.arima(train_roll)
  rolling_fc[i] <- forecast(fit_roll, h=1)$mean
  train_roll <- ts(c(train_roll, test_ts[i]), frequency=12)
}

# =====================================================
# 2️⃣ ETS Forecast
# =====================================================
fit_ets <- ets(train_ts)
fc_ets <- forecast(fit_ets, h=n_forecast)

# =====================================================
# 3️⃣ VAR Forecast
# =====================================================
lag_select <- VARselect(train_var, lag.max=12, type="const")$selection["AIC(n)"]
fit_var <- VAR(train_var, p=lag_select, type="const")
fc_var <- predict(fit_var, n.ahead=nrow(test_var))
fc_var_indpro <- sapply(fc_var$fcst$INDPRO_growth, function(x) x["fcst"])

# =====================================================
# 4️⃣ ML Forecasts using Lags + PCA
# =====================================================
# Create lagged features
lags <- 12
n <- length(ts_indpro)
X <- embed(ts_indpro, lags + 1)[, -1]
y <- embed(ts_indpro, lags + 1)[, 1]
dates <- tail(time(ts_indpro), nrow(X))

train_idx <- which(dates <= 2015 + 12/12)
test_idx  <- which(dates > 2015 + 12/12)

X_train <- X[train_idx, ]; y_train <- y[train_idx]
X_test  <- X[test_idx, ];  y_test  <- y[test_idx]

# PCA to reduce collinearity
pca <- prcomp(X_train, scale.=TRUE)
X_train_pca <- pca$x[,1:5]  # keep top 5 components
X_test_pca  <- predict(pca, newdata=X_test)[,1:5]

# Lasso regression
cv_lasso <- cv.glmnet(X_train_pca, y_train, alpha=1)
lasso_pred <- predict(cv_lasso, X_test_pca)

# Random Forest
rf_fit <- randomForest(X_train_pca, y_train)
rf_pred <- predict(rf_fit, X_test_pca)

# =====================================================
# 5️⃣ Plot forecasts
# =====================================================
autoplot(window(ts_indpro, start=c(2010,1))) +
  autolayer(fc_arima$mean, series="ARIMA Multi-step") +
  autolayer(rolling_fc, series="ARIMA Rolling") +
  autolayer(fc_ets$mean, series="ETS") +
  autolayer(fc_var_indpro, series="VAR") +
  autolayer(lasso_pred, series="Lasso PCA") +
  autolayer(rf_pred, series="RF PCA") +
  autolayer(window(ts_indpro, start=c(2016,1)), series="Actual") +
  ggtitle("INDPRO Forecasts Comparison") + ylab("Monthly Growth (%)") +
  theme_minimal() +
  scale_colour_manual(values=c("black","red","blue","green","orange","purple","black"))

# =====================================================
# 6️⃣ Compute Out-of-sample MSE
# =====================================================
results_compare <- data.frame(
  Model = c("ARIMA Multi-step","ARIMA Rolling","ETS","VAR","Lasso PCA","RF PCA"),
  MSE_Out = c(
    mean((test_ts - fc_arima$mean)^2),
    mean((test_ts - rolling_fc)^2),
    mean((test_ts - fc_ets$mean)^2),
    mean((test_var[,"INDPRO_growth"] - fc_var_indpro)^2),
    mean((y_test - lasso_pred)^2),
    mean((y_test - rf_pred)^2)
  )
)
print(results_compare)

ggplot(results_compare, aes(x=Model, y=MSE_Out, fill=Model)) +
  geom_col(width=0.6) +
  geom_text(aes(label=round(MSE_Out,4)), vjust=-0.3) +
  theme_minimal() +
  labs(title="Out-of-Sample MSE Comparison", y="MSE")

# =====================================================
# 0️⃣ Detect structural breaks
# =====================================================
library(strucchange)

# Use growth series for stationarity
ts_growth <- ts_indpro

# Breakpoints in mean
bp_mean <- breakpoints(ts_growth ~ 1)
summary(bp_mean)

# Visualize
plot(ts_growth, main="Structural Breaks in INDPRO Growth")
lines(fitted(bp_mean, breaks=2), col="red", lwd=2)
abline(v = time(ts_growth)[bp_mean$breakpoints], col="blue", lty=2)

# Extract breakpoints
breakpoints_idx <- bp_mean$breakpoints
regimes <- c(0, breakpoints_idx, length(ts_growth)) # start & end

# =====================================================
# 1️⃣ Fit ARIMA/ETS/VAR per regime
# =====================================================
fc_arima_break <- numeric(n_forecast)
fc_ets_break   <- numeric(n_forecast)

train_start <- 1
for(i in seq_along(regimes[-1])){
  train_end <- regimes[i+1]
  train_regime <- window(ts_growth, start=train_start, end=train_end)
  
  # ARIMA per regime
  fit_regime_arima <- auto.arima(train_regime)
  h <- min(n_forecast, length(test_ts)) # forecast horizon
  fc_arima_break[(1:h) + (train_start - 1)] <- forecast(fit_regime_arima, h=h)$mean
  
  # ETS per regime
  fit_regime_ets <- ets(train_regime)
  fc_ets_break[(1:h) + (train_start - 1)] <- forecast(fit_regime_ets, h=h)$mean
  
  train_start <- train_end + 1
}

# =====================================================
# 2️⃣ VAR with regime dummies
# =====================================================
# Create regime dummy variable
regime_dummy <- rep(NA, length(ts_growth))
for(i in seq_along(regimes[-1])){
  regime_dummy[(regimes[i]+1):regimes[i+1]] <- i
}
var_data_regime <- cbind(ts_var, regime_dummy=regime_dummy)
var_data_regime <- na.omit(var_data_regime)

train_var <- window(var_data_regime, end=c(2015,12))
test_var  <- window(var_data_regime, start=c(2016,1))

lag_select <- VARselect(train_var[,-ncol(train_var)], lag.max=12, type="const")$selection["AIC(n)"]
fit_var_break <- VAR(train_var[,-ncol(train_var)], p=lag_select, type="const")
fc_var_break <- predict(fit_var_break, n.ahead=nrow(test_var))
fc_var_indpro_break <- sapply(fc_var_break$fcst$INDPRO_growth, function(x) x["fcst"])

# =====================================================
# 3️⃣ Plot regime-adjusted forecasts
# =====================================================
autoplot(window(ts_growth, start=c(2010,1))) +
  autolayer(fc_arima_break, series="ARIMA with breaks") +
  autolayer(fc_ets_break, series="ETS with breaks") +
  autolayer(fc_var_indpro_break, series="VAR with breaks") +
  autolayer(window(ts_growth, start=c(2016,1)), series="Actual") +
  ggtitle("Forecasts with Structural Breaks") +
  ylab("INDPRO Growth") +
  theme_minimal()

# =====================================================
# 4️⃣ Compute out-of-sample MSE for regime-adjusted forecasts
# =====================================================
results_break <- data.frame(
  Model = c("ARIMA Breaks","ETS Breaks","VAR Breaks"),
  MSE_Out = c(
    mean((test_ts - fc_arima_break[1:n_forecast])^2),
    mean((test_ts - fc_ets_break[1:n_forecast])^2),
    mean((test_var[,"INDPRO_growth"] - fc_var_indpro_break)^2)
  )
)
print(results_break)

# ==========================================
# ML FORECASTING PIPELINE FOR INDPRO
# ==========================================

# --- Load packages ---
library(dplyr)
library(ggplot2)
library(randomForest)
library(glmnet)
library(zoo)
library(caret)
library(tidyr)

# --- Prepare lagged features ---
create_lags <- function(ts, lags = 12) {
  df <- data.frame(y = ts)
  for(i in 1:lags){
    df[[paste0("lag", i)]] <- dplyr::lag(ts, i)
  }
  return(df)
}

# Example: use 12 lags
lagged_df <- create_lags(ts_growth, lags = 12)

# Add macro variables (contemporaneous or lagged)
macro_vars <- data %>%
  select(CP3Mx, COMPAPFFx, UMCSENTx, VIXCLSx) %>%
  mutate_all(~ na.approx(.)) # linear interpolation if needed

lagged_df <- cbind(lagged_df, macro_vars)
lagged_df <- lagged_df %>% drop_na()

# --- Train/test split ---
train_idx <- which(index(lagged_df$y) <= 2015 + 12/12) # till Dec 2015
test_idx <- which(index(lagged_df$y) > 2015 + 12/12)

train_data <- lagged_df[train_idx, ]
test_data <- lagged_df[test_idx, ]

X_train <- as.matrix(train_data %>% select(-y))
y_train <- train_data$y
X_test <- as.matrix(test_data %>% select(-y))
y_test <- test_data$y

# ==========================================
# 1️⃣ Random Forest
# ==========================================
set.seed(123)
rf_model <- randomForest(x = X_train, y = y_train, ntree = 500)

rf_pred <- predict(rf_model, X_test)

# Rolling forecast (recursive)
rf_roll <- numeric(nrow(X_test))
train_roll <- X_train
y_roll <- y_train

for(i in 1:nrow(X_test)){
  rf_tmp <- randomForest(x = train_roll, y = y_roll, ntree = 200)
  rf_roll[i] <- predict(rf_tmp, X_test[i,,drop=FALSE])
  # Append the true value to rolling train
  train_roll <- rbind(train_roll, X_test[i,,drop=FALSE])
  y_roll <- c(y_roll, y_test[i])
}

# ==========================================
# 2️⃣ Lasso & Ridge Regression
# ==========================================
# Standardize features
X_train_s <- scale(X_train)
X_test_s <- scale(X_test, center = attr(X_train_s, "scaled:center"),
                  scale = attr(X_train_s, "scaled:scale"))

# Lasso
lasso_model <- cv.glmnet(X_train_s, y_train, alpha = 1)
lasso_pred <- predict(lasso_model, X_test_s, s = "lambda.min")

# Ridge
ridge_model <- cv.glmnet(X_train_s, y_train, alpha = 0)
ridge_pred <- predict(ridge_model, X_test_s, s = "lambda.min")

# ==========================================
# 3️⃣ Evaluation
# ==========================================
evaluate <- function(true, pred){
  mse <- mean((true - pred)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(true - pred))
  return(c(MSE = mse, RMSE = rmse, MAE = mae))
}

eval_rf <- evaluate(y_test, rf_pred)
eval_rf_roll <- evaluate(y_test, rf_roll)
eval_lasso <- evaluate(y_test, lasso_pred)
eval_ridge <- evaluate(y_test, ridge_pred)

results <- data.frame(
  Model = c("Random Forest", "RF Rolling", "Lasso", "Ridge"),
  MSE = c(eval_rf["MSE"], eval_rf_roll["MSE"], eval_lasso["MSE"], eval_ridge["MSE"]),
  RMSE = c(eval_rf["RMSE"], eval_rf_roll["RMSE"], eval_lasso["RMSE"], eval_ridge["RMSE"]),
  MAE = c(eval_rf["MAE"], eval_rf_roll["MAE"], eval_lasso["MAE"], eval_ridge["MAE"])
)
print(results)

# ==========================================
# 4️⃣ Plot forecasts vs actual
# ==========================================
plot_df <- data.frame(
  Date = index(test_data$y),
  Actual = y_test,
  RF = rf_pred,
  RF_Roll = rf_roll,
  Lasso = as.vector(lasso_pred),
  Ridge = as.vector(ridge_pred)
) %>%
  pivot_longer(cols = -Date, names_to = "Model", values_to = "Forecast")

ggplot(plot_df, aes(x = Date, y = Forecast, color = Model)) +
  geom_line(size = 1) +
  ggtitle("Multi-step Forecasts for INDPRO Growth") +
  theme_minimal()

# ==========================================
# ML FORECASTING PIPELINE WITH STATIC + RECURSIVE FORECASTS
# ==========================================

library(dplyr)
library(ggplot2)
library(randomForest)
library(glmnet)
library(zoo)
library(tidyr)
library(caret)

# -------------------------------
# 1️⃣ Lagged features function
# -------------------------------
create_lags <- function(ts, lags = 12) {
  df <- data.frame(y = ts)
  for(i in 1:lags){
    df[[paste0("lag", i)]] <- dplyr::lag(ts, i)
  }
  return(df)
}

# -------------------------------
# 2️⃣ Prepare dataset
# -------------------------------
lags <- 12
lagged_df <- create_lags(ts_growth, lags = lags)

macro_vars <- data %>%
  select(CP3Mx, COMPAPFFx, UMCSENTx, VIXCLSx) %>%
  mutate_all(~ na.approx(.)) # interpolate missing values

lagged_df <- cbind(lagged_df, macro_vars)
lagged_df <- lagged_df %>% drop_na()

# -------------------------------
# 3️⃣ Train/test split
# -------------------------------
train_idx <- 1:which(index(lagged_df$y) > 2015 + 12/12)[1]-1
test_idx <- setdiff(1:nrow(lagged_df), train_idx)

train_data <- lagged_df[train_idx, ]
test_data <- lagged_df[test_idx, ]

X_train <- as.matrix(train_data %>% select(-y))
y_train <- train_data$y
X_test <- as.matrix(test_data %>% select(-y))
y_test <- test_data$y

# -------------------------------
# 4️⃣ STATIC (multi-step) forecasts
# -------------------------------
# Random Forest
set.seed(123)
rf_model <- randomForest(x = X_train, y = y_train, ntree = 500)
rf_pred_static <- predict(rf_model, X_test)

# Lasso
X_train_s <- scale(X_train)
X_test_s <- scale(X_test, center = attr(X_train_s, "scaled:center"),
                  scale = attr(X_train_s, "scaled:scale"))
lasso_model <- cv.glmnet(X_train_s, y_train, alpha = 1)
lasso_pred_static <- predict(lasso_model, X_test_s, s = "lambda.min")

# Ridge
ridge_model <- cv.glmnet(X_train_s, y_train, alpha = 0)
ridge_pred_static <- predict(ridge_model, X_test_s, s = "lambda.min")

# -------------------------------
# 5️⃣ RECURSIVE (dynamic) multi-step forecasts
# -------------------------------
H <- nrow(X_test)
rf_pred_dyn <- numeric(H)
lasso_pred_dyn <- numeric(H)
ridge_pred_dyn <- numeric(H)

# Initialize rolling datasets
X_roll <- X_train
y_roll <- y_train

X_roll_s <- scale(X_roll)  # for lasso/ridge

for(i in 1:H){
  # Current test row
  x_next <- X_test[i, , drop = FALSE]
  
  # --- Random Forest ---
  rf_tmp <- randomForest(x = X_roll, y = y_roll, ntree = 200)
  rf_pred_dyn[i] <- predict(rf_tmp, x_next)
  
  # --- Lasso/Ridge ---
  x_next_s <- scale(x_next, center = attr(X_roll_s, "scaled:center"),
                    scale = attr(X_roll_s, "scaled:scale"))
  
  lasso_tmp <- cv.glmnet(X_roll_s, y_roll, alpha = 1)
  ridge_tmp <- cv.glmnet(X_roll_s, y_roll, alpha = 0)
  
  lasso_pred_dyn[i] <- predict(lasso_tmp, x_next_s, s = "lambda.min")
  ridge_pred_dyn[i] <- predict(ridge_tmp, x_next_s, s = "lambda.min")
  
  # Append the actual value to rolling dataset
  X_roll <- rbind(X_roll, x_next)
  y_roll <- c(y_roll, y_test[i])
  
  # Update scaled matrix
  X_roll_s <- scale(X_roll)
}

# -------------------------------
# 6️⃣ Evaluation metrics
# -------------------------------
evaluate <- function(true, pred){
  mse <- mean((true - pred)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(true - pred))
  return(c(MSE = mse, RMSE = rmse, MAE = mae))
}

results <- data.frame(
  Model = c("RF Static", "RF Recursive", "Lasso Static", "Lasso Recursive",
            "Ridge Static", "Ridge Recursive"),
  MSE = c(evaluate(y_test, rf_pred_static)["MSE"],
          evaluate(y_test, rf_pred_dyn)["MSE"],
          evaluate(y_test, lasso_pred_static)["MSE"],
          evaluate(y_test, lasso_pred_dyn)["MSE"],
          evaluate(y_test, ridge_pred_static)["MSE"],
          evaluate(y_test, ridge_pred_dyn)["MSE"]),
  RMSE = c(evaluate(y_test, rf_pred_static)["RMSE"],
           evaluate(y_test, rf_pred_dyn)["RMSE"],
           evaluate(y_test, lasso_pred_static)["RMSE"],
           evaluate(y_test, lasso_pred_dyn)["RMSE"],
           evaluate(y_test, ridge_pred_static)["RMSE"],
           evaluate(y_test, ridge_pred_dyn)["RMSE"]),
  MAE = c(evaluate(y_test, rf_pred_static)["MAE"],
          evaluate(y_test, rf_pred_dyn)["MAE"],
          evaluate(y_test, lasso_pred_static)["MAE"],
          evaluate(y_test, lasso_pred_dyn)["MAE"],
          evaluate(y_test, ridge_pred_static)["MAE"],
          evaluate(y_test, ridge_pred_dyn)["MAE"])
)

print(results)

# -------------------------------
# 7️⃣ Plot forecasts
# -------------------------------
plot_df <- data.frame(
  Date = index(test_data$y),
  Actual = y_test,
  RF_Static = rf_pred_static,
  RF_Recursive = rf_pred_dyn,
  Lasso_Static = as.vector(lasso_pred_static),
  Lasso_Recursive = as.vector(lasso_pred_dyn),
  Ridge_Static = as.vector(ridge_pred_static),
  Ridge_Recursive = as.vector(ridge_pred_dyn)
) %>%
  pivot_longer(cols = -Date, names_to = "Model", values_to = "Forecast")

ggplot(plot_df, aes(x = Date, y = Forecast, color = Model)) +
  geom_line(size = 1) +
  ggtitle("INDPRO Growth: Static vs Recursive Forecasts") +
  theme_minimal()

# =====================================================
# COMPREHENSIVE TIME SERIES FORECAST PIPELINE
# =====================================================

# --- Load packages ---
library(forecast)
library(vars)
library(tseries)
library(ggplot2)
library(dplyr)
library(randomForest)
library(glmnet)
library(zoo)
library(tidyr)

# --- Prepare series ---
ts_indpro <- ts(data$INDPRO, start = c(1978, 1), frequency = 12)
ts_growth <- ts(data$INDPRO_growth, start = c(1978, 2), frequency = 12)

# --- Split train/test ---
train_ts <- window(ts_indpro, end = c(2015, 12))
test_ts <- window(ts_indpro, start = c(2016, 1))
train_growth <- window(ts_growth, end = c(2015, 12))
test_growth <- window(ts_growth, start = c(2016, 1))
n_forecast <- length(test_ts)

# =====================================================
# 1️⃣ ARIMA Forecast
# =====================================================
fit_arima <- auto.arima(train_ts)
fc_arima <- forecast(fit_arima, h = n_forecast)

mse_arima <- mean((test_ts - fc_arima$mean)^2)
rmse_arima <- sqrt(mse_arima)

# =====================================================
# 2️⃣ VAR Forecast (multivariate)
# =====================================================
var_data <- data %>%
  select(INDPRO_growth, CP3Mx, COMPAPFFx, UMCSENTx, VIXCLSx) %>%
  na.omit()

ts_var <- ts(var_data, start = c(1978, 2), frequency = 12)
train_var <- window(ts_var, end = c(2015, 12))
test_var <- window(ts_var, start = c(2016, 1))

lag_select <- VARselect(train_var, lag.max = 12, type = "const")
fit_var <- VAR(train_var, p = lag_select$selection["AIC(n)"], type = "const")
fc_var <- predict(fit_var, n.ahead = nrow(test_var))
fc_var_indpro <- ts(fc_var$fcst$INDPRO_growth[, "fcst"], start = c(2016, 1), frequency = 12)

mse_var <- mean((test_var[, "INDPRO_growth"] - fc_var_indpro)^2)
rmse_var <- sqrt(mse_var)

# =====================================================
# 3️⃣ ML: Lagged features + macro variables
# =====================================================
create_lags <- function(ts, lags = 12) {
  df <- data.frame(y = ts)
  for(i in 1:lags){
    df[[paste0("lag", i)]] <- dplyr::lag(ts, i)
  }
  return(df)
}

lags <- 12
lagged_df <- create_lags(ts_growth, lags = lags)
macro_vars <- data %>% select(CP3Mx, COMPAPFFx, UMCSENTx, VIXCLSx) %>% mutate_all(~ na.approx(.))
lagged_df <- cbind(lagged_df, macro_vars) %>% drop_na()

train_idx <- 1:(nrow(lagged_df) - n_forecast)
test_idx <- (nrow(lagged_df) - n_forecast + 1):nrow(lagged_df)
train_data <- lagged_df[train_idx, ]
test_data <- lagged_df[test_idx, ]

X_train <- as.matrix(train_data %>% select(-y))
y_train <- train_data$y
X_test <- as.matrix(test_data %>% select(-y))
y_test <- test_data$y

# ----- Static forecasts -----
set.seed(123)
rf_model <- randomForest(X_train, y_train, ntree = 500)
rf_pred_static <- predict(rf_model, X_test)

X_train_s <- scale(X_train)
X_test_s <- scale(X_test, center = attr(X_train_s, "scaled:center"), scale = attr(X_train_s, "scaled:scale"))

lasso_model <- cv.glmnet(X_train_s, y_train, alpha = 1)
lasso_pred_static <- predict(lasso_model, X_test_s, s = "lambda.min")

ridge_model <- cv.glmnet(X_train_s, y_train, alpha = 0)
ridge_pred_static <- predict(ridge_model, X_test_s, s = "lambda.min")

# ----- Recursive (dynamic) forecasts -----
H <- nrow(X_test)
rf_pred_dyn <- numeric(H)
lasso_pred_dyn <- numeric(H)
ridge_pred_dyn <- numeric(H)

X_roll <- X_train
y_roll <- y_train
X_roll_s <- scale(X_roll)

for(i in 1:H){
  x_next <- X_test[i, , drop = FALSE]
  
  # Random Forest
  rf_tmp <- randomForest(X_roll, y_roll, ntree = 200)
  rf_pred_dyn[i] <- predict(rf_tmp, x_next)
  
  # Lasso / Ridge
  x_next_s <- scale(x_next, center = attr(X_roll_s, "scaled:center"), scale = attr(X_roll_s, "scaled:scale"))
  lasso_tmp <- cv.glmnet(X_roll_s, y_roll, alpha = 1)
  ridge_tmp <- cv.glmnet(X_roll_s, y_roll, alpha = 0)
  lasso_pred_dyn[i] <- predict(lasso_tmp, x_next_s, s = "lambda.min")
  ridge_pred_dyn[i] <- predict(ridge_tmp, x_next_s, s = "lambda.min")
  
  # Update rolling datasets
  X_roll <- rbind(X_roll, x_next)
  y_roll <- c(y_roll, y_test[i])
  X_roll_s <- scale(X_roll)
}

# =====================================================
# 4️⃣ Evaluation
# =====================================================
evaluate <- function(true, pred){
  mse <- mean((true - pred)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(true - pred))
  return(c(MSE = mse, RMSE = rmse, MAE = mae))
}

results <- data.frame(
  Model = c("ARIMA", "VAR", "RF Static", "RF Recursive", "Lasso Static", "Lasso Recursive",
            "Ridge Static", "Ridge Recursive"),
  MSE = c(mse_arima, mse_var,
          evaluate(y_test, rf_pred_static)["MSE"],
          evaluate(y_test, rf_pred_dyn)["MSE"],
          evaluate(y_test, lasso_pred_static)["MSE"],
          evaluate(y_test, lasso_pred_dyn)["MSE"],
          evaluate(y_test, ridge_pred_static)["MSE"],
          evaluate(y_test, ridge_pred_dyn)["MSE"]),
  RMSE = c(rmse_arima, rmse_var,
           evaluate(y_test, rf_pred_static)["RMSE"],
           evaluate(y_test, rf_pred_dyn)["RMSE"],
           evaluate(y_test, lasso_pred_static)["RMSE"],
           evaluate(y_test, lasso_pred_dyn)["RMSE"],
           evaluate(y_test, ridge_pred_static)["RMSE"],
           evaluate(y_test, ridge_pred_dyn)["RMSE"])
)
print(results)

# =====================================================
# 5️⃣ Plot forecasts
# =====================================================
plot_df <- data.frame(
  Date = index(test_data$y),
  Actual = y_test,
  ARIMA = as.vector(fc_arima$mean),
  VAR = as.vector(fc_var_indpro),
  RF_Static = rf_pred_static,
  RF_Recursive = rf_pred_dyn,
  Lasso_Static = as.vector(lasso_pred_static),
  Lasso_Recursive = as.vector(lasso_pred_dyn),
  Ridge_Static = as.vector(ridge_pred_static),
  Ridge_Recursive = as.vector(ridge_pred_dyn)
) %>% pivot_longer(cols = -Date, names_to = "Model", values_to = "Forecast")

ggplot(plot_df, aes(x = Date, y = Forecast, color = Model)) +
  geom_line(size = 1) +
  ggtitle("Forecast Comparison: ARIMA, VAR, ML Models") +
  theme_minimal()

# =====================================================
# EXTENDED TIME SERIES FORECAST PIPELINE WITH XGBOOST/LGBM + PCA
# =====================================================

# --- Load packages ---
library(forecast)
library(vars)
library(tseries)
library(ggplot2)
library(dplyr)
library(randomForest)
library(glmnet)
library(xgboost)
library(lightgbm)
library(zoo)
library(tidyr)
library(stats)  # for prcomp

# --- Prepare series ---
ts_indpro <- ts(data$INDPRO, start = c(1978, 1), frequency = 12)
ts_growth <- ts(data$INDPRO_growth, start = c(1978, 2), frequency = 12)

# --- Split train/test ---
train_ts <- window(ts_indpro, end = c(2015, 12))
test_ts <- window(ts_indpro, start = c(2016, 1))
train_growth <- window(ts_growth, end = c(2015, 12))
test_growth <- window(ts_growth, start = c(2016, 1))
n_forecast <- length(test_ts)

# --- VAR and ARIMA as before ---
fit_arima <- auto.arima(train_ts)
fc_arima <- forecast(fit_arima, h = n_forecast)
fit_var <- VAR(window(ts(data %>% select(INDPRO_growth, CP3Mx, COMPAPFFx, UMCSENTx, VIXCLSx), start=c(1978,2), frequency=12), end=c(2015,12)),
               p = 2, type="const")
fc_var <- predict(fit_var, n.ahead = n_forecast)
fc_var_indpro <- ts(fc_var$fcst$INDPRO_growth[, "fcst"], start = c(2016,1), frequency = 12)

# --- Create lagged features ---
create_lags <- function(ts, lags=12){
  df <- data.frame(y=ts)
  for(i in 1:lags){
    df[[paste0("lag",i)]] <- dplyr::lag(ts, i)
  }
  return(df)
}
lags <- 12
lagged_df <- create_lags(ts_growth, lags=lags)

# --- Macro variables ---
macro_vars <- data %>% select(CP3Mx, COMPAPFFx, UMCSENTx, VIXCLSx) %>% mutate_all(~ na.approx(.))
macro_vars <- macro_vars[(lags+1):nrow(macro_vars), ]  # align with lagged_df

lagged_df <- lagged_df[(lags+1):nrow(lagged_df), ]
X_full <- cbind(lagged_df %>% select(-y), macro_vars)
y_full <- lagged_df$y

# --- PCA on macro variables ---
pca_res <- prcomp(macro_vars, scale. = TRUE)
pc_vars <- as.data.frame(pca_res$x[, 1:3])  # first 3 PCs
X_full_pca <- cbind(lagged_df %>% select(-y), pc_vars)

# --- Train/test split ---
train_idx <- 1:(nrow(X_full_pca) - n_forecast)
test_idx <- (nrow(X_full_pca) - n_forecast + 1):nrow(X_full_pca)

X_train <- as.matrix(X_full_pca[train_idx, ])
y_train <- y_full[train_idx]
X_test <- as.matrix(X_full_pca[test_idx, ])
y_test <- y_full[test_idx]

# =====================================================
# ML MODELS
# =====================================================

# --- Random Forest ---
set.seed(123)
rf_model <- randomForest(X_train, y_train, ntree=500)
rf_pred <- predict(rf_model, X_test)

# --- Lasso ---
X_train_s <- scale(X_train)
X_test_s <- scale(X_test, center=attr(X_train_s,"scaled:center"), scale=attr(X_train_s,"scaled:scale"))
lasso_model <- cv.glmnet(X_train_s, y_train, alpha=1)
lasso_pred <- predict(lasso_model, X_test_s, s="lambda.min")

# --- Ridge ---
ridge_model <- cv.glmnet(X_train_s, y_train, alpha=0)
ridge_pred <- predict(ridge_model, X_test_s, s="lambda.min")

# --- XGBoost ---
xgb_train <- xgb.DMatrix(data=X_train, label=y_train)
xgb_test <- xgb.DMatrix(data=X_test)
xgb_model <- xgboost(data=xgb_train, nrounds=200, objective="reg:squarederror", verbose=0)
xgb_pred <- predict(xgb_model, xgb_test)

# --- LightGBM ---
dtrain <- lgb.Dataset(data=X_train, label=y_train)
lgb_model <- lgb.train(params=list(objective="regression", metric="rmse"),
                       data=dtrain, nrounds=200)
lgb_pred <- predict(lgb_model, X_test)

# =====================================================
#  Evaluation
# =====================================================
evaluate <- function(true, pred){
  mse <- mean((true - pred)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(true - pred))
  return(c(MSE=mse, RMSE=rmse, MAE=mae))
}

results <- data.frame(
  Model=c("ARIMA", "VAR", "RF", "Lasso", "Ridge", "XGBoost", "LightGBM"),
  MSE=c(mean((fc_arima$mean - test_ts)^2),
        mean((fc_var_indpro - test_ts)^2),
        evaluate(y_test, rf_pred)["MSE"],
        evaluate(y_test, lasso_pred)["MSE"],
        evaluate(y_test, ridge_pred)["MSE"],
        evaluate(y_test, xgb_pred)["MSE"],
        evaluate(y_test, lgb_pred)["MSE"]),
  RMSE=c(sqrt(mean((fc_arima$mean - test_ts)^2)),
         sqrt(mean((fc_var_indpro - test_ts)^2)),
         evaluate(y_test, rf_pred)["RMSE"],
         evaluate(y_test, lasso_pred)["RMSE"],
         evaluate(y_test, ridge_pred)["RMSE"],
         evaluate(y_test, xgb_pred)["RMSE"],
         evaluate(y_test, lgb_pred)["RMSE"])
)
print(results)

# =====================================================
# Plot all forecasts
# =====================================================
plot_df <- data.frame(
  Date = seq.Date(from=as.Date("2016-01-01"), by="month", length.out=n_forecast),
  Actual = y_test,
  ARIMA = as.vector(fc_arima$mean),
  VAR = as.vector(fc_var_indpro),
  RF = rf_pred,
  Lasso = as.vector(lasso_pred),
  Ridge = as.vector(ridge_pred),
  XGBoost = xgb_pred,
  LightGBM = lgb_pred
) %>% pivot_longer(cols=-Date, names_to="Model", values_to="Forecast")

ggplot(plot_df, aes(x=Date, y=Forecast, color=Model)) +
  geom_line(size=1) +
  ggtitle("Time Series Forecast Comparison: ARIMA, VAR, ML Models (with PCA)") +
  theme_minimal()

# =====================================================
# ROLLING (RECURSIVE) FORECASTS FOR ML MODELS
# =====================================================

n_forecast <- length(y_test)
rolling_preds <- data.frame(
  RF = numeric(n_forecast),
  Lasso = numeric(n_forecast),
  Ridge = numeric(n_forecast),
  XGBoost = numeric(n_forecast),
  LightGBM = numeric(n_forecast)
)

# Initialize training data
X_roll_train <- X_train
y_roll_train <- y_train

for(i in 1:n_forecast){
  # --- Random Forest ---
  rf_model <- randomForest(X_roll_train, y_roll_train, ntree=500)
  rolling_preds$RF[i] <- predict(rf_model, X_test[i,,drop=FALSE])
  
  # --- Lasso ---
  X_s <- scale(X_roll_train)
  X_test_s <- scale(X_test[i,,drop=FALSE], center=attr(X_s,"scaled:center"), scale=attr(X_s,"scaled:scale"))
  lasso_model <- cv.glmnet(X_s, y_roll_train, alpha=1)
  rolling_preds$Lasso[i] <- predict(lasso_model, X_test_s, s="lambda.min")
  
  # --- Ridge ---
  ridge_model <- cv.glmnet(X_s, y_roll_train, alpha=0)
  rolling_preds$Ridge[i] <- predict(ridge_model, X_test_s, s="lambda.min")
  
  # --- XGBoost ---
  xgb_train <- xgb.DMatrix(data=X_roll_train, label=y_roll_train)
  xgb_model <- xgboost(data=xgb_train, nrounds=200, objective="reg:squarederror", verbose=0)
  xgb_pred <- predict(xgb_model, xgb.DMatrix(data=X_test[i,,drop=FALSE]))
  rolling_preds$XGBoost[i] <- xgb_pred
  
  # --- LightGBM ---
  dtrain <- lgb.Dataset(data=X_roll_train, label=y_roll_train)
  lgb_model <- lgb.train(params=list(objective="regression", metric="rmse"), data=dtrain, nrounds=200)
  rolling_preds$LightGBM[i] <- predict(lgb_model, X_test[i,,drop=FALSE])
  
  # --- Update training set with the true value (recursive) ---
  X_roll_train <- rbind(X_roll_train, X_test[i,,drop=FALSE])
  y_roll_train <- c(y_roll_train, y_test[i])
}

# =====================================================
# Evaluation
# =====================================================
evaluate <- function(true, pred){
  mse <- mean((true - pred)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(true - pred))
  return(c(MSE=mse, RMSE=rmse, MAE=mae))
}

rolling_results <- sapply(rolling_preds, function(pred) evaluate(y_test, pred))
rolling_results <- as.data.frame(t(rolling_results))
print(rolling_results)

# =====================================================
# Plot rolling forecasts vs actual
# =====================================================
plot_df <- data.frame(
  Date = seq.Date(from=as.Date("2016-01-01"), by="month", length.out=n_forecast),
  Actual = y_test,
  RF = rolling_preds$RF,
  Lasso = rolling_preds$Lasso,
  Ridge = rolling_preds$Ridge,
  XGBoost = rolling_preds$XGBoost,
  LightGBM = rolling_preds$LightGBM
) %>% pivot_longer(cols=-Date, names_to="Model", values_to="Forecast")

ggplot(plot_df, aes(x=Date, y=Forecast, color=Model)) +
  geom_line(size=1) +
  geom_line(aes(y=Actual), color="black", size=1, linetype="dashed") +
  ggtitle("Rolling (Recursive) Forecasts: ML Models vs Actual") +
  theme_minimal()

# =====================================================
# UNIFIED ROLLING FORECAST COMPARISON
# =====================================================

library(forecast)
library(vars)
library(randomForest)
library(glmnet)
library(xgboost)
library(lightgbm)
library(ggplot2)
library(dplyr)
library(tidyr)

# -------------------------------
# Assume data_ts contains INDPRO_growth
# -------------------------------
ts_all <- ts(data_ts$INDPRO_growth, start=c(1978,2), frequency=12)
train_ts <- window(ts_all, end=c(2015,12))
test_ts <- window(ts_all, start=c(2016,1))
n_forecast <- length(test_ts)

# -------------------------------
# Prepare VAR data
# -------------------------------
var_vars <- data_ts %>% select(INDPRO_growth, CP3Mx, COMPAPFFx, UMCSENTx, VIXCLSx) %>% na.omit()
ts_var <- ts(var_vars, start=c(1978,2), frequency=12)
train_var <- window(ts_var, end=c(2015,12))
test_var <- window(ts_var, start=c(2016,1))
lag_sel <- VARselect(train_var, lag.max=12, type="const")$selection["AIC(n)"]

# -------------------------------
# Prepare ML features: lagged values
# -------------------------------
lags <- 12
X_full <- embed(ts_all, lags+1)[, -1]
y_full <- embed(ts_all, lags+1)[,1]

train_idx <- 1:(length(y_full) - n_forecast)
X_train <- X_full[train_idx, , drop=FALSE]
y_train <- y_full[train_idx]

X_test <- X_full[-train_idx, , drop=FALSE]
y_test <- y_full[-train_idx]

# Optional: PCA to reduce collinearity
pca <- prcomp(X_train, scale.=TRUE)
X_train_pca <- pca$x
X_test_pca <- predict(pca, newdata=X_test)

# -------------------------------
# Initialize storage for rolling forecasts
# -------------------------------
rolling_preds <- data.frame(
  ARIMA = numeric(n_forecast),
  VAR = numeric(n_forecast),
  RF = numeric(n_forecast),
  Lasso = numeric(n_forecast),
  Ridge = numeric(n_forecast),
  XGBoost = numeric(n_forecast),
  LightGBM = numeric(n_forecast)
)

# -------------------------------
# Initialize ARIMA
# -------------------------------
arima_train <- ts(train_ts, frequency=12)

# Initialize VAR
var_train <- train_var

# Rolling forecast loop
for(i in 1:n_forecast){
  
  # --- ARIMA ---
  fit_arima <- auto.arima(arima_train)
  rolling_preds$ARIMA[i] <- forecast(fit_arima, h=1)$mean
  
  # Update training for next step
  arima_train <- ts(c(arima_train, y_test[i]), frequency=12)
  
  # --- VAR ---
  fit_var <- VAR(var_train, p=lag_sel, type="const")
  fc_var <- predict(fit_var, n.ahead=1)
  rolling_preds$VAR[i] <- fc_var$fcst$INDPRO_growth[1, "fcst"]
  
  var_train <- ts(rbind(var_train, test_var[i, , drop=FALSE]), frequency=12)
  
  # --- ML models ---
  # Random Forest
  rf_model <- randomForest(X_train_pca, y_train, ntree=500)
  rolling_preds$RF[i] <- predict(rf_model, X_test_pca[i,,drop=FALSE])
  
  # Lasso
  lasso_model <- cv.glmnet(scale(X_train_pca), y_train, alpha=1)
  rolling_preds$Lasso[i] <- predict(lasso_model, scale(X_test_pca[i,,drop=FALSE]), s="lambda.min")
  
  # Ridge
  ridge_model <- cv.glmnet(scale(X_train_pca), y_train, alpha=0)
  rolling_preds$Ridge[i] <- predict(ridge_model, scale(X_test_pca[i,,drop=FALSE]), s="lambda.min")
  
  # XGBoost
  xgb_model <- xgboost(data=xgb.DMatrix(X_train_pca, label=y_train), nrounds=200, objective="reg:squarederror", verbose=0)
  rolling_preds$XGBoost[i] <- predict(xgb_model, xgb.DMatrix(X_test_pca[i,,drop=FALSE]))
  
  # LightGBM
  lgb_model <- lgb.train(params=list(objective="regression", metric="rmse"),
                         data=lgb.Dataset(X_train_pca, label=y_train), nrounds=200)
  rolling_preds$LightGBM[i] <- predict(lgb_model, X_test_pca[i,,drop=FALSE])
  
  # Update ML training set
  X_train_pca <- rbind(X_train_pca, X_test_pca[i,,drop=FALSE])
  y_train <- c(y_train, y_test[i])
}

# -------------------------------
# Evaluation metrics
# -------------------------------
evaluate <- function(true, pred){
  mse <- mean((true - pred)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(true - pred))
  return(c(MSE=mse, RMSE=rmse, MAE=mae))
}

rolling_results <- sapply(rolling_preds, function(pred) evaluate(y_test, pred))
rolling_results <- as.data.frame(t(rolling_results))
print(rolling_results)

# -------------------------------
# Plot rolling forecasts vs actual
# -------------------------------
plot_df <- data.frame(
  Date = seq.Date(from=as.Date("2016-01-01"), by="month", length.out=n_forecast),
  Actual = y_test,
  rolling_preds
) %>% pivot_longer(cols=-Date, names_to="Model", values_to="Forecast")

ggplot(plot_df, aes(x=Date, y=Forecast, color=Model)) +
  geom_line(size=1) +
  geom_line(aes(y=Actual), color="black", size=1, linetype="dashed") +
  ggtitle("Rolling Forecasts (ARIMA, VAR, ML Models) vs Actual") +
  theme_minimal()

# =====================================================
# MULTI-STEP (STATIC) FORECAST FOR ML MODELS
# =====================================================

library(forecast)
library(randomForest)
library(glmnet)
library(xgboost)
library(lightgbm)
library(ggplot2)
library(dplyr)
library(tidyr)

# -------------------------------
# Prepare features for multi-step forecast
# -------------------------------
lags <- 12
X_full <- embed(ts_all, lags + 1)[, -1]
y_full <- embed(ts_all, lags + 1)[, 1]

train_idx <- 1:(length(y_full) - n_forecast)
X_train <- X_full[train_idx, , drop = FALSE]
y_train <- y_full[train_idx]

X_test <- X_full[-train_idx, , drop = FALSE]
y_test <- y_full[-train_idx]

# PCA to reduce collinearity
pca <- prcomp(X_train, scale. = TRUE)
X_train_pca <- pca$x
X_test_pca <- predict(pca, newdata = X_test)

# -------------------------------
# Train ML models on full train set
# -------------------------------
rf_model <- randomForest(X_train_pca, y_train, ntree = 500)
lasso_model <- cv.glmnet(scale(X_train_pca), y_train, alpha = 1)
ridge_model <- cv.glmnet(scale(X_train_pca), y_train, alpha = 0)
xgb_model <- xgboost(data = xgb.DMatrix(X_train_pca, label = y_train),
                     nrounds = 200, objective = "reg:squarederror", verbose = 0)
lgb_model <- lgb.train(params = list(objective = "regression", metric = "rmse"),
                       data = lgb.Dataset(X_train_pca, label = y_train), nrounds = 200)

# -------------------------------
# Predict multi-step (static)
# -------------------------------
ml_static_preds <- data.frame(
  RF = predict(rf_model, X_test_pca),
  Lasso = as.numeric(predict(lasso_model, scale(X_test_pca), s = "lambda.min")),
  Ridge = as.numeric(predict(ridge_model, scale(X_test_pca), s = "lambda.min")),
  XGBoost = predict(xgb_model, xgb.DMatrix(X_test_pca)),
  LightGBM = predict(lgb_model, X_test_pca)
)

# -------------------------------
# Evaluate
# -------------------------------
evaluate <- function(true, pred){
  mse <- mean((true - pred)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(true - pred))
  return(c(MSE = mse, RMSE = rmse, MAE = mae))
}

static_results <- sapply(ml_static_preds, function(pred) evaluate(y_test, pred))
static_results <- as.data.frame(t(static_results))
print(static_results)

# -------------------------------
# Plot static forecast vs actual
# -------------------------------
plot_df <- data.frame(
  Date = seq.Date(from = as.Date("2016-01-01"), by = "month", length.out = n_forecast),
  Actual = y_test,
  ml_static_preds
) %>% pivot_longer(cols = -Date, names_to = "Model", values_to = "Forecast")

ggplot(plot_df, aes(x = Date, y = Forecast, color = Model)) +
  geom_line(size = 1) +
  geom_line(aes(y = Actual), color = "black", size = 1, linetype = "dashed") +
  ggtitle("Multi-step (Static) Forecasts: ML Models vs Actual") +
  theme_minimal()

# =====================================================
# ML + TS FORECAST COMPARISON (STATIC & ROLLING)
# =====================================================

library(forecast)
library(vars)
library(randomForest)
library(glmnet)
library(xgboost)
library(lightgbm)
library(ggplot2)
library(dplyr)
library(tidyr)

# -------------------------------
# 1️⃣ Prepare TS data
# -------------------------------
ts_all <- ts_indpro  # original series
lags <- 12
n_forecast <- 120  # adjust to test period length

# Train/test split
train_ts <- window(ts_all, end = c(2015,12))
test_ts <- window(ts_all, start = c(2016,1))

# Lagged features for ML
X_full <- embed(ts_all, lags + 1)[, -1]
y_full <- embed(ts_all, lags + 1)[, 1]
train_idx <- 1:(length(y_full) - n_forecast)
X_train <- X_full[train_idx, , drop = FALSE]
y_train <- y_full[train_idx]
X_test <- X_full[-train_idx, , drop = FALSE]
y_test <- y_full[-train_idx]

# PCA to reduce collinearity
pca <- prcomp(X_train, scale. = TRUE)
X_train_pca <- pca$x
X_test_pca <- predict(pca, newdata = X_test)

# -------------------------------
# 2️⃣ Train ML models
# -------------------------------
rf_model <- randomForest(X_train_pca, y_train, ntree = 500)
lasso_model <- cv.glmnet(scale(X_train_pca), y_train, alpha = 1)
ridge_model <- cv.glmnet(scale(X_train_pca), y_train, alpha = 0)
xgb_model <- xgboost(data = xgb.DMatrix(X_train_pca, label = y_train),
                     nrounds = 200, objective = "reg:squarederror", verbose = 0)
lgb_model <- lgb.train(params = list(objective = "regression", metric = "rmse"),
                       data = lgb.Dataset(X_train_pca, label = y_train), nrounds = 200)

ml_models <- list(RF = rf_model, Lasso = lasso_model, Ridge = ridge_model,
                  XGBoost = xgb_model, LightGBM = lgb_model)

# -------------------------------
# 3️⃣ Multi-step static ML forecast
# -------------------------------
ml_static_preds <- data.frame(
  RF = predict(rf_model, X_test_pca),
  Lasso = as.numeric(predict(lasso_model, scale(X_test_pca), s = "lambda.min")),
  Ridge = as.numeric(predict(ridge_model, scale(X_test_pca), s = "lambda.min")),
  XGBoost = predict(xgb_model, xgb.DMatrix(X_test_pca)),
  LightGBM = predict(lgb_model, X_test_pca)
)

# -------------------------------
# 4️⃣ Rolling ML forecast
# -------------------------------
ml_rolling_preds <- data.frame(matrix(NA, nrow = n_forecast, ncol = length(ml_models)))
colnames(ml_rolling_preds) <- names(ml_models)

for(i in 1:n_forecast){
  x_input <- t(as.matrix(X_full[train_idx[length(train_idx)] + i, , drop = FALSE]))
  x_input_pca <- predict(pca, newdata = x_input)
  
  ml_rolling_preds$RF[i] <- predict(rf_model, x_input_pca)
  ml_rolling_preds$Lasso[i] <- as.numeric(predict(lasso_model, scale(x_input_pca), s = "lambda.min"))
  ml_rolling_preds$Ridge[i] <- as.numeric(predict(ridge_model, scale(x_input_pca), s = "lambda.min"))
  ml_rolling_preds$XGBoost[i] <- predict(xgb_model, xgb.DMatrix(x_input_pca))
  ml_rolling_preds$LightGBM[i] <- predict(lgb_model, x_input_pca)
  
  # Optionally update X_full for next recursive step
  if(i < n_forecast) X_full[train_idx[length(train_idx)] + i + 1, 1] <- ml_rolling_preds$RF[i]
}

# -------------------------------
# 5️⃣ ARIMA baseline
# -------------------------------
fit_arima <- auto.arima(train_ts)
fc_arima <- forecast(fit_arima, h = n_forecast)
arima_static <- as.numeric(fc_arima$mean)

# Rolling ARIMA
rolling_arima <- numeric(n_forecast)
train_roll <- train_ts
for(i in 1:n_forecast){
  fit_roll <- auto.arima(train_roll)
  rolling_arima[i] <- forecast(fit_roll, h=1)$mean
  train_roll <- ts(c(train_roll, test_ts[i]), frequency=12)
}

# -------------------------------
# 6️⃣ VAR baseline (multivariate example)
# -------------------------------
# Suppose you have VAR variables in ts_var (INDPRO_growth + macro)
train_var <- window(ts_var, end = c(2015,12))
test_var <- window(ts_var, start = c(2016,1))
lag_select <- VARselect(train_var, lag.max = 12, type="const")$selection["AIC(n)"]
fit_var <- VAR(train_var, p = lag_select, type="const")
fc_var <- predict(fit_var, n.ahead = nrow(test_var))
var_static <- ts(fc_var$fcst$INDPRO_growth[, "fcst"], start = c(2016,1), frequency=12)

# -------------------------------
# 7️⃣ Evaluation function
# -------------------------------
evaluate <- function(true, pred){
  mse <- mean((true - pred)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(true - pred))
  return(c(MSE = mse, RMSE = rmse, MAE = mae))
}

# Compute all metrics
results_table <- data.frame(
  Model = c(names(ml_models), "ARIMA", "VAR"),
  Static_MSE = c(sapply(ml_static_preds, function(pred) evaluate(y_test, pred)["MSE"]),
                 evaluate(y_test, arima_static)["MSE"],
                 evaluate(y_test, var_static)["MSE"]),
  Static_RMSE = c(sapply(ml_static_preds, function(pred) evaluate(y_test, pred)["RMSE"]),
                  evaluate(y_test, arima_static)["RMSE"],
                  evaluate(y_test, var_static)["RMSE"]),
  Static_MAE = c(sapply(ml_static_preds, function(pred) evaluate(y_test, pred)["MAE"]),
                 evaluate(y_test, arima_static)["MAE"],
                 evaluate(y_test, var_static)["MAE"]),
  Rolling_MSE = c(sapply(ml_rolling_preds, function(pred) evaluate(y_test, pred)["MSE"]),
                  evaluate(y_test, rolling_arima)["MSE"],
                  NA),  # VAR rolling not implemented here
  Rolling_RMSE = c(sapply(ml_rolling_preds, function(pred) evaluate(y_test, pred)["RMSE"]),
                   evaluate(y_test, rolling_arima)["RMSE"],
                   NA),
  Rolling_MAE = c(sapply(ml_rolling_preds, function(pred) evaluate(y_test, pred)["MAE"]),
                  evaluate(y_test, rolling_arima)["MAE"],
                  NA)
)
print(results_table)

# -------------------------------
# 8️⃣ Plot all forecasts vs actual
# -------------------------------
plot_df <- data.frame(
  Date = seq.Date(from = as.Date("2016-01-01"), by = "month", length.out = n_forecast),
  Actual = y_test
) %>%
  bind_cols(ml_static_preds %>% rename_with(~ paste0(.,"_Static"))) %>%
  bind_cols(ml_rolling_preds %>% rename_with(~ paste0(.,"_Rolling"))) %>%
  mutate(ARIMA_Static = arima_static,
         ARIMA_Rolling = rolling_arima,
         VAR_Static = var_static) %>%
  pivot_longer(cols = -Date, names_to = "Model", values_to = "Forecast")

ggplot(plot_df, aes(x = Date, y = Forecast, color = Model)) +
  geom_line(size = 1) +
  geom_line(aes(y = Actual), color = "black", size = 1, linetype = "dashed") +
  ggtitle("Time Series Forecasts: ML vs ARIMA vs VAR") +
  theme_minimal()