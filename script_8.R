# set working directory
setwd("C:/Users/russn/Downloads/")

library(readr)
library(dplyr)
library(zoo)
library(pls)
library(ggplot2)
library(forecast)
library(tidyr)
library(gridExtra)
library(tseries)
library(urca)
library(strucchange)
library(vars)

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


# =============================
# LAGGED DATA ANALYSIS
# =============================
mse  <- function(actual, pred) mean((actual - pred)^2, na.rm = TRUE)
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2, na.rm = TRUE))

# =============================
# STEP 1: CREATE LAGGED DATA
# =============================

cat("=============================\n")
cat("CREATING LAGGED DATA\n")
cat("=============================\n\n")

MAX_LAGS <- 24  # Maximum number of lags to create

# Function to create lagged variables
create_lags <- function(data, y_col, X_cols, max_lags) {
  n <- nrow(data)
  
  # Create lagged dependent variable
  y_lags <- matrix(NA, nrow = n, ncol = max_lags)
  colnames(y_lags) <- paste0("y_lag", 1:max_lags)
  
  for (lag in 1:max_lags) {
    if (lag < n) {
      y_lags[(lag+1):n, lag] <- y_col[1:(n-lag)]
    }
  }
  
  # Create lagged predictors
  X_lags_list <- list()
  for (lag in 1:max_lags) {
    X_lag <- matrix(NA, nrow = n, ncol = ncol(X_cols))
    colnames(X_lag) <- paste0(colnames(X_cols), "_lag", lag)
    
    if (lag < n) {
      X_lag[(lag+1):n, ] <- as.matrix(X_cols[1:(n-lag), ])
    }
    X_lags_list[[lag]] <- X_lag
  }
  
  # Combine all
  X_all_lags <- do.call(cbind, X_lags_list)
  
  # Combine current X, lagged Y, and lagged X
  lagged_data <- cbind(as.matrix(X_cols), y_lags, X_all_lags)
  
  return(lagged_data)
}

# Create lagged data for training
cat("Creating lagged training data...\n")
X_train_lagged <- create_lags(train_data, y_train, X_train_scaled, MAX_LAGS)

# Create lagged data for testing
cat("Creating lagged test data...\n")
X_test_lagged <- create_lags(test_data, y_test, X_test_scaled, MAX_LAGS)

# Remove rows with NA (due to lagging)
valid_train <- complete.cases(X_train_lagged)
valid_test <- complete.cases(X_test_lagged)

X_train_lagged_clean <- X_train_lagged[valid_train, ]
y_train_clean <- y_train[valid_train]

X_test_lagged_clean <- X_test_lagged[valid_test, ]
y_test_clean <- y_test[valid_test]

cat(sprintf("\nTraining samples: %d -> %d (after removing NAs)\n", 
            length(y_train), length(y_train_clean)))
cat(sprintf("Test samples: %d -> %d (after removing NAs)\n", 
            length(y_test), length(y_test_clean)))
cat(sprintf("Total features (including lags): %d\n", ncol(X_train_lagged_clean)))

# =============================
# STEP 2: OPTIMAL LAG SELECTION
# =============================

cat("\n=============================\n")
cat("OPTIMAL LAG SELECTION\n")
cat("=============================\n\n")

# 2A: AR Model - AIC-based lag selection
cat("--- AR Model Lag Selection (AIC) ---\n")

ar_model <- ar(y_train_clean, method = "yule-walker", aic = TRUE, order.max = MAX_LAGS)
optimal_ar_lag <- ar_model$order
cat("Optimal AR lag (p):", optimal_ar_lag, "\n")
cat("AIC:", ar_model$aic, "\n\n")

# 2B: PCR with lagged data - CV-based component selection
cat("--- PCR with Lagged Data (CV) ---\n")
cat("Testing ALL lag values from 1 to", MAX_LAGS, "\n\n")
set.seed(42)

# Try ALL lag values from 1 to MAX_LAGS
lag_options <- 1:MAX_LAGS
pcr_cv_results <- data.frame()
pls_cv_results <- data.frame()

# Option for including lagged X features
INCLUDE_LAGGED_X <- TRUE  # Set to FALSE to only use lagged Y
cat("Include lagged X features:", INCLUDE_LAGGED_X, "\n\n")

for (n_lags in lag_options) {
  # Select lagged Y features
  y_lag_cols <- paste0("y_lag", 1:n_lags)
  
  # Include lagged X features
  if (INCLUDE_LAGGED_X) {
    X_lag_cols <- character()
    for (lag in 1:n_lags) {
      X_lag_cols <- c(X_lag_cols, paste0(colnames(X_train_scaled), "_lag", lag))
    }
    # Combine current X, lagged Y, and lagged X
    X_subset <- X_train_lagged_clean[, c(colnames(X_train_scaled), y_lag_cols, X_lag_cols)]
  } else {
    # Only current X and lagged Y
    X_subset <- X_train_lagged_clean[, c(colnames(X_train_scaled), y_lag_cols)]
  }
  
  # PCR
  pcr_temp <- pcr(y_train_clean ~ X_subset, validation = "CV", segments = 10)
  min_rmsep <- min(RMSEP(pcr_temp)$val[1, , -1])
  best_ncomp <- which.min(RMSEP(pcr_temp)$val[1, , -1])
  
  pcr_cv_results <- rbind(pcr_cv_results, data.frame(
    n_lags = n_lags,
    best_ncomp = best_ncomp,
    cv_rmsep = min_rmsep,
    n_features = ncol(X_subset)
  ))
  
  # PLS
  pls_temp <- plsr(y_train_clean ~ X_subset, validation = "CV", segments = 10)
  min_rmsep_pls <- min(RMSEP(pls_temp)$val[1, , -1])
  best_ncomp_pls <- which.min(RMSEP(pls_temp)$val[1, , -1])
  
  pls_cv_results <- rbind(pls_cv_results, data.frame(
    n_lags = n_lags,
    best_ncomp = best_ncomp_pls,
    cv_rmsep = min_rmsep_pls,
    n_features = ncol(X_subset)
  ))
  
  cat(sprintf("Lags=%2d (features=%4d): PCR RMSEP=%.4f (ncomp=%2d), PLS RMSEP=%.4f (ncomp=%2d)\n",
              n_lags, ncol(X_subset), min_rmsep, best_ncomp, 
              min_rmsep_pls, best_ncomp_pls))
}

# Select optimal number of lags based on minimum CV RMSEP
optimal_pcr_lags <- pcr_cv_results$n_lags[which.min(pcr_cv_results$cv_rmsep)]
optimal_pls_lags <- pls_cv_results$n_lags[which.min(pls_cv_results$cv_rmsep)]

cat("\n--- Optimal Lag Selection Summary ---\n")
cat("AR (AIC):", optimal_ar_lag, "lags\n")
cat("PCR (CV):", optimal_pcr_lags, "lags,", 
    pcr_cv_results$best_ncomp[pcr_cv_results$n_lags == optimal_pcr_lags], 
    "components\n")
cat("PLS (CV):", optimal_pls_lags, "lags,", 
    pls_cv_results$best_ncomp[pls_cv_results$n_lags == optimal_pls_lags], 
    "components\n\n")

# Show detailed results table
cat("--- Detailed PCR Results ---\n")
print(pcr_cv_results)

cat("\n--- Detailed PLS Results ---\n")
print(pls_cv_results)

# Plot CV results with number of components
p_cv1 <- ggplot() +
  geom_line(data = pcr_cv_results, aes(x = n_lags, y = cv_rmsep, color = "PCR"), 
            linewidth = 1) +
  geom_point(data = pcr_cv_results, aes(x = n_lags, y = cv_rmsep, color = "PCR"), 
             size = 3) +
  geom_line(data = pls_cv_results, aes(x = n_lags, y = cv_rmsep, color = "PLS"), 
            linewidth = 1) +
  geom_point(data = pls_cv_results, aes(x = n_lags, y = cv_rmsep, color = "PLS"), 
             size = 3) +
  geom_vline(xintercept = optimal_pcr_lags, linetype = "dashed", 
             color = "red", alpha = 0.5) +
  geom_vline(xintercept = optimal_pls_lags, linetype = "dashed", 
             color = "blue", alpha = 0.5) +
  theme_minimal() +
  labs(title = "CV RMSEP vs Number of Lags",
       subtitle = "Dashed lines show optimal lag selection",
       x = "Number of Lags",
       y = "Cross-Validation RMSEP",
       color = "Model") +
  theme(legend.position = "bottom")
print(p_cv1)

# Plot number of components selected
p_cv2 <- ggplot() +
  geom_line(data = pcr_cv_results, aes(x = n_lags, y = best_ncomp, color = "PCR"), 
            linewidth = 1) +
  geom_point(data = pcr_cv_results, aes(x = n_lags, y = best_ncomp, color = "PCR"), 
             size = 3) +
  geom_line(data = pls_cv_results, aes(x = n_lags, y = best_ncomp, color = "PLS"), 
            linewidth = 1) +
  geom_point(data = pls_cv_results, aes(x = n_lags, y = best_ncomp, color = "PLS"), 
             size = 3) +
  theme_minimal() +
  labs(title = "Optimal Number of Components vs Lags",
       x = "Number of Lags",
       y = "Number of Components Selected",
       color = "Model") +
  theme(legend.position = "bottom")
print(p_cv2)

# Plot number of features
p_cv3 <- ggplot(pcr_cv_results, aes(x = n_lags, y = n_features)) +
  geom_line(linewidth = 1, color = "steelblue") +
  geom_point(size = 3, color = "steelblue") +
  theme_minimal() +
  labs(title = "Total Number of Features vs Lags",
       subtitle = paste("Including lagged X:", INCLUDE_LAGGED_X),
       x = "Number of Lags",
       y = "Total Features")
print(p_cv3)

# =============================
# STEP 3: FIT FINAL MODELS
# =============================

cat("\n=============================\n")
cat("FITTING FINAL MODELS\n")
cat("=============================\n\n")

# Prepare data with optimal lags
y_lag_cols_pcr <- paste0("y_lag", 1:optimal_pcr_lags)
if (INCLUDE_LAGGED_X) {
  X_lag_cols_pcr <- character()
  for (lag in 1:optimal_pcr_lags) {
    X_lag_cols_pcr <- c(X_lag_cols_pcr, paste0(colnames(X_train_scaled), "_lag", lag))
  }
  X_train_pcr <- X_train_lagged_clean[, c(colnames(X_train_scaled), 
                                          y_lag_cols_pcr, X_lag_cols_pcr)]
  X_test_pcr <- X_test_lagged_clean[, c(colnames(X_train_scaled), 
                                        y_lag_cols_pcr, X_lag_cols_pcr)]
} else {
  X_train_pcr <- X_train_lagged_clean[, c(colnames(X_train_scaled), y_lag_cols_pcr)]
  X_test_pcr <- X_test_lagged_clean[, c(colnames(X_train_scaled), y_lag_cols_pcr)]
}

y_lag_cols_pls <- paste0("y_lag", 1:optimal_pls_lags)
if (INCLUDE_LAGGED_X) {
  X_lag_cols_pls <- character()
  for (lag in 1:optimal_pls_lags) {
    X_lag_cols_pls <- c(X_lag_cols_pls, paste0(colnames(X_train_scaled), "_lag", lag))
  }
  X_train_pls <- X_train_lagged_clean[, c(colnames(X_train_scaled), 
                                          y_lag_cols_pls, X_lag_cols_pls)]
  X_test_pls <- X_test_lagged_clean[, c(colnames(X_train_scaled), 
                                        y_lag_cols_pls, X_lag_cols_pls)]
} else {
  X_train_pls <- X_train_lagged_clean[, c(colnames(X_train_scaled), y_lag_cols_pls)]
  X_test_pls <- X_test_lagged_clean[, c(colnames(X_train_scaled), y_lag_cols_pls)]
}

cat(sprintf("\nFeature dimensions:\n"))
cat(sprintf("  PCR: %d features (%d lags)\n", 
            ncol(X_train_pcr), optimal_pcr_lags))
cat(sprintf("  PLS: %d features (%d lags)\n", 
            ncol(X_train_pls), optimal_pls_lags))

# Fit PCR
cat("\nFitting PCR model...\n")
pcr_model <- pcr(y_train_clean ~ X_train_pcr, validation = "CV")
best_ncomp_pcr <- which.min(RMSEP(pcr_model)$val[1, , -1])
cat("PCR lags:", optimal_pcr_lags, "\n")
cat("PCR components:", best_ncomp_pcr, "\n")

# Show validation plot
validationplot(pcr_model, val.type = "MSEP", main = "PCR Cross-Validation")

# Fit PLS
cat("\nFitting PLS model...\n")
pls_model <- plsr(y_train_clean ~ X_train_pls, validation = "CV")
best_ncomp_pls <- which.min(RMSEP(pls_model)$val[1, , -1])
cat("PLS lags:", optimal_pls_lags, "\n")
cat("PLS components:", best_ncomp_pls, "\n")

# Show validation plot
validationplot(pls_model, val.type = "MSEP", main = "PLS Cross-Validation")

# AR model already fitted
cat("\nAR lag order:", optimal_ar_lag, "\n")

# Show which features are most important in first few components
cat("\n--- PCR Loading Analysis ---\n")
cat("Top 10 features in first component:\n")
pcr_loadings <- pcr_model$loadings[, 1]
top_pcr <- sort(abs(pcr_loadings), decreasing = TRUE)[1:10]
print(names(top_pcr))

cat("\n--- PLS Loading Analysis ---\n")
cat("Top 10 features in first component:\n")
pls_loadings <- pls_model$loadings[, 1]
top_pls <- sort(abs(pls_loadings), decreasing = TRUE)[1:10]
print(names(top_pls))

# =============================
# STEP 4: PCA ANALYSIS
# =============================

cat("\n=============================\n")
cat("PCA ON LAGGED DATA\n")
cat("=============================\n\n")

pca_result <- prcomp(X_train_lagged_clean, scale. = FALSE)
var_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
cum_var <- cumsum(var_explained)

# Find number of components explaining 90% variance
n_comp_90 <- which(cum_var >= 0.90)[1]
cat(sprintf("Components needed for 90%% variance: %d\n", n_comp_90))
cat(sprintf("First 10 components explain: %.2f%%\n", 100 * cum_var[10]))

# Scree plot
scree_data <- data.frame(
  PC = 1:min(50, length(var_explained)),
  Variance = var_explained[1:min(50, length(var_explained))],
  Cumulative = cum_var[1:min(50, length(var_explained))]
)

p_scree <- ggplot(scree_data, aes(x = PC, y = Variance)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = 0.01, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(title = "Scree Plot - PCA on Lagged Data",
       x = "Principal Component",
       y = "Proportion of Variance Explained") +
  xlim(1, 50)
print(p_scree)

# =============================
# STEP 5: FORECASTING
# =============================

cat("\n=============================\n")
cat("GENERATING FORECASTS\n")
cat("=============================\n\n")

horizons <- c(1, 3, 6, 12)
forecast_results <- data.frame()
all_forecasts <- list()

for (h in horizons) {
  cat(sprintf("\n--- Horizon: %d ---\n", h))
  
  n_available <- length(y_test_clean) - h + 1
  if (n_available <= 0) {
    cat("Insufficient data for this horizon\n")
    next
  }
  
  y_test_h <- y_test_clean[h:length(y_test_clean)]
  
  ar_pred <- numeric(n_available)
  pcr_pred <- numeric(n_available)
  pls_pred <- numeric(n_available)
  
  # Recursive forecasting
  for (i in 1:n_available) {
    # Expanding window
    if (i == 1) {
      y_window <- y_train_clean
      X_window_pcr <- X_train_pcr
      X_window_pls <- X_train_pls
    } else {
      n_obs <- i - 1
      y_window <- c(y_train_clean, y_test_clean[1:n_obs])
      X_window_pcr <- rbind(X_train_pcr, X_test_pcr[1:n_obs, , drop = FALSE])
      X_window_pls <- rbind(X_train_pls, X_test_pls[1:n_obs, , drop = FALSE])
    }
    
    # AR forecast
    ar_temp <- ar(y_window, method = "mle", order.max = optimal_ar_lag, aic = FALSE)
    ar_forecast <- predict(ar_temp, n.ahead = h)
    ar_pred[i] <- ar_forecast$pred[h]
    
    # PCR forecast
    pcr_temp <- pcr(y_window ~ X_window_pcr, ncomp = best_ncomp_pcr, validation = "none")
    pcr_forecast <- predict(pcr_temp, 
                            newdata = data.frame(X_window_pcr = X_test_pcr[i, , drop = FALSE]),
                            ncomp = best_ncomp_pcr)
    pcr_pred[i] <- as.numeric(pcr_forecast)
    
    # PLS forecast
    pls_temp <- plsr(y_window ~ X_window_pls, ncomp = best_ncomp_pls, validation = "none")
    pls_forecast <- predict(pls_temp,
                            newdata = data.frame(X_window_pls = X_test_pls[i, , drop = FALSE]),
                            ncomp = best_ncomp_pls)
    pls_pred[i] <- as.numeric(pls_forecast)
  }
  
  # Store forecasts
  all_forecasts[[paste0("h", h)]] <- list(
    actual = y_test_h,
    ar = ar_pred,
    pcr = pcr_pred,
    pls = pls_pred
  )
  
  # Calculate metrics
  ar_mse <- mse(y_test_h, ar_pred)
  pcr_mse <- mse(y_test_h, pcr_pred)
  pls_mse <- mse(y_test_h, pls_pred)
  
  cat(sprintf("MSE - AR: %.4f, PCR: %.4f, PLS: %.4f\n", 
              ar_mse, pcr_mse, pls_mse))
  
  forecast_results <- rbind(forecast_results, data.frame(
    Horizon = h,
    AR_MSE = ar_mse,
    PCR_MSE = pcr_mse,
    PLS_MSE = pls_mse,
    AR_RMSE = sqrt(ar_mse),
    PCR_RMSE = sqrt(pcr_mse),
    PLS_RMSE = sqrt(pls_mse)
  ))
}

# =============================
# STEP 6: PLOT MSE AND FORECASTS
# =============================

cat("\n=============================\n")
cat("VISUALIZATIONS\n")
cat("=============================\n\n")

# MSE Comparison
mse_plot_data <- forecast_results %>%
  dplyr::select(Horizon, AR_MSE, PCR_MSE, PLS_MSE) %>%
  pivot_longer(cols = -Horizon, names_to = "Model", values_to = "MSE") %>%
  mutate(Model = gsub("_MSE", "", Model))

p_mse <- ggplot(mse_plot_data, aes(x = factor(Horizon), y = MSE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "MSE Comparison: AR vs PCR vs PLS (with Lagged Features)",
       x = "Forecast Horizon",
       y = "Mean Squared Error") +
  theme(legend.position = "bottom")
print(p_mse)

# Forecast plots
plot_list <- list()
for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(all_forecasts)) next
  
  fc <- all_forecasts[[key]]
  dates <- test_data$sasdate[valid_test][h:length(y_test_clean)]
  
  plot_data <- data.frame(
    Date = dates,
    Actual = fc$actual,
    AR = fc$ar,
    PCR = fc$pcr,
    PLS = fc$pls
  ) %>%
    pivot_longer(cols = c(Actual, AR, PCR, PLS),
                 names_to = "Series",
                 values_to = "Value")
  
  p <- ggplot(plot_data, aes(x = Date, y = Value, color = Series)) +
    geom_line(aes(linetype = Series), linewidth = 0.7) +
    scale_linetype_manual(values = c("Actual" = "solid", "AR" = "dashed",
                                     "PCR" = "dotted", "PLS" = "dotdash")) +
    theme_minimal() +
    labs(title = paste("Forecasts - Horizon", h),
         x = "Date", y = "INDPRO Growth (%)") +
    theme(legend.position = "bottom")
  
  plot_list[[key]] <- p
}

do.call(grid.arrange, c(plot_list, ncol = 2))

# =============================
# STEP 7: STATISTICAL TESTS
# =============================

cat("\n=============================\n")
cat("STATISTICAL TESTS\n")
cat("=============================\n\n")

# Theil's U Statistic
cat("--- THEIL'S U STATISTIC ---\n")
cat("(U < 1 indicates model outperforms AR baseline)\n\n")

theil_u <- function(actual, pred, naive_pred) {
  numerator <- sqrt(mse(actual, pred))
  denominator <- sqrt(mse(actual, naive_pred))
  return(numerator / denominator)
}

theil_results <- data.frame()

for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(all_forecasts)) next
  
  fc <- all_forecasts[[key]]
  
  u_pcr <- theil_u(fc$actual, fc$pcr, fc$ar)
  u_pls <- theil_u(fc$actual, fc$pls, fc$ar)
  
  theil_results <- rbind(theil_results, data.frame(
    Horizon = h,
    PCR_vs_AR = u_pcr,
    PLS_vs_AR = u_pls
  ))
  
  cat(sprintf("Horizon %2d: PCR = %.4f, PLS = %.4f\n", h, u_pcr, u_pls))
}

# Diebold-Mariano Test
cat("\n--- DIEBOLD-MARIANO TEST ---\n")
cat("(p-value < 0.05 indicates significant difference)\n\n")

dm_results <- data.frame()

for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(all_forecasts)) next
  
  fc <- all_forecasts[[key]]
  
  # PCR vs AR
  dm_pcr <- dm.test(fc$ar - fc$actual,
                    fc$pcr - fc$actual,
                    alternative = "two.sided", h = h)
  
  # PLS vs AR
  dm_pls <- dm.test(fc$ar - fc$actual,
                    fc$pls - fc$actual,
                    alternative = "two.sided", h = h)
  
  dm_results <- rbind(dm_results, data.frame(
    Horizon = h,
    PCR_stat = dm_pcr$statistic,
    PCR_pval = dm_pcr$p.value,
    PLS_stat = dm_pls$statistic,
    PLS_pval = dm_pls$p.value
  ))
  
  cat(sprintf("Horizon %2d:\n", h))
  cat(sprintf("  PCR vs AR: DM = %7.3f, p-value = %.4f %s\n",
              dm_pcr$statistic, dm_pcr$p.value,
              ifelse(dm_pcr$p.value < 0.05, "***", "")))
  cat(sprintf("  PLS vs AR: DM = %7.3f, p-value = %.4f %s\n",
              dm_pls$statistic, dm_pls$p.value,
              ifelse(dm_pls$p.value < 0.05, "***", "")))
}

# =============================
# SUMMARY TABLES
# =============================

cat("\n=============================\n")
cat("SUMMARY RESULTS\n")
cat("=============================\n\n")

cat("--- FORECAST PERFORMANCE ---\n")
print(forecast_results)

cat("\n--- THEIL'S U STATISTIC ---\n")
print(theil_results)

cat("\n--- DIEBOLD-MARIANO TEST ---\n")
print(dm_results)

# Relative performance
forecast_results$PCR_vs_AR <- forecast_results$PCR_MSE / forecast_results$AR_MSE
forecast_results$PLS_vs_AR <- forecast_results$PLS_MSE / forecast_results$AR_MSE

cat("\n--- RELATIVE PERFORMANCE (ratio < 1 = better than AR) ---\n")
print(forecast_results[, c("Horizon", "PCR_vs_AR", "PLS_vs_AR")])

cat("\n=============================\n")
cat("KEY FINDINGS\n")
cat("=============================\n\n")

cat("FEATURE CONFIGURATION:\n")
cat(sprintf("  Include lagged X features: %s\n", INCLUDE_LAGGED_X))
cat(sprintf("  Original features: %d\n", ncol(X_train_scaled)))
cat(sprintf("  Total features with lags: %d\n", ncol(X_train_lagged_clean)))

cat("\nOPTIMAL LAGS:\n")
cat(sprintf("  AR (AIC):  %d lags\n", optimal_ar_lag))
cat(sprintf("  PCR (CV):  %d lags, %d components\n", 
            optimal_pcr_lags, best_ncomp_pcr))
cat(sprintf("  PLS (CV):  %d lags, %d components\n", 
            optimal_pls_lags, best_ncomp_pls))

# =============================
# STATIONARITY TESTS
# =============================

cat("=============================\n")
cat("STATIONARITY TESTS\n")
cat("=============================\n\n")

# Create time series object
ts_train <- ts(y_train, frequency = 12)
ts_full <- ts(c(y_train, y_test), frequency = 12)

# 1. Augmented Dickey-Fuller (ADF) Test
cat("--- Augmented Dickey-Fuller Test ---\n")
adf_test <- adf.test(y_train)
cat("ADF Statistic:", adf_test$statistic, "\n")
cat("p-value:", adf_test$p.value, "\n")
cat("Interpretation:", ifelse(adf_test$p.value < 0.05, 
                              "STATIONARY (reject unit root)",
                              "NON-STATIONARY (fail to reject unit root)"), "\n\n")

# 2. KPSS Test (null: stationary)
cat("--- KPSS Test ---\n")
kpss_test <- kpss.test(y_train)
cat("KPSS Statistic:", kpss_test$statistic, "\n")
cat("p-value:", kpss_test$p.value, "\n")
cat("Interpretation:", ifelse(kpss_test$p.value > 0.05,
                              "STATIONARY (fail to reject stationarity)",
                              "NON-STATIONARY (reject stationarity)"), "\n\n")

# 3. Phillips-Perron Test
cat("--- Phillips-Perron Test ---\n")
pp_test <- pp.test(y_train)
cat("PP Statistic:", pp_test$statistic, "\n")
cat("p-value:", pp_test$p.value, "\n")
cat("Interpretation:", ifelse(pp_test$p.value < 0.05,
                              "STATIONARY (reject unit root)",
                              "NON-STATIONARY (fail to reject unit root)"), "\n\n")

# 4. Summary
cat("--- SUMMARY ---\n")
stationarity_summary <- data.frame(
  Test = c("ADF", "KPSS", "PP"),
  Statistic = c(adf_test$statistic, kpss_test$statistic, pp_test$statistic),
  P_Value = c(adf_test$p.value, kpss_test$p.value, pp_test$p.value),
  Conclusion = c(
    ifelse(adf_test$p.value < 0.05, "Stationary", "Non-Stationary"),
    ifelse(kpss_test$p.value > 0.05, "Stationary", "Non-Stationary"),
    ifelse(pp_test$p.value < 0.05, "Stationary", "Non-Stationary")
  )
)
print(stationarity_summary)

# Plot time series
p_ts <- ggplot(data.frame(Time = 1:length(y_train), Value = y_train),
               aes(x = Time, y = Value)) +
  geom_line(color = "steelblue") +
  theme_minimal() +
  labs(title = "INDPRO Growth Rate Time Series",
       x = "Time", y = "Growth Rate (%)")
print(p_ts)

# =============================
# DECOMPOSITION
# =============================

cat("\n=============================\n")
cat("TIME SERIES DECOMPOSITION\n")
cat("=============================\n\n")

# STL Decomposition (Seasonal-Trend decomposition using LOESS)
stl_decomp <- stl(ts_train, s.window = "periodic")
cat("STL Decomposition completed\n")

# Plot decomposition
plot(stl_decomp, main = "STL Decomposition of INDPRO Growth")

# Extract components
trend <- stl_decomp$time.series[, "trend"]
seasonal <- stl_decomp$time.series[, "seasonal"]
remainder <- stl_decomp$time.series[, "remainder"]

# Calculate variance explained
var_total <- var(y_train, na.rm = TRUE)
var_trend <- var(trend, na.rm = TRUE)
var_seasonal <- var(seasonal, na.rm = TRUE)
var_remainder <- var(remainder, na.rm = TRUE)

cat("\n--- Variance Decomposition ---\n")
cat(sprintf("Trend:     %.2f%% of total variance\n", 100 * var_trend / var_total))
cat(sprintf("Seasonal:  %.2f%% of total variance\n", 100 * var_seasonal / var_total))
cat(sprintf("Remainder: %.2f%% of total variance\n", 100 * var_remainder / var_total))

# Test for seasonality
cat("\n--- Seasonality Test ---\n")
if (frequency(ts_train) > 1) {
  seasonal_strength <- 1 - var(remainder, na.rm = TRUE) / 
    var(seasonal + remainder, na.rm = TRUE)
  cat("Seasonal Strength:", round(seasonal_strength, 4), "\n")
  cat("Interpretation:", ifelse(seasonal_strength > 0.6,
                                "STRONG seasonality",
                                ifelse(seasonal_strength > 0.3,
                                       "MODERATE seasonality",
                                       "WEAK seasonality")), "\n")
}

# =============================
# STRUCTURAL BREAK TESTS
# =============================

cat("\n=============================\n")
cat("STRUCTURAL BREAK TESTS\n")
cat("=============================\n\n")
# Bai-Perron Test (multiple breaks)
cat("--- Bai-Perron Multiple Breakpoint Test ---\n")
bp_test <- breakpoints(y_train ~ 1, h = 0.1)
cat("Number of breakpoints detected:", length(bp_test$breakpoints), "\n")

if (length(bp_test$breakpoints) > 0 && !any(is.na(bp_test$breakpoints))) {
  cat("Breakpoint locations:\n")
  print(bp_test$breakpoints)
  
  # Plot breakpoints
  plot(bp_test, main = "Bai-Perron Breakpoint Test")
  
  # Plot with breaks
  bp_dates <- bp_test$breakpoints
  plot_data <- data.frame(Time = 1:length(y_train), Value = y_train)
  
  p_breaks <- ggplot(plot_data, aes(x = Time, y = Value)) +
    geom_line(color = "steelblue") +
    geom_vline(xintercept = bp_dates, color = "red", linetype = "dashed") +
    theme_minimal() +
    labs(title = "Detected Structural Breaks",
         x = "Time", y = "INDPRO Growth (%)")
  print(p_breaks)
} else {
  cat("No significant breakpoints detected\n")
}
