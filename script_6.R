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
# PRINCIPAL COMPONENT ANALYSIS (PCA)
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
# PRINCIPAL COMPONENT REGRESSION (PCR)
# =============================
set.seed(42)
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
# PARTIAL LEAST SQUARES (PLS)
# =============================
set.seed(42)
pls_model <- plsr(y_train ~ X_train_scaled, validation = "CV")

best_ncomp_pls <- which.min(RMSEP(pls_model)$val[1, , -1])
cat("Optimal number of components for PLS:", best_ncomp_pls, "\n")

validationplot(pls_model, val.type = "MSEP")

pls_pred_in  <- predict(pls_model, ncomp = best_ncomp_pls)
pls_pred_out <- predict(pls_model, newdata = data.frame(X_train_scaled = X_test_scaled),
                        ncomp = best_ncomp_pls)

# =============================
# PERFORMANCE EVALUATION
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
# COMPREHENSIVE FORECASTING COMPARISON
# Supports: Static, Recursive, Rolling Window, Direct Multi-step
# =============================
horizons <- c(1, 3, 6, 12)
window_size <- 120  # For rolling window approach

# =============================
# BASELINE AR MODEL
# =============================
cat("=============================\n")
cat("FITTING BASELINE AR MODEL\n")
cat("=============================\n")

ar_model <- ar(y_train, method = "mle", aic = TRUE)
best_p <- ar_model$order
cat("Optimal AR lag (p) selected by AIC:", best_p, "\n")

# =============================
# OPTIMAL COMPONENT SELECTION
# =============================
cat("\n=============================\n")
cat("SELECTING OPTIMAL COMPONENTS\n")
cat("=============================\n")

set.seed(42)
pcr_model <- pcr(y_train ~ X_train_scaled, validation = "CV")
best_ncomp_pcr <- which.min(RMSEP(pcr_model)$val[1, , -1])
cat("Optimal PCR components:", best_ncomp_pcr, "\n")

pls_model <- plsr(y_train ~ X_train_scaled, validation = "CV")
best_ncomp_pls <- which.min(RMSEP(pls_model)$val[1, , -1])
cat("Optimal PLS components:", best_ncomp_pls, "\n")

# =============================
# FORECASTING FUNCTIONS
# =============================

# Function to generate forecasts with different strategies
generate_forecasts <- function(strategy = "static", horizon = 1, use_direct = FALSE) {
  
  n_available <- length(y_test) - horizon + 1
  if (n_available <= 0) return(NULL)
  
  y_test_h <- y_test[horizon:length(y_test)]
  
  ar_pred <- numeric(n_available)
  pcr_pred <- numeric(n_available)
  pls_pred <- numeric(n_available)
  
  for (i in 1:n_available) {
    if (strategy == "static") {
      # Static: always use original training data only
      y_window <- y_train
      X_window <- X_train_scaled
      
    } else if (strategy == "recursive") {
      # Recursive: expanding window (add all observed test data)
      if (i == 1) {
        y_window <- y_train
        X_window <- X_train_scaled
      } else {
        n_obs <- i - 1 + horizon - 1
        y_window <- c(y_train, y_test[1:n_obs])
        X_window <- rbind(X_train_scaled, X_test_scaled[1:n_obs, , drop = FALSE])
      }
      
    } else if (strategy == "rolling") {
      # Rolling: fixed window size, slides forward
      if (i == 1) {
        y_window <- tail(y_train, window_size)
        X_window <- tail(X_train_scaled, window_size)
      } else {
        n_obs <- i - 1
        y_all <- c(y_train, y_test[1:n_obs])
        X_all <- rbind(X_train_scaled, X_test_scaled[1:n_obs, , drop = FALSE])
        y_window <- tail(y_all, window_size)
        X_window <- tail(X_all, window_size)
      }
    }
    
    if (use_direct && horizon > 1) {
      # Direct: train to predict y[t+h] from X[t]
      # Need to align: y_direct[j] corresponds to y_window[j+h]
      if (length(y_window) > horizon) {
        y_direct <- y_window[-(1:(horizon-1))]  # Remove first h-1 observations
        X_direct <- X_window[1:(nrow(X_window)-(horizon-1)), , drop = FALSE]
      } else {
        y_direct <- y_window
        X_direct <- X_window
      }
    } else {
      y_direct <- y_window
      X_direct <- X_window
    }
    
    # ===========================
    # AR MODEL
    # ===========================
    if (strategy == "static" && i > 1) {
      # For static AR, just extend the original forecast
      max_ahead <- i + horizon - 1
      ar_forecast <- predict(ar_model, n.ahead = max_ahead)
      ar_pred[i] <- ar_forecast$pred[max_ahead]
    } else {
      # Fit AR on window and forecast h steps
      ar_temp <- ar(y_window, method = "mle", order.max = best_p, aic = FALSE)
      ar_forecast <- predict(ar_temp, n.ahead = horizon)
      ar_pred[i] <- ar_forecast$pred[horizon]
    }
    
    # ===========================
    # PCR MODEL
    # ===========================
    if (strategy == "static" && i > 1) {
      # For static, use original model
      pcr_forecast <- predict(pcr_model,
                              newdata = data.frame(X_train_scaled = X_test_scaled[i, , drop = FALSE]),
                              ncomp = best_ncomp_pcr)
      pcr_pred[i] <- as.numeric(pcr_forecast)
    } else {
      # Refit model
      if (length(y_direct) > 10 && nrow(X_direct) > 10) {
        pcr_temp <- pcr(y_direct ~ X_direct, validation = "CV")
        ncomp <- which.min(RMSEP(pcr_temp)$val[1, , -1])
        pcr_forecast <- predict(pcr_temp,
                                newdata = data.frame(X_direct = X_test_scaled[i, , drop = FALSE]),
                                ncomp = ncomp)
        pcr_pred[i] <- as.numeric(pcr_forecast)
      } else {
        pcr_pred[i] <- mean(y_window)  # Fallback to mean
      }
    }
    
    # ===========================
    # PLS MODEL
    # ===========================
    if (strategy == "static" && i > 1) {
      # For static, use original model
      pls_forecast <- predict(pls_model,
                              newdata = data.frame(X_train_scaled = X_test_scaled[i, , drop = FALSE]),
                              ncomp = best_ncomp_pls)
      pls_pred[i] <- as.numeric(pls_forecast)
    } else {
      # Refit model
      if (length(y_direct) > 10 && nrow(X_direct) > 10) {
        pls_temp <- plsr(y_direct ~ X_direct, validation = "CV")
        ncomp <- which.min(RMSEP(pls_temp)$val[1, , -1])
        pls_forecast <- predict(pls_temp,
                                newdata = data.frame(X_direct = X_test_scaled[i, , drop = FALSE]),
                                ncomp = ncomp)
        pls_pred[i] <- as.numeric(pls_forecast)
      } else {
        pls_pred[i] <- mean(y_window)  # Fallback to mean
      }
    }
  }
  
  return(list(
    ar = ar_pred,
    pcr = pcr_pred,
    pls = pls_pred,
    actual = y_test_h
  ))
}

# =============================
# GENERATE ALL FORECASTS
# =============================

strategies <- c("static", "recursive", "rolling")
forecast_types <- c("iterative", "direct")

all_forecasts <- list()
all_metrics <- data.frame()

cat("\n=============================\n")
cat("GENERATING FORECASTS\n")
cat("=============================\n")

for (strategy in strategies) {
  cat("\n--- Strategy:", toupper(strategy), "---\n")
  
  for (ftype in forecast_types) {
    use_direct <- (ftype == "direct")
    label <- paste0(strategy, "_", ftype)
    
    cat("\nForecast type:", ftype, "\n")
    
    for (h in horizons) {
      forecasts <- generate_forecasts(strategy = strategy, 
                                      horizon = h, 
                                      use_direct = use_direct)
      
      if (is.null(forecasts)) next
      
      # Store forecasts
      key <- paste0(label, "_h", h)
      all_forecasts[[key]] <- forecasts
      
      # Calculate metrics
      metrics <- data.frame(
        Strategy = strategy,
        Type = ftype,
        Horizon = h,
        AR_MSE = mse(forecasts$actual, forecasts$ar),
        PCR_MSE = mse(forecasts$actual, forecasts$pcr),
        PLS_MSE = mse(forecasts$actual, forecasts$pls),
        AR_RMSE = rmse(forecasts$actual, forecasts$ar),
        PCR_RMSE = rmse(forecasts$actual, forecasts$pcr),
        PLS_RMSE = rmse(forecasts$actual, forecasts$pls)
      )
      
      all_metrics <- rbind(all_metrics, metrics)
      
      cat(sprintf("  h=%2d: AR=%.4f, PCR=%.4f, PLS=%.4f\n", 
                  h, metrics$AR_MSE, metrics$PCR_MSE, metrics$PLS_MSE))
    }
  }
}

# =============================
# STATISTICAL TESTS
# =============================

cat("\n=============================\n")
cat("STATISTICAL TESTS\n")
cat("=============================\n")

theil_results <- data.frame()
dm_results <- data.frame()

for (strategy in strategies) {
  #cat("\n--- Strategy:", toupper(strategy), "---\n")
  
  for (ftype in forecast_types) {
    label <- paste0(strategy, "_", ftype)
    #cat("\nType:", ftype, "\n")
    
    for (h in horizons) {
      key <- paste0(label, "_h", h)
      if (!key %in% names(all_forecasts)) next
      
      forecasts <- all_forecasts[[key]]
      
      # Theil's U
      u_pcr <- sqrt(mse(forecasts$actual, forecasts$pcr)) / 
        sqrt(mse(forecasts$actual, forecasts$ar))
      u_pls <- sqrt(mse(forecasts$actual, forecasts$pls)) / 
        sqrt(mse(forecasts$actual, forecasts$ar))
      
      theil_results <- rbind(theil_results, data.frame(
        Strategy = strategy,
        Type = ftype,
        Horizon = h,
        PCR_vs_AR = u_pcr,
        PLS_vs_AR = u_pls
      ))
      
      # Diebold-Mariano test
      dm_pcr <- dm.test(forecasts$ar - forecasts$actual,
                        forecasts$pcr - forecasts$actual,
                        alternative = "two.sided", h = h)
      dm_pls <- dm.test(forecasts$ar - forecasts$actual,
                        forecasts$pls - forecasts$actual,
                        alternative = "two.sided", h = h)
      
      dm_results <- rbind(dm_results, data.frame(
        Strategy = strategy,
        Type = ftype,
        Horizon = h,
        PCR_stat = dm_pcr$statistic,
        PCR_pval = dm_pcr$p.value,
        PLS_stat = dm_pls$statistic,
        PLS_pval = dm_pls$p.value
      ))
      
      #cat(sprintf("  h=%2d: Diebold Mariano - PCR=%.4f, PLS=%.4f\n", h, dm_pcr, dm_pls))
      #cat(sprintf("  h=%2d: Theil U - PCR=%.4f, PLS=%.4f\n", h, u_pcr, u_pls))
    }
  }
}

# =============================
# VISUALIZATIONS
# =============================

cat("\n=============================\n")
cat("VISUALIZATIONS\n")
cat("=============================\n")

# MSE comparison across all strategies
mse_data <- all_metrics %>%
  dplyr::select(Strategy, Type, Horizon, AR_MSE, PCR_MSE, PLS_MSE) %>%
  pivot_longer(cols = c(AR_MSE, PCR_MSE, PLS_MSE),
               names_to = "Model",
               values_to = "MSE") %>%
  mutate(Model = gsub("_MSE", "", Model),
         Method = paste(Strategy, Type, sep = "_"))

# Plot 1: MSE by Strategy and Type
p1 <- ggplot(mse_data, aes(x = factor(Horizon), y = MSE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_grid(Strategy ~ Type) +
  theme_minimal() +
  labs(title = "MSE Comparison Across All Methods",
       x = "Forecast Horizon",
       y = "Mean Squared Error") +
  theme(legend.position = "bottom")
print(p1)

# Plot 2: MSE ratio (PCR/AR and PLS/AR)
ratio_data <- all_metrics %>%
  mutate(PCR_ratio = PCR_MSE / AR_MSE,
         PLS_ratio = PLS_MSE / AR_MSE,
         Method = paste(Strategy, Type, sep = "_")) %>%
  dplyr::select(Method, Horizon, PCR_ratio, PLS_ratio) %>%
  pivot_longer(cols = c(PCR_ratio, PLS_ratio),
               names_to = "Model",
               values_to = "Ratio")

p2 <- ggplot(ratio_data, aes(x = factor(Horizon), y = Ratio, 
                             color = Model, group = Model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  facet_wrap(~ Method, ncol = 3) +
  theme_minimal() +
  labs(title = "MSE Ratio vs AR Baseline (< 1 = better than AR)",
       x = "Forecast Horizon",
       y = "MSE Ratio (Model / AR)") +
  theme(legend.position = "bottom")
print(p2)

# Plot 3: Forecast plots for each strategy (h=1 only for clarity)
plot_list <- list()
idx <- 1

for (strategy in strategies) {
  for (ftype in forecast_types) {
    key <- paste0(strategy, "_", ftype, "_h1")
    if (!key %in% names(all_forecasts)) next
    
    forecasts <- all_forecasts[[key]]
    dates <- test_data$sasdate[1:length(forecasts$actual)]
    
    plot_data <- data.frame(
      Date = dates,
      Actual = forecasts$actual,
      AR = forecasts$ar,
      PCR = forecasts$pcr,
      PLS = forecasts$pls
    ) %>%
      pivot_longer(cols = c(Actual, AR, PCR, PLS),
                   names_to = "Series",
                   values_to = "Value")
    
    p <- ggplot(plot_data, aes(x = Date, y = Value, color = Series)) +
      geom_line(aes(linetype = Series), linewidth = 0.6) +
      scale_linetype_manual(values = c("Actual" = "solid", "AR" = "dashed",
                                       "PCR" = "dotted", "PLS" = "dotdash")) +
      theme_minimal() +
      labs(title = paste(strategy, ftype, "h=1"),
           x = "Date", y = "INDPRO Growth (%)") +
      theme(legend.position = "bottom")
    
    plot_list[[idx]] <- p
    idx <- idx + 1
  }
}

do.call(grid.arrange, c(plot_list, ncol = 2))

# =============================
# SUMMARY TABLES
# =============================

cat("\n=============================\n")
cat("SUMMARY RESULTS\n")
cat("=============================\n")

cat("\n--- MSE COMPARISON ---\n")
print(all_metrics)

cat("\n--- THEIL'S U STATISTIC ---\n")
print(theil_results)

cat("\n--- DIEBOLD-MARIANO TEST ---\n")
print(dm_results)

# Find best performing method for each horizon
cat("\n--- BEST METHODS BY HORIZON ---\n")
best_methods <- all_metrics %>%
  group_by(Horizon) %>%
  summarise(
    Best_AR = paste(Strategy[which.min(AR_MSE)], Type[which.min(AR_MSE)]),
    Best_PCR = paste(Strategy[which.min(PCR_MSE)], Type[which.min(PCR_MSE)]),
    Best_PLS = paste(Strategy[which.min(PLS_MSE)], Type[which.min(PLS_MSE)]),
    Min_MSE_Overall = min(AR_MSE, PCR_MSE, PLS_MSE),
    Best_Model = c("AR", "PCR", "PLS")[which.min(c(min(AR_MSE), min(PCR_MSE), min(PLS_MSE)))]
  )
print(best_methods)





# =============================
# TIME SERIES DIAGNOSTICS & ADVANCED MODELING
# =============================

library(tseries)
library(forecast)
library(urca)
library(strucchange)
library(vars)
library(ggplot2)
library(gridExtra)
library(dplyr)
library(tidyr)

# Helper functions
mse  <- function(actual, pred) mean((actual - pred)^2, na.rm = TRUE)
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2, na.rm = TRUE))

# =============================
# PART 1: STATIONARITY TESTS
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
# PART 2: DECOMPOSITION
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
# PART 3: STRUCTURAL BREAK TESTS
# =============================

cat("\n=============================\n")
cat("STRUCTURAL BREAK TESTS\n")
cat("=============================\n\n")

# 1. Chow Test (if you have a specific break date in mind)
# Using middle of the series as hypothetical break
break_point <- floor(length(y_train) / 2)
cat("--- Chow Test (mid-point break) ---\n")
cat("Testing for break at observation:", break_point, "\n")

# Fit models before and after break
model_full <- lm(y_train ~ seq_along(y_train))
model_before <- lm(y_train[1:break_point] ~ seq_along(y_train[1:break_point]))
model_after <- lm(y_train[(break_point+1):length(y_train)] ~ 
                    seq_along(y_train[(break_point+1):length(y_train)]))

# Chow statistic
rss_full <- sum(residuals(model_full)^2)
rss_split <- sum(residuals(model_before)^2) + sum(residuals(model_after)^2)
n <- length(y_train)
k <- 2  # number of parameters
chow_stat <- ((rss_full - rss_split) / k) / (rss_split / (n - 2*k))
chow_pval <- 1 - pf(chow_stat, k, n - 2*k)

cat("Chow Statistic:", chow_stat, "\n")
cat("p-value:", chow_pval, "\n")
cat("Conclusion:", ifelse(chow_pval < 0.05,
                          "Structural break detected",
                          "No significant break"), "\n\n")

# 2. Bai-Perron Test (multiple breaks)
cat("--- Bai-Perron Multiple Breakpoint Test ---\n")
bp_test <- breakpoints(y_train ~ 1, h = 0.15)  # minimum segment size = 15%
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

# 3. CUSUM Test
cat("\n--- CUSUM Test ---\n")
ocus <- efp(y_train ~ 1, type = "OLS-CUSUM")
cusum_test <- sctest(ocus)
cat("CUSUM Test p-value:", cusum_test$p.value, "\n")
cat("Conclusion:", ifelse(cusum_test$p.value < 0.05,
                          "Parameter instability detected",
                          "Parameters appear stable"), "\n")
plot(ocus, main = "CUSUM Test")

# =============================
# PART 4: ARIMA MODELING
# =============================

cat("\n=============================\n")
cat("ARIMA MODELING\n")
cat("=============================\n\n")

# Auto ARIMA (finds optimal p, d, q)
cat("--- Auto ARIMA Model Selection ---\n")
auto_arima_model <- auto.arima(ts_train, 
                               seasonal = TRUE,
                               stepwise = TRUE,
                               approximation = FALSE,
                               trace = TRUE)

cat("\n--- Selected ARIMA Model ---\n")
print(summary(auto_arima_model))

# Manual ARIMA based on ACF/PACF
cat("\n--- ACF and PACF Analysis ---\n")
par(mfrow = c(2, 1))
acf(y_train, main = "ACF of INDPRO Growth")
pacf(y_train, main = "PACF of INDPRO Growth")
par(mfrow = c(1, 1))

# Extract ARIMA order
arima_order <- arimaorder(auto_arima_model)
cat("\nARIMA Order (p,d,q):", arima_order[1:3], "\n")
cat("Seasonal Order (P,D,Q)[m]:", arima_order[4:6], "[", arima_order[7], "]\n")

# Residual diagnostics
cat("\n--- ARIMA Residual Diagnostics ---\n")
checkresiduals(auto_arima_model)

# Ljung-Box test for residuals
lb_test <- Box.test(residuals(auto_arima_model), lag = 20, type = "Ljung-Box")
cat("Ljung-Box Test p-value:", lb_test$p.value, "\n")
cat("Interpretation:", ifelse(lb_test$p.value > 0.05,
                              "Residuals appear to be white noise (good)",
                              "Residuals show autocorrelation (may need improvement)"), "\n")

# =============================
# PART 5: VAR MODELING
# =============================

cat("\n=============================\n")
cat("VAR MODELING\n")
cat("=============================\n\n")

# Prepare multivariate data (use first few principal components + target)
# Select top predictors based on correlation
correlations <- cor(X_train_scaled, y_train)
top_vars <- order(abs(correlations), decreasing = TRUE)[1:5]
cat("Top 5 correlated predictors selected for VAR:\n")
print(colnames(X_train_scaled)[top_vars])

# Create multivariate time series
var_data <- cbind(y_train, X_train_scaled[, top_vars])
colnames(var_data)[1] <- "INDPRO_growth"

# Select VAR lag order
cat("\n--- VAR Lag Order Selection ---\n")
var_select <- VARselect(var_data, lag.max = 12, type = "const")
print(var_select$selection)

optimal_var_lag <- var_select$selection["AIC(n)"]
cat("\nOptimal VAR lag (by AIC):", optimal_var_lag, "\n")

# Fit VAR model
cat("\n--- Fitting VAR Model ---\n")
var_model <- VAR(var_data, p = optimal_var_lag, type = "const")
print(summary(var_model))

# VAR diagnostics
cat("\n--- VAR Residual Diagnostics ---\n")
serial_test <- serial.test(var_model, lags.pt = 16, type = "PT.asymptotic")
cat("Portmanteau Test p-value:", serial_test$serial$p.value, "\n")
cat("Interpretation:", ifelse(serial_test$serial$p.value > 0.05,
                              "No serial correlation in residuals",
                              "Serial correlation detected"), "\n")

# =============================
# PART 6: FORECAST COMPARISON
# =============================

cat("\n=============================\n")
cat("FORECAST COMPARISON\n")
cat("=============================\n\n")

horizons <- c(1, 3, 6, 12)
forecast_results <- data.frame()

# Baseline AR model (from earlier)
ar_model <- ar(y_train, method = "mle", aic = TRUE)
cat("Baseline AR order:", ar_model$order, "\n\n")

for (h in horizons) {
  cat("--- Horizon", h, "---\n")
  
  n_available <- length(y_test) - h + 1
  if (n_available <= 0) next
  
  y_test_h <- y_test[h:length(y_test)]
  
  # Initialize forecast vectors
  ar_forecasts <- numeric(n_available)
  arima_forecasts <- numeric(n_available)
  var_forecasts <- numeric(n_available)
  
  # Rolling window forecasts #to try recursive also?
  for (i in 1:n_available) {
    # Expanding window
    if (i == 1) {
      y_window <- y_train
      X_window <- X_train_scaled
    } else {
      n_obs <- i - 1
      y_window <- c(y_train, y_test[1:n_obs])
      X_window <- rbind(X_train_scaled, X_test_scaled[1:n_obs, , drop = FALSE])
    }
    
    # AR forecast
    ar_temp <- ar(y_window, method = "mle", order.max = ar_model$order, aic = FALSE)
    ar_pred <- predict(ar_temp, n.ahead = h)
    ar_forecasts[i] <- ar_pred$pred[h]
    
    # ARIMA forecast
    ts_window <- ts(y_window, frequency = 12)
    arima_temp <- tryCatch({
      Arima(ts_window, 
            order = arima_order[1:3],
            seasonal = list(order = arima_order[4:6], period = arima_order[7]))
    }, error = function(e) {
      auto.arima(ts_window, seasonal = TRUE)
    })
    arima_pred <- forecast(arima_temp, h = h)
    arima_forecasts[i] <- arima_pred$mean[h]
    
    # VAR forecast
    var_window <- cbind(y_window, X_window[, top_vars])
    colnames(var_window)[1] <- "INDPRO_growth"
    
    var_temp <- tryCatch({
      VAR(var_window, p = optimal_var_lag, type = "const")
    }, error = function(e) {
      VAR(var_window, p = 1, type = "const")
    })
    
    # VAR prediction needs new X values
    if (i + h - 1 <= nrow(X_test_scaled)) {
      var_pred <- predict(var_temp, n.ahead = h)
      var_forecasts[i] <- var_pred$fcst$INDPRO_growth[h, "fcst"]
    } else {
      var_forecasts[i] <- mean(y_window)  # Fallback
    }
  }
  
  # Calculate metrics
  ar_mse <- mse(y_test_h, ar_forecasts)
  arima_mse <- mse(y_test_h, arima_forecasts)
  var_mse <- mse(y_test_h, var_forecasts)
  
  cat(sprintf("MSE - AR: %.4f, ARIMA: %.4f, VAR: %.4f\n", 
              ar_mse, arima_mse, var_mse))
  
  # Store results
  forecast_results <- rbind(forecast_results, data.frame(
    Horizon = h,
    AR_MSE = ar_mse,
    ARIMA_MSE = arima_mse,
    VAR_MSE = var_mse,
    AR_RMSE = sqrt(ar_mse),
    ARIMA_RMSE = sqrt(arima_mse),
    VAR_RMSE = sqrt(var_mse)
  ))
  
  # Diebold-Mariano tests
  dm_arima <- dm.test(ar_forecasts - y_test_h,
                      arima_forecasts - y_test_h,
                      alternative = "two.sided", h = h)
  dm_var <- dm.test(ar_forecasts - y_test_h,
                    var_forecasts - y_test_h,
                    alternative = "two.sided", h = h)
  
  cat(sprintf("DM Test - ARIMA vs AR: stat=%.3f, p=%.4f %s\n",
              dm_arima$statistic, dm_arima$p.value,
              ifelse(dm_arima$p.value < 0.05, "***", "")))
  cat(sprintf("DM Test - VAR vs AR: stat=%.3f, p=%.4f %s\n\n",
              dm_var$statistic, dm_var$p.value,
              ifelse(dm_var$p.value < 0.05, "***", "")))
}

# =============================
# PART 7: SUMMARY VISUALIZATION
# =============================

cat("\n=============================\n")
cat("SUMMARY RESULTS\n")
cat("=============================\n\n")

print(forecast_results)

# MSE comparison plot
mse_plot_data <- forecast_results %>%
  dplyr::select(Horizon, AR_MSE, ARIMA_MSE, VAR_MSE) %>%
  pivot_longer(cols = -Horizon, names_to = "Model", values_to = "MSE") %>%
  mutate(Model = gsub("_MSE", "", Model))

p_mse <- ggplot(mse_plot_data, aes(x = factor(Horizon), y = MSE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "MSE Comparison: AR vs ARIMA vs VAR",
       x = "Forecast Horizon",
       y = "Mean Squared Error") +
  theme(legend.position = "bottom")

print(p_mse)

# Relative performance
forecast_results$ARIMA_vs_AR <- forecast_results$ARIMA_MSE / forecast_results$AR_MSE
forecast_results$VAR_vs_AR <- forecast_results$VAR_MSE / forecast_results$AR_MSE

cat("\n--- Relative Performance (ratio < 1 = better than AR) ---\n")
print(forecast_results[, c("Horizon", "ARIMA_vs_AR", "VAR_vs_AR")])
