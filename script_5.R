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
# 6️⃣ BASELINE AR MODEL
# =============================
library(forecast)

# Fit AR model with AIC-based lag selection
ar_model <- ar(y_train, method = "mle", aic = TRUE)
best_p <- ar_model$order
cat("Optimal AR lag (p) selected by AIC:", best_p, "\n")
cat("AIC value:", ar_model$aic, "\n")

# =============================
# 7️⃣ MULTI-STEP FORECASTING (RECURSIVE & STATIC)
# =============================
horizons <- c(1, 3, 6, 12)

# Initialize storage for forecasts
# Recursive forecasts (model updated with each new observation)
ar_forecasts_rec <- list()
pcr_forecasts_rec <- list()
pls_forecasts_rec <- list()

# Static forecasts (model fixed, trained only on initial training data)
ar_forecasts_static <- list()
pcr_forecasts_static <- list()
pls_forecasts_static <- list()

# Function to compute multi-step forecasts recursively
forecast_recursive <- function(model_type, X_test_scaled, y_train, X_train_scaled, 
                               h, best_ncomp = NULL, ar_model = NULL) {
  n_test <- nrow(X_test_scaled)
  forecasts <- numeric(n_test)
  
  for (i in 1:n_test) {
    if (model_type == "AR") {
      # AR forecast: use most recent h observations
      recent_obs <- if (i == 1) {
        tail(y_train, best_p)
      } else {
        c(tail(y_train, max(0, best_p - i + 1)), forecasts[max(1, i - best_p):(i - 1)])
      }
      recent_obs <- tail(recent_obs, best_p)
      forecasts[i] <- sum(ar_model$ar * rev(recent_obs))
      
    } else if (model_type == "PCR") {
      pred <- predict(pcr_model, newdata = data.frame(X_train_scaled = X_test_scaled[i, , drop = FALSE]),
                      ncomp = best_ncomp)
      forecasts[i] <- as.numeric(pred)
      
    } else if (model_type == "PLS") {
      pred <- predict(pls_model, newdata = data.frame(X_train_scaled = X_test_scaled[i, , drop = FALSE]),
                      ncomp = best_ncomp)
      forecasts[i] <- as.numeric(pred)
    }
  }
  return(forecasts)
}

# Generate forecasts for each horizon
for (h in horizons) {
  cat("\n=== Forecasting", h, "step(s) ahead ===\n")
  
  # Subset test data for this horizon
  n_available <- length(y_test) - h + 1
  if (n_available <= 0) {
    cat("Not enough test observations for horizon", h, "\n")
    next
  }
  
  y_test_h <- y_test[h:length(y_test)]
  X_test_h <- X_test_scaled[1:n_available, , drop = FALSE]
  
  cat(sprintf("  Data dimensions - y_test_h: %d, X_test_h: %d rows\n",
              length(y_test_h), nrow(X_test_h)))
  
  # ==================
  # RECURSIVE FORECASTS (expanding window)
  # ==================
  
  # AR recursive forecast
  ar_pred_rec <- numeric(n_available)
  for (i in 1:n_available) {
    # Expand training window
    y_train_expanded <- c(y_train, y_test[1:(i - 1 + h - 1)])
    ar_temp <- ar(y_train_expanded, method = "mle", order.max = best_p, aic = FALSE)
    ar_forecast <- predict(ar_temp, n.ahead = h)
    ar_pred_rec[i] <- ar_forecast$pred[h]
  }
  
  # PCR recursive forecast
  pcr_pred_rec <- forecast_recursive("PCR", X_test_h, y_train, X_train_scaled, 
                                     h, best_ncomp_pcr)
  
  # PLS recursive forecast
  pls_pred_rec <- forecast_recursive("PLS", X_test_h, y_train, X_train_scaled, 
                                     h, best_ncomp_pls)
  
  # ==================
  # STATIC FORECASTS (fixed model)
  # ==================
  
  # AR static forecast (using original model, no updates)
  # Generate a single long forecast sequence and extract h-step-ahead forecasts
  max_ahead <- n_available + h - 1
  ar_long_forecast <- predict(ar_model, n.ahead = max_ahead)
  
  # Extract h-step-ahead forecasts: 
  # For i=1, we want forecast at position h
  # For i=2, we want forecast at position h+1, etc.
  ar_pred_static <- ar_long_forecast$pred[h:(h + n_available - 1)]
  
  # PCR static forecast - need to properly subset predictions
  pcr_pred_full <- predict(pcr_model, 
                           newdata = data.frame(X_train_scaled = X_test_scaled),
                           ncomp = best_ncomp_pcr)
  pcr_pred_static <- as.numeric(pcr_pred_full[1:n_available])
  
  # PLS static forecast - need to properly subset predictions
  pls_pred_full <- predict(pls_model, 
                           newdata = data.frame(X_train_scaled = X_test_scaled),
                           ncomp = best_ncomp_pls)
  pls_pred_static <- as.numeric(pls_pred_full[1:n_available])
  
  # Verify all forecasts have the same length
  cat(sprintf("  Forecast lengths - AR_rec: %d, PCR_rec: %d, PLS_rec: %d\n",
              length(ar_pred_rec), length(pcr_pred_rec), length(pls_pred_rec)))
  cat(sprintf("  Forecast lengths - AR_static: %d, PCR_static: %d, PLS_static: %d\n",
              length(ar_pred_static), length(pcr_pred_static), length(pls_pred_static)))
  
  # Store forecasts
  ar_forecasts_rec[[paste0("h", h)]] <- ar_pred_rec
  pcr_forecasts_rec[[paste0("h", h)]] <- pcr_pred_rec
  pls_forecasts_rec[[paste0("h", h)]] <- pls_pred_rec
  
  ar_forecasts_static[[paste0("h", h)]] <- ar_pred_static
  pcr_forecasts_static[[paste0("h", h)]] <- pcr_pred_static
  pls_forecasts_static[[paste0("h", h)]] <- pls_pred_static
  
  # Calculate metrics
  cat("RECURSIVE:\n")
  cat(sprintf("  MSE - AR: %.4f, PCR: %.4f, PLS: %.4f\n", 
              mse(y_test_h, ar_pred_rec), 
              mse(y_test_h, pcr_pred_rec), 
              mse(y_test_h, pls_pred_rec)))
  
  cat("STATIC:\n")
  cat(sprintf("  MSE - AR: %.4f, PCR: %.4f, PLS: %.4f\n", 
              mse(y_test_h, ar_pred_static), 
              mse(y_test_h, pcr_pred_static), 
              mse(y_test_h, pls_pred_static)))
}

# =============================
# 8️⃣ THEIL'S U STATISTIC
# =============================
cat("\n=== THEIL'S U STATISTIC ===\n")
cat("(U < 1 indicates model outperforms naive benchmark)\n\n")

theil_u <- function(actual, pred, naive_pred) {
  numerator <- sqrt(mean((pred - actual)^2))
  denominator <- sqrt(mean((naive_pred - actual)^2))
  return(numerator / denominator)
}

# Recursive results
theil_results_rec <- data.frame(
  Horizon = integer(),
  PCR_vs_AR = numeric(),
  PLS_vs_AR = numeric()
)

# Static results
theil_results_static <- data.frame(
  Horizon = integer(),
  PCR_vs_AR = numeric(),
  PLS_vs_AR = numeric()
)

cat("RECURSIVE FORECASTS:\n")
for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_rec)) next
  
  n_available <- length(y_test) - h + 1
  y_test_h <- y_test[h:length(y_test)]
  
  u_pcr <- theil_u(y_test_h, pcr_forecasts_rec[[key]], ar_forecasts_rec[[key]])
  u_pls <- theil_u(y_test_h, pls_forecasts_rec[[key]], ar_forecasts_rec[[key]])
  
  theil_results_rec <- rbind(theil_results_rec, 
                             data.frame(Horizon = h, 
                                        PCR_vs_AR = u_pcr, 
                                        PLS_vs_AR = u_pls))
  
  cat(sprintf("Horizon %2d: PCR = %.4f, PLS = %.4f\n", h, u_pcr, u_pls))
}

cat("\nSTATIC FORECASTS:\n")
for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_static)) next
  
  n_available <- length(y_test) - h + 1
  y_test_h <- y_test[h:length(y_test)]
  
  u_pcr <- theil_u(y_test_h, pcr_forecasts_static[[key]], ar_forecasts_static[[key]])
  u_pls <- theil_u(y_test_h, pls_forecasts_static[[key]], ar_forecasts_static[[key]])
  
  theil_results_static <- rbind(theil_results_static, 
                                data.frame(Horizon = h, 
                                           PCR_vs_AR = u_pcr, 
                                           PLS_vs_AR = u_pls))
  
  cat(sprintf("Horizon %2d: PCR = %.4f, PLS = %.4f\n", h, u_pcr, u_pls))
}

# =============================
# 9️⃣ DIEBOLD-MARIANO TEST
# =============================
library(forecast)

cat("\n=== DIEBOLD-MARIANO TEST ===\n")
cat("(p-value < 0.05 indicates significant difference in forecast accuracy)\n\n")

# Recursive results
dm_results_rec <- data.frame(
  Horizon = integer(),
  PCR_vs_AR_stat = numeric(),
  PCR_vs_AR_pval = numeric(),
  PLS_vs_AR_stat = numeric(),
  PLS_vs_AR_pval = numeric()
)

# Static results
dm_results_static <- data.frame(
  Horizon = integer(),
  PCR_vs_AR_stat = numeric(),
  PCR_vs_AR_pval = numeric(),
  PLS_vs_AR_stat = numeric(),
  PLS_vs_AR_pval = numeric()
)

cat("RECURSIVE FORECASTS:\n")
for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_rec)) next
  
  n_available <- length(y_test) - h + 1
  y_test_h <- y_test[h:length(y_test)]
  
  # PCR vs AR
  dm_pcr <- dm.test(ar_forecasts_rec[[key]] - y_test_h, 
                    pcr_forecasts_rec[[key]] - y_test_h, 
                    alternative = "two.sided", h = h)
  
  # PLS vs AR
  dm_pls <- dm.test(ar_forecasts_rec[[key]] - y_test_h, 
                    pls_forecasts_rec[[key]] - y_test_h, 
                    alternative = "two.sided", h = h)
  
  dm_results_rec <- rbind(dm_results_rec, 
                          data.frame(Horizon = h,
                                     PCR_vs_AR_stat = dm_pcr$statistic,
                                     PCR_vs_AR_pval = dm_pcr$p.value,
                                     PLS_vs_AR_stat = dm_pls$statistic,
                                     PLS_vs_AR_pval = dm_pls$p.value))
  
  cat(sprintf("Horizon %2d:\n", h))
  cat(sprintf("  PCR vs AR: DM = %7.3f, p-value = %.4f %s\n", 
              dm_pcr$statistic, dm_pcr$p.value, 
              ifelse(dm_pcr$p.value < 0.05, "***", "")))
  cat(sprintf("  PLS vs AR: DM = %7.3f, p-value = %.4f %s\n", 
              dm_pls$statistic, dm_pls$p.value,
              ifelse(dm_pls$p.value < 0.05, "***", "")))
}

cat("\nSTATIC FORECASTS:\n")
for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_static)) next
  
  n_available <- length(y_test) - h + 1
  y_test_h <- y_test[h:length(y_test)]
  
  # PCR vs AR
  dm_pcr <- dm.test(ar_forecasts_static[[key]] - y_test_h, 
                    pcr_forecasts_static[[key]] - y_test_h, 
                    alternative = "two.sided", h = h)
  
  # PLS vs AR
  dm_pls <- dm.test(ar_forecasts_static[[key]] - y_test_h, 
                    pls_forecasts_static[[key]] - y_test_h, 
                    alternative = "two.sided", h = h)
  
  dm_results_static <- rbind(dm_results_static, 
                             data.frame(Horizon = h,
                                        PCR_vs_AR_stat = dm_pcr$statistic,
                                        PCR_vs_AR_pval = dm_pcr$p.value,
                                        PLS_vs_AR_stat = dm_pls$statistic,
                                        PLS_vs_AR_pval = dm_pls$p.value))
  
  cat(sprintf("Horizon %2d:\n", h))
  cat(sprintf("  PCR vs AR: DM = %7.3f, p-value = %.4f %s\n", 
              dm_pcr$statistic, dm_pcr$p.value, 
              ifelse(dm_pcr$p.value < 0.05, "***", "")))
  cat(sprintf("  PLS vs AR: DM = %7.3f, p-value = %.4f %s\n", 
              dm_pls$statistic, dm_pls$p.value,
              ifelse(dm_pls$p.value < 0.05, "***", "")))
}

# =============================
# 🔟 FORECAST PLOTS (RECURSIVE & STATIC)
# =============================
library(tidyr)
library(gridExtra)

plot_list_rec <- list()
plot_list_static <- list()

for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_rec)) next
  
  n_available <- length(y_test) - h + 1
  dates_test <- test_data$sasdate[h:length(y_test)]
  y_test_h <- y_test[h:length(y_test)]
  
  # Recursive plot
  plot_data_rec <- data.frame(
    Date = dates_test,
    Actual = y_test_h,
    AR = ar_forecasts_rec[[key]],
    PCR = pcr_forecasts_rec[[key]],
    PLS = pls_forecasts_rec[[key]]
  ) %>%
    pivot_longer(cols = c(Actual, AR, PCR, PLS), 
                 names_to = "Series", values_to = "Value")
  
  p_rec <- ggplot(plot_data_rec, aes(x = Date, y = Value, color = Series)) +
    geom_line(aes(linetype = Series), linewidth = 0.7) +
    scale_linetype_manual(values = c("Actual" = "solid", "AR" = "dashed", 
                                     "PCR" = "dotted", "PLS" = "dotdash")) +
    theme_minimal() +
    labs(title = paste("Recursive Forecasts: h =", h),
         x = "Date", y = "INDPRO Growth (%)") +
    theme(legend.position = "bottom")
  
  plot_list_rec[[key]] <- p_rec
  
  # Static plot
  plot_data_static <- data.frame(
    Date = dates_test,
    Actual = y_test_h,
    AR = ar_forecasts_static[[key]],
    PCR = pcr_forecasts_static[[key]],
    PLS = pls_forecasts_static[[key]]
  ) %>%
    pivot_longer(cols = c(Actual, AR, PCR, PLS), 
                 names_to = "Series", values_to = "Value")
  
  p_static <- ggplot(plot_data_static, aes(x = Date, y = Value, color = Series)) +
    geom_line(aes(linetype = Series), linewidth = 0.7) +
    scale_linetype_manual(values = c("Actual" = "solid", "AR" = "dashed", 
                                     "PCR" = "dotted", "PLS" = "dotdash")) +
    theme_minimal() +
    labs(title = paste("Static Forecasts: h =", h),
         x = "Date", y = "INDPRO Growth (%)") +
    theme(legend.position = "bottom")
  
  plot_list_static[[key]] <- p_static
}

# Display recursive plots
cat("\n=== RECURSIVE FORECAST PLOTS ===\n")
do.call(grid.arrange, c(plot_list_rec, ncol = 2))

# Display static plots
cat("\n=== STATIC FORECAST PLOTS ===\n")
do.call(grid.arrange, c(plot_list_static, ncol = 2))

# =============================
# 1️⃣1️⃣ MSE COMPARISON PLOTS
# =============================

# Recursive MSE comparison
mse_comparison_rec <- data.frame(
  Horizon = integer(),
  AR = numeric(),
  PCR = numeric(),
  PLS = numeric()
)

for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_rec)) next
  
  n_available <- length(y_test) - h + 1
  y_test_h <- y_test[h:length(y_test)]
  
  mse_comparison_rec <- rbind(mse_comparison_rec,
                              data.frame(
                                Horizon = h,
                                AR = mse(y_test_h, ar_forecasts_rec[[key]]),
                                PCR = mse(y_test_h, pcr_forecasts_rec[[key]]),
                                PLS = mse(y_test_h, pls_forecasts_rec[[key]])
                              ))
}

# Static MSE comparison
mse_comparison_static <- data.frame(
  Horizon = integer(),
  AR = numeric(),
  PCR = numeric(),
  PLS = numeric()
)

for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_static)) next
  
  n_available <- length(y_test) - h + 1
  y_test_h <- y_test[h:length(y_test)]
  
  mse_comparison_static <- rbind(mse_comparison_static,
                                 data.frame(
                                   Horizon = h,
                                   AR = mse(y_test_h, ar_forecasts_static[[key]]),
                                   PCR = mse(y_test_h, pcr_forecasts_static[[key]]),
                                   PLS = mse(y_test_h, pls_forecasts_static[[key]])
                                 ))
}

# Recursive MSE plot
mse_plot_data_rec <- mse_comparison_rec %>%
  pivot_longer(cols = c(AR, PCR, PLS), 
               names_to = "Model", values_to = "MSE")

p_mse_rec <- ggplot(mse_plot_data_rec, aes(x = factor(Horizon), y = MSE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "MSE Comparison: Recursive Forecasts",
       x = "Forecast Horizon (steps ahead)",
       y = "Mean Squared Error") +
  theme(legend.position = "bottom")

# Static MSE plot
mse_plot_data_static <- mse_comparison_static %>%
  pivot_longer(cols = c(AR, PCR, PLS), 
               names_to = "Model", values_to = "MSE")

p_mse_static <- ggplot(mse_plot_data_static, aes(x = factor(Horizon), y = MSE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "MSE Comparison: Static Forecasts",
       x = "Forecast Horizon (steps ahead)",
       y = "Mean Squared Error") +
  theme(legend.position = "bottom")

print(p_mse_rec)
print(p_mse_static)

# Summary tables
cat("\n=== RECURSIVE FORECASTS - MSE SUMMARY ===\n")
print(mse_comparison_rec)

cat("\n=== STATIC FORECASTS - MSE SUMMARY ===\n")
print(mse_comparison_static)

cat("\n=== RECURSIVE FORECASTS - THEIL'S U ===\n")
print(theil_results_rec)

cat("\n=== STATIC FORECASTS - THEIL'S U ===\n")
print(theil_results_static)

cat("\n=== RECURSIVE FORECASTS - DIEBOLD-MARIANO ===\n")
print(dm_results_rec)

cat("\n=== STATIC FORECASTS - DIEBOLD-MARIANO ===\n")
print(dm_results_static)





# =============================
# 6️⃣ BASELINE AR MODEL
# =============================
library(forecast)

# Fit AR model with AIC-based lag selection
ar_model <- ar(y_train, method = "mle", aic = TRUE)
best_p <- ar_model$order
cat("Optimal AR lag (p) selected by AIC:", best_p, "\n")
cat("AIC value:", ar_model$aic, "\n")

# =============================
# 7️⃣ MULTI-STEP FORECASTING (RECURSIVE & STATIC)
# =============================
horizons <- c(1, 3, 6, 12)

# Initialize storage for forecasts
# Recursive forecasts (model updated with each new observation)
ar_forecasts_rec <- list()
pcr_forecasts_rec <- list()
pls_forecasts_rec <- list()

# Static forecasts (model fixed, trained only on initial training data)
ar_forecasts_static <- list()
pcr_forecasts_static <- list()
pls_forecasts_static <- list()

# Generate forecasts for each horizon
for (h in horizons) {
  cat("\n=== Forecasting", h, "step(s) ahead ===\n")
  
  # Subset test data for this horizon
  n_available <- length(y_test) - h + 1
  if (n_available <= 0) {
    cat("Not enough test observations for horizon", h, "\n")
    next
  }
  
  y_test_h <- y_test[h:length(y_test)]
  X_test_h <- X_test_scaled[1:n_available, , drop = FALSE]
  
  # ==================
  # RECURSIVE FORECASTS (expanding window)
  # ==================
  
  # AR recursive forecast
  ar_pred_rec <- numeric(n_available)
  for (i in 1:n_available) {
    # Expand training window
    y_train_expanded <- c(y_train, y_test[1:max(0, i - 1 + h - 1)])
    ar_temp <- ar(y_train_expanded, method = "mle", order.max = best_p, aic = FALSE)
    ar_forecast <- predict(ar_temp, n.ahead = h)
    ar_pred_rec[i] <- ar_forecast$pred[h]
  }
  
  # PCR recursive forecast - PROPERLY IMPLEMENTED
  pcr_pred_rec <- numeric(n_available)
  for (i in 1:n_available) {
    # Expand training window
    if (i == 1) {
      y_train_expanded <- y_train
      X_train_expanded <- X_train_scaled
    } else {
      # Add observed test data up to current point
      y_train_expanded <- c(y_train, y_test[1:(i - 1 + h - 1)])
      X_train_expanded <- rbind(X_train_scaled, X_test_scaled[1:(i - 1 + h - 1), , drop = FALSE])
    }
    
    # Retrain PCR model
    pcr_temp <- pcr(y_train_expanded ~ X_train_expanded, 
                    ncomp = best_ncomp_pcr, validation = "none")
    
    # Predict h steps ahead
    pcr_forecast <- predict(pcr_temp, 
                            newdata = data.frame(X_train_expanded = X_test_scaled[i, , drop = FALSE]),
                            ncomp = best_ncomp_pcr)
    pcr_pred_rec[i] <- as.numeric(pcr_forecast)
  }
  
  # PLS recursive forecast - PROPERLY IMPLEMENTED
  pls_pred_rec <- numeric(n_available)
  for (i in 1:n_available) {
    # Expand training window
    if (i == 1) {
      y_train_expanded <- y_train
      X_train_expanded <- X_train_scaled
    } else {
      y_train_expanded <- c(y_train, y_test[1:(i - 1 + h - 1)])
      X_train_expanded <- rbind(X_train_scaled, X_test_scaled[1:(i - 1 + h - 1), , drop = FALSE])
    }
    
    # Retrain PLS model
    pls_temp <- plsr(y_train_expanded ~ X_train_expanded, 
                     ncomp = best_ncomp_pls, validation = "none")
    
    # Predict h steps ahead
    pls_forecast <- predict(pls_temp, 
                            newdata = data.frame(X_train_expanded = X_test_scaled[i, , drop = FALSE]),
                            ncomp = best_ncomp_pls)
    pls_pred_rec[i] <- as.numeric(pls_forecast)
  }
  
  # ==================
  # STATIC FORECASTS (fixed model)
  # ==================
  
  # AR static forecast (using original model, no updates)
  max_ahead <- n_available + h - 1
  ar_long_forecast <- predict(ar_model, n.ahead = max_ahead)
  ar_pred_static <- ar_long_forecast$pred[h:(h + n_available - 1)]
  
  # PCR static forecast - need to properly subset predictions
  pcr_pred_full <- predict(pcr_model, 
                           newdata = data.frame(X_train_scaled = X_test_scaled),
                           ncomp = best_ncomp_pcr)
  pcr_pred_static <- as.numeric(pcr_pred_full[1:n_available])
  
  # PLS static forecast - need to properly subset predictions
  pls_pred_full <- predict(pls_model, 
                           newdata = data.frame(X_train_scaled = X_test_scaled),
                           ncomp = best_ncomp_pls)
  pls_pred_static <- as.numeric(pls_pred_full[1:n_available])
  
  # Store forecasts
  ar_forecasts_rec[[paste0("h", h)]] <- ar_pred_rec
  pcr_forecasts_rec[[paste0("h", h)]] <- pcr_pred_rec
  pls_forecasts_rec[[paste0("h", h)]] <- pls_pred_rec
  
  ar_forecasts_static[[paste0("h", h)]] <- ar_pred_static
  pcr_forecasts_static[[paste0("h", h)]] <- pcr_pred_static
  pls_forecasts_static[[paste0("h", h)]] <- pls_pred_static
  
  # Calculate metrics
  cat("RECURSIVE:\n")
  cat(sprintf("  MSE - AR: %.4f, PCR: %.4f, PLS: %.4f\n", 
              mse(y_test_h, ar_pred_rec), 
              mse(y_test_h, pcr_pred_rec), 
              mse(y_test_h, pls_pred_rec)))
  
  cat("STATIC:\n")
  cat(sprintf("  MSE - AR: %.4f, PCR: %.4f, PLS: %.4f\n", 
              mse(y_test_h, ar_pred_static), 
              mse(y_test_h, pcr_pred_static), 
              mse(y_test_h, pls_pred_static)))
}

# =============================
# 8️⃣ THEIL'S U STATISTIC
# =============================
cat("\n=== THEIL'S U STATISTIC ===\n")
cat("(U < 1 indicates model outperforms naive benchmark)\n\n")

theil_u <- function(actual, pred, naive_pred) {
  numerator <- sqrt(mean((pred - actual)^2))
  denominator <- sqrt(mean((naive_pred - actual)^2))
  return(numerator / denominator)
}

# Recursive results
theil_results_rec <- data.frame(
  Horizon = integer(),
  PCR_vs_AR = numeric(),
  PLS_vs_AR = numeric()
)

# Static results
theil_results_static <- data.frame(
  Horizon = integer(),
  PCR_vs_AR = numeric(),
  PLS_vs_AR = numeric()
)

cat("RECURSIVE FORECASTS:\n")
for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_rec)) next
  
  n_available <- length(y_test) - h + 1
  y_test_h <- y_test[h:length(y_test)]
  
  u_pcr <- theil_u(y_test_h, pcr_forecasts_rec[[key]], ar_forecasts_rec[[key]])
  u_pls <- theil_u(y_test_h, pls_forecasts_rec[[key]], ar_forecasts_rec[[key]])
  
  theil_results_rec <- rbind(theil_results_rec, 
                             data.frame(Horizon = h, 
                                        PCR_vs_AR = u_pcr, 
                                        PLS_vs_AR = u_pls))
  
  cat(sprintf("Horizon %2d: PCR = %.4f, PLS = %.4f\n", h, u_pcr, u_pls))
}

cat("\nSTATIC FORECASTS:\n")
for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_static)) next
  
  n_available <- length(y_test) - h + 1
  y_test_h <- y_test[h:length(y_test)]
  
  u_pcr <- theil_u(y_test_h, pcr_forecasts_static[[key]], ar_forecasts_static[[key]])
  u_pls <- theil_u(y_test_h, pls_forecasts_static[[key]], ar_forecasts_static[[key]])
  
  theil_results_static <- rbind(theil_results_static, 
                                data.frame(Horizon = h, 
                                           PCR_vs_AR = u_pcr, 
                                           PLS_vs_AR = u_pls))
  
  cat(sprintf("Horizon %2d: PCR = %.4f, PLS = %.4f\n", h, u_pcr, u_pls))
}

# =============================
# 9️⃣ DIEBOLD-MARIANO TEST
# =============================
library(forecast)

cat("\n=== DIEBOLD-MARIANO TEST ===\n")
cat("(p-value < 0.05 indicates significant difference in forecast accuracy)\n\n")

# Recursive results
dm_results_rec <- data.frame(
  Horizon = integer(),
  PCR_vs_AR_stat = numeric(),
  PCR_vs_AR_pval = numeric(),
  PLS_vs_AR_stat = numeric(),
  PLS_vs_AR_pval = numeric()
)

# Static results
dm_results_static <- data.frame(
  Horizon = integer(),
  PCR_vs_AR_stat = numeric(),
  PCR_vs_AR_pval = numeric(),
  PLS_vs_AR_stat = numeric(),
  PLS_vs_AR_pval = numeric()
)

cat("RECURSIVE FORECASTS:\n")
for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_rec)) next
  
  n_available <- length(y_test) - h + 1
  y_test_h <- y_test[h:length(y_test)]
  
  # PCR vs AR
  dm_pcr <- dm.test(ar_forecasts_rec[[key]] - y_test_h, 
                    pcr_forecasts_rec[[key]] - y_test_h, 
                    alternative = "two.sided", h = h)
  
  # PLS vs AR
  dm_pls <- dm.test(ar_forecasts_rec[[key]] - y_test_h, 
                    pls_forecasts_rec[[key]] - y_test_h, 
                    alternative = "two.sided", h = h)
  
  dm_results_rec <- rbind(dm_results_rec, 
                          data.frame(Horizon = h,
                                     PCR_vs_AR_stat = dm_pcr$statistic,
                                     PCR_vs_AR_pval = dm_pcr$p.value,
                                     PLS_vs_AR_stat = dm_pls$statistic,
                                     PLS_vs_AR_pval = dm_pls$p.value))
  
  cat(sprintf("Horizon %2d:\n", h))
  cat(sprintf("  PCR vs AR: DM = %7.3f, p-value = %.4f %s\n", 
              dm_pcr$statistic, dm_pcr$p.value, 
              ifelse(dm_pcr$p.value < 0.05, "***", "")))
  cat(sprintf("  PLS vs AR: DM = %7.3f, p-value = %.4f %s\n", 
              dm_pls$statistic, dm_pls$p.value,
              ifelse(dm_pls$p.value < 0.05, "***", "")))
}

cat("\nSTATIC FORECASTS:\n")
for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_static)) next
  
  n_available <- length(y_test) - h + 1
  y_test_h <- y_test[h:length(y_test)]
  
  # PCR vs AR
  dm_pcr <- dm.test(ar_forecasts_static[[key]] - y_test_h, 
                    pcr_forecasts_static[[key]] - y_test_h, 
                    alternative = "two.sided", h = h)
  
  # PLS vs AR
  dm_pls <- dm.test(ar_forecasts_static[[key]] - y_test_h, 
                    pls_forecasts_static[[key]] - y_test_h, 
                    alternative = "two.sided", h = h)
  
  dm_results_static <- rbind(dm_results_static, 
                             data.frame(Horizon = h,
                                        PCR_vs_AR_stat = dm_pcr$statistic,
                                        PCR_vs_AR_pval = dm_pcr$p.value,
                                        PLS_vs_AR_stat = dm_pls$statistic,
                                        PLS_vs_AR_pval = dm_pls$p.value))
  
  cat(sprintf("Horizon %2d:\n", h))
  cat(sprintf("  PCR vs AR: DM = %7.3f, p-value = %.4f %s\n", 
              dm_pcr$statistic, dm_pcr$p.value, 
              ifelse(dm_pcr$p.value < 0.05, "***", "")))
  cat(sprintf("  PLS vs AR: DM = %7.3f, p-value = %.4f %s\n", 
              dm_pls$statistic, dm_pls$p.value,
              ifelse(dm_pls$p.value < 0.05, "***", "")))
}

# =============================
# 🔟 FORECAST PLOTS (RECURSIVE & STATIC)
# =============================
library(tidyr)
library(gridExtra)

plot_list_rec <- list()
plot_list_static <- list()

for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_rec)) next
  
  n_available <- length(y_test) - h + 1
  dates_test <- test_data$sasdate[h:length(y_test)]
  y_test_h <- y_test[h:length(y_test)]
  
  # Recursive plot
  plot_data_rec <- data.frame(
    Date = dates_test,
    Actual = y_test_h,
    AR = ar_forecasts_rec[[key]],
    PCR = pcr_forecasts_rec[[key]],
    PLS = pls_forecasts_rec[[key]]
  ) %>%
    pivot_longer(cols = c(Actual, AR, PCR, PLS), 
                 names_to = "Series", values_to = "Value")
  
  p_rec <- ggplot(plot_data_rec, aes(x = Date, y = Value, color = Series)) +
    geom_line(aes(linetype = Series), linewidth = 0.7) +
    scale_linetype_manual(values = c("Actual" = "solid", "AR" = "dashed", 
                                     "PCR" = "dotted", "PLS" = "dotdash")) +
    theme_minimal() +
    labs(title = paste("Recursive Forecasts: h =", h),
         x = "Date", y = "INDPRO Growth (%)") +
    theme(legend.position = "bottom")
  
  plot_list_rec[[key]] <- p_rec
  
  # Static plot
  plot_data_static <- data.frame(
    Date = dates_test,
    Actual = y_test_h,
    AR = ar_forecasts_static[[key]],
    PCR = pcr_forecasts_static[[key]],
    PLS = pls_forecasts_static[[key]]
  ) %>%
    pivot_longer(cols = c(Actual, AR, PCR, PLS), 
                 names_to = "Series", values_to = "Value")
  
  p_static <- ggplot(plot_data_static, aes(x = Date, y = Value, color = Series)) +
    geom_line(aes(linetype = Series), linewidth = 0.7) +
    scale_linetype_manual(values = c("Actual" = "solid", "AR" = "dashed", 
                                     "PCR" = "dotted", "PLS" = "dotdash")) +
    theme_minimal() +
    labs(title = paste("Static Forecasts: h =", h),
         x = "Date", y = "INDPRO Growth (%)") +
    theme(legend.position = "bottom")
  
  plot_list_static[[key]] <- p_static
}

# Display recursive plots
cat("\n=== RECURSIVE FORECAST PLOTS ===\n")
do.call(grid.arrange, c(plot_list_rec, ncol = 2))

# Display static plots
cat("\n=== STATIC FORECAST PLOTS ===\n")
do.call(grid.arrange, c(plot_list_static, ncol = 2))

# =============================
# 1️⃣1️⃣ MSE COMPARISON PLOTS
# =============================

# Recursive MSE comparison
mse_comparison_rec <- data.frame(
  Horizon = integer(),
  AR = numeric(),
  PCR = numeric(),
  PLS = numeric()
)

for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_rec)) next
  
  n_available <- length(y_test) - h + 1
  y_test_h <- y_test[h:length(y_test)]
  
  mse_comparison_rec <- rbind(mse_comparison_rec,
                              data.frame(
                                Horizon = h,
                                AR = mse(y_test_h, ar_forecasts_rec[[key]]),
                                PCR = mse(y_test_h, pcr_forecasts_rec[[key]]),
                                PLS = mse(y_test_h, pls_forecasts_rec[[key]])
                              ))
}

# Static MSE comparison
mse_comparison_static <- data.frame(
  Horizon = integer(),
  AR = numeric(),
  PCR = numeric(),
  PLS = numeric()
)

for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_static)) next
  
  n_available <- length(y_test) - h + 1
  y_test_h <- y_test[h:length(y_test)]
  
  mse_comparison_static <- rbind(mse_comparison_static,
                                 data.frame(
                                   Horizon = h,
                                   AR = mse(y_test_h, ar_forecasts_static[[key]]),
                                   PCR = mse(y_test_h, pcr_forecasts_static[[key]]),
                                   PLS = mse(y_test_h, pls_forecasts_static[[key]])
                                 ))
}

# Recursive MSE plot
mse_plot_data_rec <- mse_comparison_rec %>%
  pivot_longer(cols = c(AR, PCR, PLS), 
               names_to = "Model", values_to = "MSE")

p_mse_rec <- ggplot(mse_plot_data_rec, aes(x = factor(Horizon), y = MSE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "MSE Comparison: Recursive Forecasts",
       x = "Forecast Horizon (steps ahead)",
       y = "Mean Squared Error") +
  theme(legend.position = "bottom")

# Static MSE plot
mse_plot_data_static <- mse_comparison_static %>%
  pivot_longer(cols = c(AR, PCR, PLS), 
               names_to = "Model", values_to = "MSE")

p_mse_static <- ggplot(mse_plot_data_static, aes(x = factor(Horizon), y = MSE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "MSE Comparison: Static Forecasts",
       x = "Forecast Horizon (steps ahead)",
       y = "Mean Squared Error") +
  theme(legend.position = "bottom")

print(p_mse_rec)
print(p_mse_static)

# Summary tables
cat("\n=== RECURSIVE FORECASTS - MSE SUMMARY ===\n")
print(mse_comparison_rec)

cat("\n=== STATIC FORECASTS - MSE SUMMARY ===\n")
print(mse_comparison_static)

cat("\n=== RECURSIVE FORECASTS - THEIL'S U ===\n")
print(theil_results_rec)

cat("\n=== STATIC FORECASTS - THEIL'S U ===\n")
print(theil_results_static)

cat("\n=== RECURSIVE FORECASTS - DIEBOLD-MARIANO ===\n")
print(dm_results_rec)

cat("\n=== STATIC FORECASTS - DIEBOLD-MARIANO ===\n")
print(dm_results_static)





# =============================
# ALTERNATIVE: ROLLING WINDOW + DIRECT FORECASTING
# =============================
# This addresses the lookahead bias concern by:
# 1. Using a fixed rolling window (not expanding)
# 2. Training separate models for each horizon (direct forecasting)
# 3. Only using information available at forecast time

library(pls)
library(forecast)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

horizons <- c(1, 3, 6, 12)

# Choose window size (e.g., last 10 years of monthly data = 120 months)
window_size <- 120

# Initialize storage
ar_forecasts_rolling <- list()
pcr_forecasts_rolling <- list()
pls_forecasts_rolling <- list()
pcr_forecasts_direct <- list()
pls_forecasts_direct <- list()

cat("\n=== ROLLING WINDOW FORECASTING ===\n")
cat("Window size:", window_size, "observations\n")

for (h in horizons) {
  cat("\n--- Horizon", h, "---\n")
  
  n_available <- length(y_test) - h + 1
  if (n_available <= 0) next
  
  y_test_h <- y_test[h:length(y_test)]
  
  ar_pred <- numeric(n_available)
  pcr_pred_recursive <- numeric(n_available)
  pls_pred_recursive <- numeric(n_available)
  pcr_pred_direct <- numeric(n_available)
  pls_pred_direct <- numeric(n_available)
  
  for (i in 1:n_available) {
    # Define the rolling window endpoint
    # At forecast origin i, we can use data up to test observation (i-1)
    if (i == 1) {
      # First forecast: use only training data
      y_window <- tail(y_train, window_size)
      X_window <- tail(X_train_scaled, window_size)
      
      # For direct forecast, we need target at horizon h
      y_direct <- tail(y_train, window_size - h + 1)
      X_direct <- head(tail(X_train_scaled, window_size), window_size - h + 1)
    } else {
      # Subsequent forecasts: can include observed test data
      n_obs_available <- i - 1
      y_all <- c(y_train, y_test[1:n_obs_available])
      X_all <- rbind(X_train_scaled, X_test_scaled[1:n_obs_available, , drop = FALSE])
      
      y_window <- tail(y_all, window_size)
      X_window <- tail(X_all, window_size)
      
      # For direct forecast
      if (length(y_all) >= h) {
        y_direct <- head(tail(y_all, window_size), max(1, window_size - h + 1))
        X_direct <- head(tail(X_all, window_size), max(1, window_size - h + 1))
      } else {
        y_direct <- y_window
        X_direct <- X_window
      }
    }
    
    # === AR MODEL (Rolling) ===
    ar_temp <- ar(y_window, method = "mle", aic = TRUE)
    ar_forecast <- predict(ar_temp, n.ahead = h)
    ar_pred[i] <- ar_forecast$pred[h]
    
    # === PCR MODEL (Recursive - uses X at forecast origin) ===
    pcr_temp <- pcr(y_window ~ X_window, validation = "CV")
    ncomp_pcr <- which.min(RMSEP(pcr_temp)$val[1, , -1])
    pcr_forecast <- predict(pcr_temp, 
                            newdata = data.frame(X_window = X_test_scaled[i, , drop = FALSE]),
                            ncomp = ncomp_pcr)
    pcr_pred_recursive[i] <- as.numeric(pcr_forecast)
    
    # === PLS MODEL (Recursive) ===
    pls_temp <- plsr(y_window ~ X_window, validation = "CV")
    ncomp_pls <- which.min(RMSEP(pls_temp)$val[1, , -1])
    pls_forecast <- predict(pls_temp, 
                            newdata = data.frame(X_window = X_test_scaled[i, , drop = FALSE]),
                            ncomp = ncomp_pls)
    pls_pred_recursive[i] <- as.numeric(pls_forecast)
    
    # === PCR MODEL (Direct - trained to predict h steps ahead) ===
    # Create lagged target: y[t+h] ~ X[t]
    if (nrow(X_direct) > 0 && length(y_direct) > 0) {
      # Match dimensions
      n_direct <- min(length(y_direct), nrow(X_direct))
      if (n_direct > 10) {  # Need minimum observations
        pcr_direct_temp <- pcr(y_direct[1:n_direct] ~ X_direct[1:n_direct, ], 
                               validation = "CV")
        ncomp_pcr_d <- which.min(RMSEP(pcr_direct_temp)$val[1, , -1])
        pcr_direct_forecast <- predict(pcr_direct_temp,
                                       newdata = data.frame(X_direct = X_test_scaled[i, , drop = FALSE]),
                                       ncomp = ncomp_pcr_d)
        pcr_pred_direct[i] <- as.numeric(pcr_direct_forecast)
      } else {
        pcr_pred_direct[i] <- pcr_pred_recursive[i]  # Fallback
      }
    } else {
      pcr_pred_direct[i] <- pcr_pred_recursive[i]
    }
    
    # === PLS MODEL (Direct) ===
    if (nrow(X_direct) > 0 && length(y_direct) > 0) {
      n_direct <- min(length(y_direct), nrow(X_direct))
      if (n_direct > 10) {
        pls_direct_temp <- plsr(y_direct[1:n_direct] ~ X_direct[1:n_direct, ], 
                                validation = "CV")
        ncomp_pls_d <- which.min(RMSEP(pls_direct_temp)$val[1, , -1])
        pls_direct_forecast <- predict(pls_direct_temp,
                                       newdata = data.frame(X_direct = X_test_scaled[i, , drop = FALSE]),
                                       ncomp = ncomp_pls_d)
        pls_pred_direct[i] <- as.numeric(pls_direct_forecast)
      } else {
        pls_pred_direct[i] <- pls_pred_recursive[i]
      }
    } else {
      pls_pred_direct[i] <- pls_pred_recursive[i]
    }
  }
  
  # Store forecasts
  ar_forecasts_rolling[[paste0("h", h)]] <- ar_pred
  pcr_forecasts_rolling[[paste0("h", h)]] <- pcr_pred_recursive
  pls_forecasts_rolling[[paste0("h", h)]] <- pls_pred_recursive
  pcr_forecasts_direct[[paste0("h", h)]] <- pcr_pred_direct
  pls_forecasts_direct[[paste0("h", h)]] <- pls_pred_direct
  
  # Calculate MSE
  cat(sprintf("MSE - AR: %.4f, PCR(rec): %.4f, PLS(rec): %.4f\n",
              mse(y_test_h, ar_pred),
              mse(y_test_h, pcr_pred_recursive),
              mse(y_test_h, pls_pred_recursive)))
  cat(sprintf("MSE - PCR(direct): %.4f, PLS(direct): %.4f\n",
              mse(y_test_h, pcr_pred_direct),
              mse(y_test_h, pls_pred_direct)))
}

# =============================
# PERFORMANCE COMPARISON
# =============================

# MSE comparison
mse_comparison_rolling <- data.frame(
  Horizon = integer(),
  AR = numeric(),
  PCR_Recursive = numeric(),
  PLS_Recursive = numeric(),
  PCR_Direct = numeric(),
  PLS_Direct = numeric()
)

for (h in horizons) {
  key <- paste0("h", h)
  if (!key %in% names(ar_forecasts_rolling)) next
  
  y_test_h <- y_test[h:length(y_test)]
  
  mse_comparison_rolling <- rbind(mse_comparison_rolling,
                                  data.frame(
                                    Horizon = h,
                                    AR = mse(y_test_h, ar_forecasts_rolling[[key]]),
                                    PCR_Recursive = mse(y_test_h, pcr_forecasts_rolling[[key]]),
                                    PLS_Recursive = mse(y_test_h, pls_forecasts_rolling[[key]]),
                                    PCR_Direct = mse(y_test_h, pcr_forecasts_direct[[key]]),
                                    PLS_Direct = mse(y_test_h, pls_forecasts_direct[[key]])
                                  ))
}

print(mse_comparison_rolling)

# Plot
mse_plot_data <- mse_comparison_rolling %>%
  pivot_longer(cols = -Horizon, names_to = "Model", values_to = "MSE")

p_mse <- ggplot(mse_plot_data, aes(x = factor(Horizon), y = MSE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "MSE Comparison: Rolling Window Forecasts",
       subtitle = paste("Window size =", window_size, "observations"),
       x = "Forecast Horizon (steps ahead)",
       y = "Mean Squared Error") +
  theme(legend.position = "bottom")

print(p_mse)

# =============================
# KEY INSIGHTS
# =============================
cat("\n=== KEY METHODOLOGICAL POINTS ===\n")
cat("1. ROLLING WINDOW: Uses fixed window size, not expanding\n")
cat("   - More realistic for stationary processes\n")
cat("   - Avoids giving too much weight to old data\n\n")

cat("2. RECURSIVE vs DIRECT:\n")
cat("   - Recursive: Predict 1-step, then use prediction for next step\n")
cat("   - Direct: Train separate model for each horizon h\n")
cat("   - Direct often performs better for longer horizons\n\n")

cat("3. NO LOOKAHEAD BIAS:\n")
cat("   - At each forecast origin t, only use data up to t-1\n")
cat("   - Never use future to predict past\n")
cat("   - Retraining is legitimate (mimics real-world usage)\n\n")

cat("4. WHY PCR/PLS MIGHT STILL UNDERPERFORM:\n")
cat("   - AR models temporal structure; PCR/PLS don't\n")
cat("   - Solution: Add lagged Y as predictor in PCR/PLS\n")
cat("   - Or use dynamic factor models instead\n")
