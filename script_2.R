# set working directory
setwd("C:/Users/russn/Downloads/")

library(readr)
library(dplyr)
library(zoo)
library(pls)
library(ggplot2)
library(lubridate)
library(forecast)

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
#data <- data %>%
#  mutate(
#    CP3Mx = na.approx(CP3Mx, na.rm = FALSE),
#    COMPAPFFx = na.approx(COMPAPFFx, na.rm = FALSE)
#  )
data <- data %>%
  arrange(sasdate) %>%
  mutate(
    CP3Mx = na.approx(CP3Mx, x = as.numeric(sasdate), na.rm = FALSE, rule = 2),
    COMPAPFFx = na.approx(COMPAPFFx, x = as.numeric(sasdate), na.rm = FALSE, rule = 2)
  )

# check that CP3Mx and COMPAPFFx is filled (should give 0, 0)
colSums(is.na(data[c("CP3Mx", "COMPAPFFx")]))

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

# Also create full scaled matrix for rolling routines (scale with training stats)
X_full <- data %>% dplyr::select(-sasdate, -INDPRO, -INDPRO_growth)
X_full_scaled <- scale(X_full,
                       center = attr(X_train_scaled, "scaled:center"),
                       scale  = attr(X_train_scaled, "scaled:scale"))
X_full_scaled <- as.data.frame(X_full_scaled)

# helper full series
y_full <- data$INDPRO_growth
dates  <- data$sasdate

# ---------------------------
# 2) PCA (exploratory) - scree plot assigned to p & printed
# ---------------------------
pca_result <- prcomp(X_train_scaled, center = FALSE, scale. = FALSE)
var_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
p <- ggplot(data.frame(PC = 1:length(var_explained), Variance = var_explained),
            aes(x = PC, y = Variance)) +
  geom_line() + geom_point() + theme_minimal() +
  ggtitle("Scree Plot of PCA") + ylab("Proportion of Variance Explained")
print(p)

# ---------------------------
# Utility functions
# ---------------------------
mse  <- function(a, f) mean((a - f)^2, na.rm = TRUE)
rmse <- function(a, f) sqrt(mse(a, f))
mae  <- function(a, f) mean(abs(a - f), na.rm = TRUE)

select_ncomp_safe <- function(model_obj, fallback = 5) {
  # Try selectNcomp (onesigma), else RMSEP argmin, else fallback
  n <- NA
  try({
    n_try <- try(selectNcomp(model_obj, method = "onesigma", comp = base::min(10, ncol(model_obj$X))), silent = TRUE)
    if (!inherits(n_try, "try-error") && !is.na(n_try)) n <- n_try
  }, silent = TRUE)
  if (is.na(n)) {
    rm <- tryCatch(RMSEP(model_obj)$val, error = function(e) NULL)
    if (!is.null(rm)) {
      rm_vec <- as.numeric(rm[1, -1, drop = TRUE])
      if (length(rm_vec) > 0) n <- which.min(rm_vec)
    }
  }
  if (is.na(n)) n <- fallback
  as.integer(n)
}

dm_test_safe <- function(e1, e2, h = 1) {
  out <- tryCatch({
    res <- forecast::dm.test(e1, e2, h = h, power = 2, alternative = "two.sided")
    list(stat = as.numeric(res$statistic), p.value = as.numeric(res$p.value))
  }, error = function(e) list(stat = NA, p.value = NA))
  out
}

# ---------------------------
# 3) PCR & PLS baseline models (train on training set)
# ---------------------------
train_df <- cbind(y_train = y_train, as.data.frame(X_train_scaled))
pcr_model <- pcr(y_train ~ ., data = train_df, validation = "CV", scale = FALSE)
pls_model <- plsr(y_train ~ ., data = train_df, validation = "CV", scale = FALSE)

best_ncomp_pcr <- select_ncomp_safe(pcr_model, fallback = 5)
best_ncomp_pls <- select_ncomp_safe(pls_model, fallback = 5)
cat("Selected ncomp (PCR):", best_ncomp_pcr, "\n")
cat("Selected ncomp (PLS):", best_ncomp_pls, "\n")

# In-sample fitted & out-of-sample single-step (h=1) predictions
pcr_pred_in  <- as.numeric(predict(pcr_model, ncomp = best_ncomp_pcr))
pls_pred_in  <- as.numeric(predict(pls_model, ncomp = best_ncomp_pls))

X_test_scaled_df <- as.data.frame(X_test_scaled)
pcr_pred_out <- as.numeric(predict(pcr_model, newdata = X_test_scaled_df, ncomp = best_ncomp_pcr))
pls_pred_out <- as.numeric(predict(pls_model, newdata = X_test_scaled_df, ncomp = best_ncomp_pls))

# ---------------------------
# 4) Naive benchmarks: lag-1 and AR(p) (static & rolling)
# ---------------------------
# lag-1 naive aligned to test period
lag1_full <- dplyr::lag(y_full, 1)
y_test_naive <- lag1_full[dates >= as.Date("2016-01-01")]
valid_idx_naive <- !is.na(y_test_naive)
y_test_valid <- y_test[valid_idx_naive]
y_test_naive <- y_test_naive[valid_idx_naive]

# AR(p) static using training sample (order by AIC via ar())
ar_static_fit <- tryCatch(ar(y_train, aic = TRUE, order.max = 24), error = function(e) NULL)
ar_static_order <- if (!is.null(ar_static_fit)) ar_static_fit$order else NA
cat("Static AR selected order:", ar_static_order, "\n")
ar_static_forecast <- rep(NA, length(y_test))
if (!is.null(ar_static_fit)) {
  ar_fc <- tryCatch(predict(ar_static_fit, n.ahead = length(y_test)), error = function(e) NULL)
  if (!is.null(ar_fc)) ar_static_forecast <- as.numeric(ar_fc$pred[seq_along(y_test)])
}
mse_ar_static <- mse(y_test, ar_static_forecast)
cat("Static AR MSE on test set:", mse_ar_static, "\n")

# Rolling AR (expanding-window) one-step forecasts across test period
rolling_ar_forecasts <- rep(NA, length(y_test))
for (i in seq_along(y_test)) {
  cutoff_date <- test_data$sasdate[i]
  train_idx_dyn <- which(dates < cutoff_date)
  if (length(train_idx_dyn) < 24) next
  y_train_dyn <- y_full[train_idx_dyn]
  ar_dyn <- tryCatch(ar(y_train_dyn, aic = TRUE, order.max = 24), error = function(e) NULL)
  if (is.null(ar_dyn)) next
  rolling_ar_forecasts[i] <- tryCatch(predict(ar_dyn, n.ahead = 1)$pred[1], error = function(e) NA)
}
mse_ar_rolling <- mse(y_test, rolling_ar_forecasts)
cat("Rolling AR MSE (expanding-window):", mse_ar_rolling, "\n")

# ---------------------------
# 5) STATIC (DIRECT) MULTI-STEP forecasts (PCR & PLS separate direct models per horizon)
# ---------------------------
HORIZONS <- c(1, 3, 6, 12)
forecast_start <- as.Date("2016-01-01")
ncomp_use <- base::max(1, base::min(5, best_ncomp_pcr))

multi_static_list <- list()
multi_static_stats_list <- list()
plots_static <- list()

for (h in HORIZONS) {
  # Build training data to predict y_{t+h} from X_t using training period (sasdate < 2016-01-01)
  # valid idxs are those t for which t+h exists in series
  valid_idx_all <- which((seq_along(dates) + h) <= length(dates))
  train_idx <- valid_idx_all[valid_idx_all <= which(dates < forecast_start) %>% base::max(., na.rm = TRUE)]
  if (length(train_idx) < 10) {
    multi_static_list[[paste0(h)]] <- data.frame(Horizon = h, Type = "Static", MSE_PCR = NA, MSE_PLS = NA)
    next
  }
  df_train_direct <- cbind(y_target = y_full[train_idx + h], as.data.frame(X_full_scaled[train_idx, , drop = FALSE]))
  df_train_direct <- df_train_direct[complete.cases(df_train_direct), , drop = FALSE]
  if (nrow(df_train_direct) < 10) {
    multi_static_list[[paste0(h)]] <- data.frame(Horizon = h, Type = "Static", MSE_PCR = NA, MSE_PLS = NA)
    next
  }
  # Fit direct PCR & PLS for this horizon
  pcr_direct <- pcr(y_target ~ ., data = df_train_direct, validation = "CV", scale = FALSE)
  pls_direct <- plsr(y_target ~ ., data = df_train_direct, validation = "CV", scale = FALSE)
  ncomp_pcr_dir <- select_ncomp_safe(pcr_direct, fallback = base::min(5, ncol(df_train_direct) - 1))
  ncomp_pls_dir <- select_ncomp_safe(pls_direct, fallback = base::min(5, ncol(df_train_direct) - 1))
  # Prepare test indices: origins t where t >= forecast_start and t+h exists
  origin_test_idx <- which(dates >= forecast_start & (seq_along(dates) + h) <= length(dates))
  if (length(origin_test_idx) == 0) {
    multi_static_list[[paste0(h)]] <- data.frame(Horizon = h, Type = "Static", MSE_PCR = NA, MSE_PLS = NA)
    next
  }
  X_test_direct <- as.data.frame(X_full_scaled[origin_test_idx, , drop = FALSE])
  actuals_direct <- y_full[origin_test_idx + h]
  pred_pcr_dir <- tryCatch(as.numeric(predict(pcr_direct, newdata = X_test_direct, ncomp = ncomp_pcr_dir)), error = function(e) rep(NA, length(origin_test_idx)))
  pred_pls_dir <- tryCatch(as.numeric(predict(pls_direct, newdata = X_test_direct, ncomp = ncomp_pls_dir)), error = function(e) rep(NA, length(origin_test_idx)))
  mse_pcr_dir <- mse(actuals_direct, pred_pcr_dir)
  mse_pls_dir <- mse(actuals_direct, pred_pls_dir)
  multi_static_list[[paste0(h)]] <- data.frame(Horizon = h, Type = "Static", MSE_PCR = mse_pcr_dir, MSE_PLS = mse_pls_dir)
  # Stats
  valid_rows <- complete.cases(actuals_direct, pred_pcr_dir, pred_pls_dir)
  if (sum(valid_rows) > 5) {
    yv <- actuals_direct[valid_rows]; pcrv <- pred_pcr_dir[valid_rows]; plsv <- pred_pls_dir[valid_rows]
    corr_pcr <- cor(yv, pcrv); corr_pls <- cor(yv, plsv); corr_forecasts <- cor(pcrv, plsv)
    dm <- dm_test_safe(yv - pcrv, yv - plsv, h = 1)
    ftest <- tryCatch(var.test(pcrv, plsv), error = function(e) list(p.value = NA))
    ttest <- tryCatch(t.test(pcrv, plsv), error = function(e) list(p.value = NA))
    multi_static_stats_list[[paste0(h)]] <- data.frame(Horizon = h, Corr_PCR_Actual = corr_pcr, Corr_PLS_Actual = corr_pls, Corr_PCR_PLS = corr_forecasts, DM_pvalue = dm$p.value, F_pvalue = ftest$p.value, T_pvalue = ttest$p.value)
  } else {
    multi_static_stats_list[[paste0(h)]] <- data.frame(Horizon = h, Corr_PCR_Actual = NA, Corr_PLS_Actual = NA, Corr_PCR_PLS = NA, DM_pvalue = NA, F_pvalue = NA, T_pvalue = NA)
  }
  # Plot actual vs forecasts (example plot for this horizon)
  plot_df <- data.frame(Index = seq_along(actuals_direct), Actual = actuals_direct, PCR = pred_pcr_dir, PLS = pred_pls_dir)
  p <- ggplot(plot_df, aes(x = Index)) +
    geom_line(aes(y = Actual, color = "Actual")) +
    geom_line(aes(y = PCR, color = "PCR (direct)"), linetype = "dashed") +
    geom_line(aes(y = PLS, color = "PLS (direct)"), linetype = "dotted") +
    theme_minimal() + labs(title = paste0("Static Direct Forecasts (H=", h, ")"), x = "Origin index", y = "INDPRO growth (%)", color = "Series")
  print(p)
  plots_static[[paste0("H", h)]] <- p
} # end h loop

multi_static_df <- do.call(rbind, multi_static_list)
multi_static_stats_df <- do.call(rbind, multi_static_stats_list)

print(multi_static_df)
print(multi_static_stats_df)

# ---------------------------
# 6) RECURSIVE (iterated) Multi-step forecasts (separate PCR & PLS chains)
# ---------------------------
multi_recursive_list <- list()
multi_recursive_stats_list <- list()
plots_recursive <- list()

origin_idx <- which(dates == forecast_start)
if (length(origin_idx) == 0) origin_idx <- which(dates >= forecast_start)[1]

for (h in HORIZONS) {
  # initialize separate chains
  X_dyn_pcr <- as.data.frame(X_full_scaled[1:origin_idx, , drop = FALSE]); y_dyn_pcr <- y_full[1:origin_idx]
  X_dyn_pls <- as.data.frame(X_full_scaled[1:origin_idx, , drop = FALSE]); y_dyn_pls <- y_full[1:origin_idx]
  
  preds_pcr <- rep(NA, h); preds_pls <- rep(NA, h)
  for (step in 1:h) {
    idx_forecast <- origin_idx + (step - 1)
    if (idx_forecast > nrow(X_full_scaled)) break
    X_future <- as.data.frame(X_full_scaled[idx_forecast, , drop = FALSE])
    
    fit_pcr_dyn <- tryCatch(pcr(y_dyn_pcr ~ ., data = cbind(y_dyn_pcr = y_dyn_pcr, X_dyn_pcr), validation = "none", ncomp = base::min(ncomp_use, base::max(1, ncol(X_dyn_pcr)))), error = function(e) NULL)
    fit_pls_dyn <- tryCatch(plsr(y_dyn_pls ~ ., data = cbind(y_dyn_pls = y_dyn_pls, X_dyn_pls), validation = "none", ncomp = base::min(ncomp_use, base::max(1, ncol(X_dyn_pls)))), error = function(e) NULL)
    
    preds_pcr[step] <- if (!is.null(fit_pcr_dyn)) tryCatch(as.numeric(predict(fit_pcr_dyn, newdata = X_future, ncomp = base::min(ncomp_use, base::max(1, ncol(X_future))))), error = function(e) NA) else NA
    preds_pls[step] <- if (!is.null(fit_pls_dyn)) tryCatch(as.numeric(predict(fit_pls_dyn, newdata = X_future, ncomp = base::min(ncomp_use, base::max(1, ncol(X_future))))), error = function(e) NA) else NA
    
    # append predictions to respective chains
    y_dyn_pcr <- c(y_dyn_pcr, preds_pcr[step]); X_dyn_pcr <- rbind(X_dyn_pcr, X_future)
    y_dyn_pls <- c(y_dyn_pls, preds_pls[step]); X_dyn_pls <- rbind(X_dyn_pls, X_future)
  }
  
  actual_indices <- origin_idx:(origin_idx + h - 1)
  y_actual_seq <- if (base::max(actual_indices) <= length(y_full)) y_full[actual_indices] else rep(NA, h)
  
  mse_pcr <- mse(y_actual_seq, preds_pcr)
  mse_pls <- mse(y_actual_seq, preds_pls)
  multi_recursive_list[[paste0(h)]] <- data.frame(Horizon = h, Type = "Recursive", MSE_PCR = mse_pcr, MSE_PLS = mse_pls)
  
  valid_rows <- complete.cases(y_actual_seq, preds_pcr, preds_pls)
  if (sum(valid_rows) > 5) {
    yv <- y_actual_seq[valid_rows]; pcrv <- preds_pcr[valid_rows]; plsv <- preds_pls[valid_rows]
    corr_pcr <- cor(yv, pcrv); corr_pls <- cor(yv, plsv); corr_forecasts <- cor(pcrv, plsv)
    dm <- dm_test_safe(yv - pcrv, yv - plsv, h = 1)
    ftest <- tryCatch(var.test(pcrv, plsv), error = function(e) list(p.value = NA))
    ttest <- tryCatch(t.test(pcrv, plsv), error = function(e) list(p.value = NA))
    multi_recursive_stats_list[[paste0(h)]] <- data.frame(Horizon = h, Corr_PCR_Actual = corr_pcr, Corr_PLS_Actual = corr_pls, Corr_PCR_PLS = corr_forecasts, DM_pvalue = dm$p.value, F_pvalue = ftest$p.value, T_pvalue = ttest$p.value)
  } else {
    multi_recursive_stats_list[[paste0(h)]] <- data.frame(Horizon = h, Corr_PCR_Actual = NA, Corr_PLS_Actual = NA, Corr_PCR_PLS = NA, DM_pvalue = NA, F_pvalue = NA, T_pvalue = NA)
  }
  
  # Plot actual vs recursive predictions (Step = 1..h)
  plot_df <- data.frame(Step = 1:h, Actual = y_actual_seq, PCR = preds_pcr, PLS = preds_pls)
  p <- ggplot(plot_df, aes(x = Step)) +
    geom_line(aes(y = Actual, color = "Actual"), linewidth = 1) +
    geom_line(aes(y = PCR, color = "PCR (recursive)"), linetype = "dashed") +
    geom_line(aes(y = PLS, color = "PLS (recursive)"), linetype = "dotted") +
    theme_minimal() + labs(title = paste0("Recursive Forecasts (H=", h, ") from origin ", forecast_start),
                           x = "Step", y = "INDPRO growth (%)", color = "Series")
  print(p)
  plots_recursive[[paste0("H", h)]] <- p
}

multi_recursive_df <- do.call(rbind, multi_recursive_list)
multi_recursive_stats_df <- do.call(rbind, multi_recursive_stats_list)

print(multi_recursive_df)
print(multi_recursive_stats_df)

# ---------------------------
# 7) ROLLING (EXPANDING WINDOW) MULTI-STEP forecasts (separate PCR & PLS fits per origin)
# ---------------------------
rolling_results_list <- list()
rolling_stats_list <- list()
plots_rolling <- list()

min_train_obs <- 60
for (h in HORIZONS) {
  errors_pcr <- c(); errors_pls <- c(); errors_naive <- c()
  forecasts_pcr <- c(); forecasts_pls <- c(); actuals_collected <- c()
  for (t in seq_len(nrow(data))) {
    current_date <- dates[t]
    if (current_date < forecast_start) next
    train_idx <- which(dates < current_date)
    if (length(train_idx) < min_train_obs) next
    
    X_train_dyn <- as.data.frame(X_full_scaled[train_idx, , drop = FALSE])
    y_train_dyn <- y_full[train_idx]
    
    # fit PCR and PLS on expanding window (use ncomp_use or smaller)
    pcr_fit <- tryCatch(pcr(y_train_dyn ~ ., data = cbind(y_train_dyn = y_train_dyn, X_train_dyn), validation = "none", ncomp = base::min(ncomp_use, ncol(X_train_dyn))), error = function(e) NULL)
    pls_fit <- tryCatch(plsr(y_train_dyn ~ ., data = cbind(y_train_dyn = y_train_dyn, X_train_dyn), validation = "none", ncomp = base::min(ncomp_use, ncol(X_train_dyn))), error = function(e) NULL)
    
    idx_target <- t + (h - 1)
    if (idx_target > nrow(X_full_scaled)) next
    
    X_test_dyn <- as.data.frame(X_full_scaled[idx_target, , drop = FALSE])
    y_actual <- y_full[idx_target]
    
    pred_pcr <- if (!is.null(pcr_fit)) tryCatch(as.numeric(predict(pcr_fit, newdata = X_test_dyn, ncomp = base::min(ncomp_use, ncol(X_test_dyn)))), error = function(e) NA) else NA
    pred_pls <- if (!is.null(pls_fit)) tryCatch(as.numeric(predict(pls_fit, newdata = X_test_dyn, ncomp = base::min(ncomp_use, ncol(X_test_dyn)))), error = function(e) NA) else NA
    pred_naive <- y_full[t]  # previous observed growth at origin t
    
    forecasts_pcr <- c(forecasts_pcr, pred_pcr)
    forecasts_pls <- c(forecasts_pls, pred_pls)
    actuals_collected <- c(actuals_collected, y_actual)
    errors_pcr <- c(errors_pcr, y_actual - pred_pcr)
    errors_pls <- c(errors_pls, y_actual - pred_pls)
    errors_naive <- c(errors_naive, y_actual - pred_naive)
  } # end rolling loop for origins
  
  mse_pcr_roll <- mean((errors_pcr)^2, na.rm = TRUE)
  mse_pls_roll <- mean((errors_pls)^2, na.rm = TRUE)
  mse_naive_roll <- mean((errors_naive)^2, na.rm = TRUE)
  rolling_results_list[[paste0(h)]] <- data.frame(Horizon = h, Type = "Rolling", MSE_PCR = mse_pcr_roll, MSE_PLS = mse_pls_roll, MSE_Naive = mse_naive_roll)
  
  # Stats for rolling: require enough valid cases
  valid_rows <- complete.cases(actuals_collected, forecasts_pcr, forecasts_pls)
  if (sum(valid_rows) > 10) {
    yv <- actuals_collected[valid_rows]; pcrv <- forecasts_pcr[valid_rows]; plsv <- forecasts_pls[valid_rows]
    corr_pcr <- cor(yv, pcrv); corr_pls <- cor(yv, plsv); corr_forecasts <- cor(pcrv, plsv)
    dm <- dm_test_safe(yv - pcrv, yv - plsv, h = 1)
    ftest <- tryCatch(var.test(pcrv, plsv), error = function(e) list(p.value = NA))
    ttest <- tryCatch(t.test(pcrv, plsv), error = function(e) list(p.value = NA))
    rolling_stats_list[[paste0(h)]] <- data.frame(Horizon = h, Corr_PCR_Actual = corr_pcr, Corr_PLS_Actual = corr_pls, Corr_PCR_PLS = corr_forecasts, DM_pvalue = dm$p.value, F_pvalue = ftest$p.value, T_pvalue = ttest$p.value)
  } else {
    rolling_stats_list[[paste0(h)]] <- data.frame(Horizon = h, Corr_PCR_Actual = NA, Corr_PLS_Actual = NA, Corr_PCR_PLS = NA, DM_pvalue = NA, F_pvalue = NA, T_pvalue = NA)
  }
  
  # Plot last 100 points of actual vs forecast (rolling) as example
  plot_n <- base::min(200, length(actuals_collected))
  plot_df <- data.frame(Index = seq_len(plot_n), Actual = tail(actuals_collected, plot_n), PCR = tail(forecasts_pcr, plot_n), PLS = tail(forecasts_pls, plot_n))
  p <- ggplot(plot_df, aes(x = Index)) +
    geom_line(aes(y = Actual, color = "Actual")) +
    geom_line(aes(y = PCR, color = "PCR (rolling)"), linetype = "dashed") +
    geom_line(aes(y = PLS, color = "PLS (rolling)"), linetype = "dotted") +
    theme_minimal() + labs(title = paste0("Rolling Forecasts (H=", h, ") - recent origins"), x = "Recent origins", y = "INDPRO growth (%)", color = "Series")
  print(p)
  plots_rolling[[paste0("H", h)]] <- p
} # end h loop

rolling_results_df <- do.call(rbind, rolling_results_list)
rolling_stats_df <- do.call(rbind, rolling_stats_list)

print(rolling_results_df)
print(rolling_stats_df)

# ---------------------------
# 8) Summaries & Comparison Table printing
# ---------------------------
cat("\n=== STATIC RESULTS ===\n")
print(multi_static_df)
cat("\n=== STATIC STATS ===\n")
print(multi_static_stats_df)

cat("\n=== RECURSIVE RESULTS ===\n")
print(multi_recursive_df)
cat("\n=== RECURSIVE STATS ===\n")
print(multi_recursive_stats_df)

cat("\n=== ROLLING RESULTS ===\n")
print(rolling_results_df)
cat("\n=== ROLLING STATS ===\n")
print(rolling_stats_df)

# Optional: save results to RData
# save(multi_static_df, multi_static_stats_df, multi_recursive_df, multi_recursive_stats_df, rolling_results_df, rolling_stats_df, file = "forecast_results_all.RData")

# End of script

