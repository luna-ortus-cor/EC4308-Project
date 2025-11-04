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
# 1) PCA (exploratory)
# ---------------------------
pca_result <- prcomp(X_train_scaled, center = FALSE, scale. = FALSE)
var_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
pca_plot <- ggplot(data.frame(PC = 1:length(var_explained), Variance = var_explained),
                   aes(x = PC, y = Variance)) +
  geom_line() + geom_point() + theme_minimal() +
  ggtitle("Scree Plot of PCA") + ylab("Proportion of Variance Explained")
print(pca_plot)






# ---------------------------
# 1) Helper Functions
# ---------------------------
mse  <- function(a, f) mean((a - f)^2, na.rm = TRUE)
rmse <- function(a, f) sqrt(mse(a, f))
mae  <- function(a, f) mean(abs(a - f), na.rm = TRUE)

dm_test_safe <- function(e1, e2, h = 1) {
  tryCatch({
    res <- forecast::dm.test(e1, e2, h = h, power = 2, alternative = "two.sided")
    list(stat = as.numeric(res$statistic), p.value = as.numeric(res$p.value))
  }, error = function(e) list(stat = NA, p.value = NA))
}

select_ncomp_safe <- function(model_obj, fallback = 5) {
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

meanrev_forecast <- function(series, window) {
  if(length(series) < window) return(NA)
  mean(tail(series, window), na.rm = TRUE)
}

# ---------------------------
# 2) Forecasting Settings
# ---------------------------
HORIZONS <- c(1,3,6,12)
forecast_start <- as.Date("2016-01-01")
all_models <- c("PCR","PLS","AR","MeanRev")
meanrev_window <- 12

# ---------------------------
# 3) Static, Rolling, Recursive Forecasting
# ---------------------------
forecast_types <- c("Static","Rolling","Recursive")
results_list <- list()
mse_plot_df <- data.frame()
plot_forecasts <- list()
summary_dm <- list()

for(ftype in forecast_types){
  for(h in HORIZONS){
    
    origin_idx <- which(dates >= forecast_start)
    y_pred_mat <- matrix(NA, nrow=length(origin_idx), ncol=length(all_models))
    colnames(y_pred_mat) <- all_models
    y_actual <- y_full[origin_idx + (h-1)]
    
    for(i in seq_along(origin_idx)){
      idx <- origin_idx[i]
      if((idx + h - 1) > length(y_full)) next
      
      # Define training data depending on forecast type
      if(ftype=="Static"){
        train_idx <- 1:(idx-1)
      } else if(ftype=="Rolling"){
        train_idx <- 1:(idx-1)  # expanding window (could use fixed window if preferred)
      } else if(ftype=="Recursive"){
        train_idx <- 1:(idx-1)  # recursive uses same training but iteratively predicts multiple steps
      }
      
      y_train_tmp <- y_full[train_idx]
      X_train_tmp <- X_full_scaled[train_idx,,drop=FALSE]
      
      # PCR
      pcr_fit <- pcr(y_train_tmp ~ ., data = cbind(y_train_tmp,X_train_tmp), validation = "CV", scale=FALSE)
      ncomp_pcr <- select_ncomp_safe(pcr_fit)
      X_idx <- X_full_scaled[idx,,drop=FALSE]
      y_pred_mat[i,"PCR"] <- as.numeric(predict(pcr_fit, newdata=X_idx, ncomp=ncomp_pcr))
      
      # PLS
      pls_fit <- plsr(y_train_tmp ~ ., data = cbind(y_train_tmp,X_train_tmp), validation = "CV", scale=FALSE)
      ncomp_pls <- select_ncomp_safe(pls_fit)
      y_pred_mat[i,"PLS"] <- as.numeric(predict(pls_fit, newdata=X_idx, ncomp=ncomp_pls))
      
      # AR
      if(length(y_train_tmp) >= 2){
        ar_fit <- ar(y_train_tmp, aic=TRUE, order.max=24)
        ar_fc <- tryCatch(predict(ar_fit, n.ahead=h)$pred[h], error=function(e) NA)
        y_pred_mat[i,"AR"] <- ar_fc
      }
      
      # Mean-Reversion
      y_pred_mat[i,"MeanRev"] <- meanrev_forecast(y_train_tmp, meanrev_window)
      
      # For recursive: fill forecasted value for next steps (iterated)
      if(ftype=="Recursive" & h>1){
        for(k in 2:h){
          if((idx+k-1) > length(y_full)) break
          y_pred_mat[i,"AR"] <- tryCatch(predict(ar_fit, n.ahead=k)$pred[k], error=function(e) NA)
          y_pred_mat[i,"MeanRev"] <- meanrev_forecast(c(y_train_tmp, y_pred_mat[i,"MeanRev"]), meanrev_window)
          y_pred_mat[i,"PCR"] <- as.numeric(predict(pcr_fit, newdata=X_idx, ncomp=ncomp_pcr))
          y_pred_mat[i,"PLS"] <- as.numeric(predict(pls_fit, newdata=X_idx, ncomp=ncomp_pls))
        }
      }
      
    } # end idx loop
    
    # Compute stats
    mse_vals <- sapply(all_models, function(m) mse(y_actual, y_pred_mat[,m]))
    rmse_vals <- sapply(all_models, function(m) rmse(y_actual, y_pred_mat[,m]))
    mae_vals  <- sapply(all_models, function(m) mae(y_actual, y_pred_mat[,m]))
    
    # DM test pairwise
    dm_matrix <- matrix(NA, nrow=length(all_models), ncol=length(all_models))
    colnames(dm_matrix) <- rownames(dm_matrix) <- all_models
    for(m1 in all_models){
      for(m2 in all_models){
        dm_res <- dm_test_safe(y_actual - y_pred_mat[,m1], y_actual - y_pred_mat[,m2], h=1)
        dm_matrix[m1,m2] <- dm_res$p.value
      }
    }
    
    # Store
    results_list[[paste0(ftype,"_H",h)]] <- data.frame(Horizon=h, ForecastType=ftype, t(y_pred_mat),
                                                       MSE=t(mse_vals), RMSE=t(rmse_vals), MAE=t(mae_vals))
    summary_dm[[paste0(ftype,"_H",h)]] <- dm_matrix
    
    # For MSE plot
    mse_plot_df <- rbind(mse_plot_df, data.frame(Horizon=h, ForecastType=ftype, Model=all_models, MSE=mse_vals))
    
    # Forecast plot (first 50 points)
    plot_df <- data.frame(Index=1:min(50,length(y_actual)), Actual=y_actual[1:min(50,length(y_actual))],
                          PCR=y_pred_mat[1:min(50,length(y_actual)),"PCR"],
                          PLS=y_pred_mat[1:min(50,length(y_actual)),"PLS"],
                          AR=y_pred_mat[1:min(50,length(y_actual)),"AR"],
                          MeanRev=y_pred_mat[1:min(50,length(y_actual)),"MeanRev"])
    p <- ggplot(plot_df, aes(x=Index)) +
      geom_line(aes(y=Actual, color="Actual")) +
      geom_line(aes(y=PCR, color="PCR"), linetype="dashed") +
      geom_line(aes(y=PLS, color="PLS"), linetype="dotted") +
      geom_line(aes(y=AR, color="AR"), linetype="twodash") +
      geom_line(aes(y=MeanRev, color="MeanRev"), linetype="dotdash") +
      theme_minimal() +
      labs(title=paste0(ftype," Forecasts H=",h), y="INDPRO growth (%)")
    print(p)
    plot_forecasts[[paste0(ftype,"_H",h)]] <- p
    
  } # end horizon
} # end forecast type

# ---------------------------
# 4) Plot MSE across horizons
# ---------------------------
mse_plot <- ggplot(mse_plot_df, aes(x=Horizon, y=MSE, color=Model, linetype=ForecastType)) +
  geom_line() + geom_point() + theme_minimal() +
  ggtitle("MSE Across Horizons and Forecast Types") + ylab("MSE") + xlab("Forecast Horizon")
print(mse_plot)

# ---------------------------
# 5) Summarize Results
# ---------------------------
summary_metrics <- do.call(rbind, results_list)
print(summary_metrics)
print(summary_dm)

# ---------------------------
# 3) Static, Rolling, Recursive Forecasting with Moving Windows
# ---------------------------
forecast_types <- c("Static","Rolling","Recursive")
results_list <- list()
mse_plot_df <- data.frame()
plot_forecasts <- list()
summary_dm <- list()
window_size <- 120  # last 10 years of monthly data

for(ftype in forecast_types){
  for(h in HORIZONS){
    
    origin_idx <- which(dates >= forecast_start)
    y_pred_mat <- matrix(NA, nrow=length(origin_idx), ncol=length(all_models))
    colnames(y_pred_mat) <- all_models
    y_actual <- y_full[origin_idx + (h-1)]
    
    for(i in seq_along(origin_idx)){
      idx <- origin_idx[i]
      if((idx + h - 1) > length(y_full)) next
      
      # Define training data depending on forecast type
      if(ftype=="Static"){
        train_idx <- 1:(idx-1)
      } else if(ftype %in% c("Rolling","Recursive")){
        start_idx <- max(1, idx - window_size)
        train_idx <- start_idx:(idx-1)
      }
      
      y_train_tmp <- y_full[train_idx]
      X_train_tmp <- X_full_scaled[train_idx,,drop=FALSE]
      
      # PCR
      pcr_fit <- pcr(y_train_tmp ~ ., data = cbind(y_train_tmp,X_train_tmp), validation = "CV", scale=FALSE)
      ncomp_pcr <- select_ncomp_safe(pcr_fit)
      X_idx <- X_full_scaled[idx,,drop=FALSE]
      y_pred_mat[i,"PCR"] <- as.numeric(predict(pcr_fit, newdata=X_idx, ncomp=ncomp_pcr))
      
      # PLS
      pls_fit <- plsr(y_train_tmp ~ ., data = cbind(y_train_tmp,X_train_tmp), validation = "CV", scale=FALSE)
      ncomp_pls <- select_ncomp_safe(pls_fit)
      y_pred_mat[i,"PLS"] <- as.numeric(predict(pls_fit, newdata=X_idx, ncomp=ncomp_pls))
      
      # AR
      if(length(y_train_tmp) >= 2){
        ar_fit <- ar(y_train_tmp, aic=TRUE, order.max=24)
        ar_fc <- tryCatch(predict(ar_fit, n.ahead=h)$pred[h], error=function(e) NA)
        y_pred_mat[i,"AR"] <- ar_fc
      }
      
      # Mean-Reversion
      y_pred_mat[i,"MeanRev"] <- meanrev_forecast(y_train_tmp, meanrev_window)
      
      # Recursive multi-step iterated forecast
      if(ftype=="Recursive" & h>1){
        y_temp <- y_train_tmp
        for(k in 1:h){
          # AR
          if(length(y_temp) >= 2){
            ar_fit_k <- ar(y_temp, aic=TRUE, order.max=24)
            ar_fc_k <- tryCatch(predict(ar_fit_k, n.ahead=1)$pred[1], error=function(e) NA)
            if(k==h) y_pred_mat[i,"AR"] <- ar_fc_k
          }
          # Mean-reversion
          y_pred_mat[i,"MeanRev"] <- meanrev_forecast(tail(y_temp, meanrev_window), meanrev_window)
          # PCR/PLS remain the same (predict with current X_idx)
          y_pred_mat[i,"PCR"] <- as.numeric(predict(pcr_fit, newdata=X_idx, ncomp=ncomp_pcr))
          y_pred_mat[i,"PLS"] <- as.numeric(predict(pls_fit, newdata=X_idx, ncomp=ncomp_pls))
          
          # Append predicted AR for next iteration
          y_temp <- c(y_temp, ar_fc_k)
        }
      }
      
    } # end idx loop
    
    # Compute stats
    mse_vals <- sapply(all_models, function(m) mse(y_actual, y_pred_mat[,m]))
    rmse_vals <- sapply(all_models, function(m) rmse(y_actual, y_pred_mat[,m]))
    mae_vals  <- sapply(all_models, function(m) mae(y_actual, y_pred_mat[,m]))
    
    # DM test pairwise
    dm_matrix <- matrix(NA, nrow=length(all_models), ncol=length(all_models))
    colnames(dm_matrix) <- rownames(dm_matrix) <- all_models
    for(m1 in all_models){
      for(m2 in all_models){
        dm_res <- dm_test_safe(y_actual - y_pred_mat[,m1], y_actual - y_pred_mat[,m2], h=1)
        dm_matrix[m1,m2] <- dm_res$p.value
      }
    }
    
    # Store
    results_list[[paste0(ftype,"_H",h)]] <- data.frame(Horizon=h, ForecastType=ftype, t(y_pred_mat),
                                                       MSE=t(mse_vals), RMSE=t(rmse_vals), MAE=t(mae_vals))
    summary_dm[[paste0(ftype,"_H",h)]] <- dm_matrix
    
    # For MSE plot
    mse_plot_df <- rbind(mse_plot_df, data.frame(Horizon=h, ForecastType=ftype, Model=all_models, MSE=mse_vals))
    
    # Forecast plot (first 50 points)
    plot_df <- data.frame(Index=1:min(50,length(y_actual)), Actual=y_actual[1:min(50,length(y_actual))],
                          PCR=y_pred_mat[1:min(50,length(y_actual)),"PCR"],
                          PLS=y_pred_mat[1:min(50,length(y_actual)),"PLS"],
                          AR=y_pred_mat[1:min(50,length(y_actual)),"AR"],
                          MeanRev=y_pred_mat[1:min(50,length(y_actual)),"MeanRev"])
    p <- ggplot(plot_df, aes(x=Index)) +
      geom_line(aes(y=Actual, color="Actual")) +
      geom_line(aes(y=PCR, color="PCR"), linetype="dashed") +
      geom_line(aes(y=PLS, color="PLS"), linetype="dotted") +
      geom_line(aes(y=AR, color="AR"), linetype="twodash") +
      geom_line(aes(y=MeanRev, color="MeanRev"), linetype="dotdash") +
      theme_minimal() +
      labs(title=paste0(ftype," Forecasts H=",h), y="INDPRO growth (%)")
    print(p)
    plot_forecasts[[paste0(ftype,"_H",h)]] <- p
    
  } # end horizon
} # end forecast type

