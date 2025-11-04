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
# Forecast Settings
# ---------------------------
library(Metrics)
library(lmtest)
library(plm)
library(reshape2)

horizons <- c(1,3,6,12)
rolling_window <- 120 # 10 years monthly
meanrev_window <- 12  # last year
ncomp_pcr <- 5
ncomp_pls <- 5

origin_idx <- which(dates >= as.Date("2016-01-01"))

methods <- c("PCR","PLS","AR","MeanRev")
modes <- c("Static","Rolling","Recursive")

# ---------------------------
# Utility Functions
# ---------------------------

compute_metrics <- function(y_true, y_pred){
  mse <- mean((y_true - y_pred)^2, na.rm = TRUE)
  rmse <- sqrt(mse)
  mae <- mean(abs(y_true - y_pred), na.rm = TRUE)
  mape <- mean(abs((y_true - y_pred)/y_true), na.rm = TRUE)*100
  return(c(MSE=mse, RMSE=rmse, MAE=mae, MAPE=mape))
}

# PCR / PLS
pcr_pls_forecast <- function(X_train, y_train, X_test, method="PCR", ncomp=5){
  df_train <- data.frame(y = y_train, X_train)
  df_test  <- data.frame(X_test)
  
  if(method=="PCR"){
    fit <- pls::pcr(y ~ ., data=df_train, ncomp=ncomp, validation="none")
  } else if(method=="PLS"){
    fit <- pls::plsr(y ~ ., data=df_train, ncomp=ncomp, validation="none")
  }
  
  pred_array <- predict(fit, newdata=df_test, ncomp=ncomp)
  
  # Safe extraction
  if(length(dim(pred_array)) == 3){       # >1 row in test set
    as.numeric(pred_array[1, ncomp, 1])
  } else if(length(dim(pred_array)) == 2){ # exactly 1 row in test set
    as.numeric(pred_array[1, ncomp])
  } else {                                # fallback
    as.numeric(pred_array[1])
  }
}


# Mean-reversion
meanrev_forecast <- function(y_window, n_steps){
  rep(mean(tail(y_window, n_steps)), n_steps)
}

# AR forecast
ar_forecast <- function(y_train, h){
  fit <- ar(y_train, aic=TRUE, order.max=12, method="ols")
  forecast::forecast(fit, h=h)$mean[h]
}

dm_matrix <- function(y_true, pred_list){
  n <- length(pred_list)
  dm_res <- matrix(NA, nrow=n, ncol=n)
  for(i in 1:n){
    for(j in 1:n){
      if(i != j){
        valid_idx <- which(!is.na(pred_list[[i]]) & !is.na(pred_list[[j]]) & !is.na(y_true))
        if(length(valid_idx) > 1){
          dm_res[i,j] <- tryCatch({
            dm.test(y_true[valid_idx], pred_list[[i]][valid_idx], pred_list[[j]][valid_idx], alternative="two.sided")$statistic
          }, error=function(e) NA)
        }
      }
    }
  }
  rownames(dm_res) <- colnames(dm_res) <- names(pred_list)
  return(dm_res)
}

# ---------------------------
# Preallocate Storage
# ---------------------------
forecast_results <- list()
metrics_results <- list()
dm_results <- list()

# ---------------------------
# Forecast Loop
# ---------------------------
for(mode in modes){
  cat("Mode:", mode, "\n")
  
  y_pred_all <- array(NA, dim=c(length(origin_idx), length(horizons), length(methods)),
                      dimnames=list(NULL, paste0("H",horizons), methods))
  
  for(o in seq_along(origin_idx)){
    idx <- origin_idx[o]
    
    if(mode=="Static"){
      train_idx <- 1:(idx-1)
    } else if(mode=="Rolling"){
      train_idx <- max(1, idx-rolling_window):(idx-1)
    } else if(mode=="Recursive"){
      train_idx <- 1:(idx-1)
    }
    
    y_train_tmp <- y_full[train_idx]
    X_train_tmp <- as.matrix(X_full_scaled[train_idx, ])
    
    for(h in horizons){
      test_idx <- idx + h - 1
      if(test_idx > length(y_full)) next
      X_test_tmp <- as.matrix(X_full_scaled[test_idx, , drop=FALSE])
      
      # PCR
      y_pred_all[o, paste0("H",h), "PCR"] <- pcr_pls_forecast(X_train_tmp, y_train_tmp, X_test_tmp, "PCR", ncomp_pcr)
      # PLS
      y_pred_all[o, paste0("H",h), "PLS"] <- pcr_pls_forecast(X_train_tmp, y_train_tmp, X_test_tmp, "PLS", ncomp_pls)
      # AR
      y_pred_all[o, paste0("H",h), "AR"] <- ar_forecast(y_train_tmp, h)
      # Mean-reversion
      y_pred_all[o, paste0("H",h), "MeanRev"] <- meanrev_forecast(y_train_tmp, meanrev_window)[h]
    }
  }
  
  forecast_results[[mode]] <- y_pred_all

  print(y_pred_all)
  
  # Metrics
  metrics_list <- list()
  for(m in methods){
    metrics_h <- matrix(NA, nrow=length(horizons), ncol=4, dimnames=list(paste0("H",horizons), c("MSE","RMSE","MAE","MAPE")))
    for(i in seq_along(horizons)){
      h <- horizons[i]
      valid_idx <- which(!is.na(y_pred_all[,paste0("H",h),m]))
      y_true <- as.numeric(sapply(origin_idx[valid_idx], function(x) y_full[x + h - 1]))
      y_pred <- as.numeric(y_pred_all[valid_idx,paste0("H",h),m])
      metrics_h[i,] <- compute_metrics(y_true, y_pred)
    }
    metrics_list[[m]] <- metrics_h
  }
  metrics_results[[mode]] <- metrics_list
  
  # DM Tests
  dm_list <- list()
  for(h in horizons){
    valid_idx <- which(!is.na(y_pred_all[,paste0("H",h),methods[1]]))
    y_true <- sapply(origin_idx[valid_idx], function(x) y_full[x + h - 1])
    pred_list <- lapply(methods, function(m) y_pred_all[valid_idx,paste0("H",h),m])
    names(pred_list) <- methods
    dm_list[[paste0("H",h)]] <- dm_matrix(y_true, pred_list)
  }
  dm_results[[mode]] <- dm_list
}

# ---------------------------
# Summary Table
# ---------------------------
summary_metrics <- data.frame()
for(mode in modes){
  for(m in methods){
    tmp <- metrics_results[[mode]][[m]]
    tmp_df <- as.data.frame(tmp)
    tmp_df$Horizon <- rownames(tmp)
    tmp_df$Method <- m
    tmp_df$Mode <- mode
    summary_metrics <- rbind(summary_metrics, tmp_df)
  }
}
print(summary_metrics)

# ---------------------------
# Plot MSE per Model / Horizon
# ---------------------------
plot_data <- summary_metrics %>% dplyr::select(Horizon, Method, Mode, MSE)
ggplot(plot_data, aes(x=Horizon, y=MSE, color=Method, group=Method)) +
  geom_line() + geom_point() +
  facet_wrap(~Mode, scales="free_y") +
  theme_bw() + ggtitle("MSE per Model / Horizon / Mode")

# ---------------------------
# Plot Forecasts vs Actual (first 100 points)
# ---------------------------
plot_points <- 100
df_plot <- data.frame(
  Date = dates[origin_idx][1:plot_points],
  Actual = y_full[origin_idx][1:plot_points]
)
for(m in methods){
  df_plot[[m]] <- sapply(1:plot_points, function(i) forecast_results[["Static"]][i,"H1",m])
}
df_melt <- melt(df_plot, id.vars="Date")
ggplot(df_melt, aes(x=Date, y=value, color=variable)) +
  geom_line() + theme_bw() + ggtitle("Forecast vs Actual (Static, h=1)")

