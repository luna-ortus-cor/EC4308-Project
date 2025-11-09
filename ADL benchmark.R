# set working directory
setwd("C:/Users/Grace/Desktop/EC4308 Project")

library(readr)
library(dplyr)
library(zoo)

###############################################################################
# data cleaning
raw_data <- read_csv("2025-09-MD.csv")

# Convert to monthly growth (%)
data <- raw_data %>%
  mutate(INDPRO_growth = 100 * (INDPRO / lag(INDPRO) - 1)) %>%
  filter(!is.na(INDPRO_growth)) 

# Calculate % of missing values per column
na_share <- colMeans(is.na(data))
na_share[na_share > 0]

data$sasdate <- as.Date(data$sasdate, format = "%m/%d/%Y")

data <- data %>% 
  select(-ACOGNO)%>%
  filter(sasdate >= as.Date("1978-01-01"))%>%
  filter(!(sasdate %in% as.Date(c("2025-07-01", "2025-08-01"))))

# impute CP3Mx and COMPAPFFx using linear interpolation
data <- data %>%
  mutate(
    CP3Mx = na.approx(CP3Mx, na.rm = FALSE),
    COMPAPFFx = na.approx(COMPAPFFx, na.rm = FALSE)
  )

# check that CP3Mx and COMPAPFFx is filled (should give 0, 0)
colSums(is.na(data[c("CP3Mx", "COMPAPFFx")]))

###############################################################################

# create spreads
library(dplyr)
library(zoo)
library(lmtest)
library(sandwich)

# --- spreads & base df ---
df <- data %>%
  arrange(sasdate) %>%
  mutate(
    BAA_AAA = BAA - AAA,
    T10Y3M  = GS10 - TB3MS
  )

# --- windows ---
train_end <- as.Date("2015-12-01")
oos_start <- as.Date("2016-01-01")
oos_end   <- as.Date("2025-06-01")

# --- helpers ---
lag_k <- function(x, k) dplyr::lag(x, k)

add_x_lags <- function(dat, var, K = 3) {
  stopifnot(var %in% names(dat))
  for (k in 1:K) {
    dat[[paste0(var, "_L", k)]] <- dplyr::lag(dat[[var]], k)
  }
  dat
}

adl_base <- df %>% 
  transmute(date = sasdate, 
             y = INDPRO_growth, 
             BAA_AAA = BAA_AAA, 
             HOUST = HOUST, 
             T10Y3M = T10Y3M, 
             UMCSENTx = UMCSENTx, 
             y_L1 = lag_k(INDPRO_growth, 1), 
             y_L2 = lag_k(INDPRO_growth, 2), 
             y_L3 = lag_k(INDPRO_growth, 3) ) %>% 
  add_x_lags("BAA_AAA", K = 4) %>% 
  add_x_lags("HOUST", K = 4) %>% 
  add_x_lags("T10Y3M", K = 4) %>% 
  add_x_lags("UMCSENTx",K = 4) %>% 
  # ensure common sample where all required lags exist 
  filter(!is.na(BAA_AAA_L4))

# --- SETTINGS ---
p_star <- 3                 # fixed (already chosen)
K_max  <- 4                 # search up to 4 lags for each X
X_vars <- c("BAA_AAA","HOUST","T10Y3M","UMCSENTx")
adl_train <- adl_base %>% filter(date <= train_end)
adl_oos   <- adl_base %>% filter(date >= oos_start, date <= oos_end)


# helper
get_bic <- function(fml) BIC(lm(fml, data = adl_train))

# fixed AR rhs
rhs_ar <- paste0("y_L", 1:p_star, collapse = " + ")

# ========== Select k_x ∈ {0..K_max} for each X by BIC (conditional on p*=3) ==========
best_k <- setNames(integer(length(X_vars)), X_vars)
for (v in X_vars) {
  bic_k <- sapply(0:K_max, function(k){
    rhs <- rhs_ar
    if (k > 0) rhs <- paste(rhs, paste0(v, "_L", 1:k, collapse = " + "), sep = " + ")
    get_bic(as.formula(paste("y ~", rhs)))
  })
  best_k[v] <- which.min(bic_k) - 1
  cat("Best lags for", v, "by BIC:", best_k[v], "\n")
}

# Start with AR(p*) only
rhs <- c(paste0("y_L", 1:p_star))
current_bic <- BIC(lm(as.formula(paste("y ~", paste(rhs, collapse=" + "))), data = adl_train))

# Stepwise add indicators that passed the "keep" test
keep_vars <- names(best_k[best_k > 0])

for(v in keep_vars){
  k <- best_k[v]
  trial_rhs <- c(rhs, paste0(v, "_L", 1:k))
  fml_trial <- as.formula(paste("y ~", paste(trial_rhs, collapse=" + ")))
  trial_bic <- BIC(lm(fml_trial, data=adl_train))
  
  cat(v, ": Trial BIC:", trial_bic, "| Current BIC:", current_bic, "\n")
  
  if(trial_bic < current_bic){
    rhs <- trial_rhs     # Accept indicator
    current_bic <- trial_bic
    cat("→ KEEP", v, "with", k, "lags.\n\n")
  } else {
    cat("→ DROP", v, "\n\n")
  }
}
###############################################################################
# oos forecasting
library(lubridate)
library(Metrics)

final_rhs <- c("y_L1","y_L2","y_L3",
               "BAA_AAA_L1","BAA_AAA_L2","BAA_AAA_L3","UMCSENTx_L1", "UMCSENTx_L2", "UMCSENTx_L3")

fml_adl <- as.formula(paste("y ~", paste(final_rhs, collapse=" + ")))

fml_ar  <- as.formula("y ~ y_L1 + y_L2 + y_L3")

# forecast 1 step ahead
# We'll forecast one-step-ahead: y_{t+1}
oos_dates <- adl_oos$date
n_oos <- nrow(adl_oos)

fc_ar  <- numeric(n_oos)
fc_adl <- numeric(n_oos)
y_true <- adl_oos$y

for(i in seq_len(n_oos)) {
  
  # Expand the training sample up to t = oos_dates[i] - 1 month
  train_cutoff <- oos_dates[i] %m-% months(1)
  train_data <- adl_base %>% filter(date <= train_cutoff)
  
  # Fit AR(3)
  fit_ar  <- lm(fml_ar, data=train_data)
  fc_ar[i]  <- predict(fit_ar, newdata=adl_oos[i,])
  
  # Fit ADL benchmark
  fit_adl <- lm(fml_adl, data=train_data)
  fc_adl[i] <- predict(fit_adl, newdata=adl_oos[i,])
}

#oos forecast evaluation

rmse_ar  <- rmse(y_true, fc_ar)
rmse_adl <- rmse(y_true, fc_adl)

mae_ar  <- mae(y_true, fc_ar)
mae_adl <- mae(y_true, fc_adl)

# print table for comparison
cat("OOS RMSE:\nAR(3) =", rmse_ar, "\nADL =", rmse_adl, "\n\n")
cat("OOS MAE:\nAR(3) =", mae_ar, "\nADL =", mae_adl, "\n")

results <- data.frame(
  Model = c("AR(3)", "ADL Benchmark (AR(3)+BAA_AAA(4))"),
  RMSE = c(rmse_ar, rmse_adl),
  MAE  = c(mae_ar, mae_adl)
)

print(results)

# R^2 evaluation
# build dataframe
fcst_adl <- data.frame(
  date  = oos_dates,
  y_act = y_true,
  y_hat = fc_adl
)

# mean forecast baseline
train_mean <- mean(adl_train$y, na.rm = TRUE)

SSE_adl     <- sum((fcst_adl$y_act - fcst_adl$y_hat)^2, na.rm = TRUE)
SSE_mean    <- sum((fcst_adl$y_act - train_mean)^2, na.rm = TRUE)

R2_oos_adl <- 1 - SSE_adl / SSE_mean
R2_oos_adl

fcst_adl <- fcst_adl %>%
  mutate(
    year = format(date, "%Y"),
    y_rw = dplyr::lag(y_act, 1)   # Random-walk benchmark
  )

subsample_r2_mean_adl <- function(df) {
  SSE_model <- sum((df$y_act - df$y_hat)^2, na.rm = TRUE)
  SSE_mean  <- sum((df$y_act - train_mean)^2, na.rm = TRUE)
  1 - SSE_model / SSE_mean
}

subsample_r2_rw_adl <- function(df) {
  SSE_model <- sum((df$y_act - df$y_hat)^2, na.rm = TRUE)
  SSE_rw    <- sum((df$y_act - df$y_rw)^2,  na.rm = TRUE)
  1 - SSE_model / SSE_rw
}

# Yearly subsample R²
yearly_r2_adl <- fcst_adl %>%
  group_by(year) %>%
  summarise(
    n_obs  = n(),
    R2_mean = subsample_r2_mean_adl(cur_data()),
    R2_rw   = subsample_r2_rw_adl(cur_data())
  ) %>%
  arrange(year)

yearly_r2_adl



