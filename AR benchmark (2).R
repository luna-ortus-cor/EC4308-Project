# set working directory
#setwd("C:/Users/Grace/Desktop/EC4308 Project")

library(readr)
library(dplyr)
library(zoo)

###############################################################################
# data cleaning
raw_file <- read_csv("2025-09-MD.csv")

# extract transformation code
tcode_row <- raw_file[1, ]

data_raw  <- raw_file[-1, ] # data without transformation row

#convert sas date to raw
data_raw <- data_raw %>%
  mutate(sasdate = mdy(sasdate))%>%
  mutate(UMCSENTx = na.approx(UMCSENTx, na.rm = FALSE))

# Build a named integer vector of tcodes for the numeric columns
num_cols <- setdiff(names(data_raw), "sasdate")
tcodes <- suppressWarnings(as.integer(tcode_row[num_cols] %>% unlist()))
names(tcodes) <- num_cols

# ---- Define transformations ----
L <- function(x, k = 1) dplyr::lag(x, k)

apply_tcode <- function(x, tcode, scale100 = TRUE) {
  # guard for logs
  safe_log <- function(v) { v[v <= 0] <- NA_real_; log(v) }
  out <- switch(as.character(tcode),
                "1" = x,                                   # level
                "2" = x - L(x, 1),                         # Δx
                "3" = (x - L(x, 1)) - (L(x,1) - L(x,2)),   # Δ^2 x
                "4" = safe_log(x),                         # log x
                "5" = { y <- safe_log(x) - safe_log(L(x,1)); if (scale100) 100*y else y },                  # Δ log x
                "6" = { y <- (safe_log(x) - safe_log(L(x,1))) - (safe_log(L(x,1)) - safe_log(L(x,2))); 
                if (scale100) 100*y else y },      # Δ^2 log x
                "7" = { g <- (x / L(x,1)) - 1; d <- g - L(g,1); if (scale100) 100*d else d },               # Δ(simple growth)
                x  # fallback: return original
  )
  out
}

stationary <- data_raw %>%
  mutate(across(all_of(num_cols),
                ~ apply_tcode(.x, tcodes[cur_column()]),
                .names = "{.col}"))%>%
  rename(INDPRO_growth = INDPRO)

data <- stationary %>% 
  select(-ACOGNO)%>%
  mutate(
    CP3Mx = na.approx(CP3Mx, na.rm = FALSE),
    COMPAPFFx = na.approx(COMPAPFFx, na.rm = FALSE)
  )%>%
  filter(sasdate >= as.Date("1978-01-01"))%>%
  filter(!(sasdate %in% as.Date(c("2025-07-01", "2025-08-01"))))

###############################################################################
library(lmtest)
library(sandwich)

# create spreads
df <- data %>%
  arrange(sasdate) %>%
  mutate(
    BAA_AAA = BAA - AAA,
    T10Y3M = GS10 - TB3MS
  )

# define train/ test windows
train_end <- as.Date("2015-12-01")
oos_start <- as.Date("2016-01-01")
oos_end   <- as.Date("2025-06-01")

#define helper functions
lag_k <- function(x, k) dplyr::lag(x, k)

get_ic <- function(fit, type = c("AIC","BIC")) {
  type <- match.arg(type)
  if (type == "AIC") AIC(fit) else BIC(fit)
}

# build AR frame & select AR(p) by IC on a common sample
p_max <- 6

# create up to 6 lags of y
ar_frame <- df %>%
  transmute(
    date = sasdate,
    y = INDPRO_growth,
    y_L1 = lag_k(INDPRO_growth,1),
    y_L2 = lag_k(INDPRO_growth,2),
    y_L3 = lag_k(INDPRO_growth,3),
    y_L4 = lag_k(INDPRO_growth,4),
    y_L5 = lag_k(INDPRO_growth,5),
    y_L6 = lag_k(INDPRO_growth,6)
  ) %>%
  # drop rows until max lag available (ensures SAME sample for all p <= 6)
  filter(!is.na(y_L6))

# restrict to training period
ar_train <- ar_frame %>% filter(date <= train_end)

# fit AR(1)…AR(6) with OLS on same sample
ics <- lapply(1:p_max, function(p) {
  rhs <- paste0("y_L", 1:p, collapse = " + ")
  fit <- lm(as.formula(paste0("y ~ ", rhs)), data = ar_train)
  c(p = p, AIC = get_ic(fit,"AIC"), BIC = get_ic(fit,"BIC"))
})
ics <- do.call(rbind, ics) %>% as.data.frame()

# choose by BIC (more conservative)
p_star <- ics$p[which.min(ics$BIC)]
p_star #3

###############################################################################

# forecasting using AR benchmark
# Recursive expanding-window OOS forecasts
oos_dates <- seq(oos_start, oos_end, by = "month")

rhs <- paste0("y_L", 1:p_star, collapse = " + ")
fml <- as.formula(paste0("y ~ ", rhs))

fcst_list <- vector("list", length(oos_dates))

for (i in seq_along(oos_dates)) {
  t <- oos_dates[i]
  
  # 1) Estimation sample: all data up to t-1
  est <- ar_frame %>%
    filter(date <= train_end | (date < t & date > train_end)) %>%  # equivalently: date < t
    filter(!is.na(y), !is.na(.data[[paste0("y_L", p_star)]]))
  
  fit <- lm(fml, data = est)
  
  # 2) Forecast y(t) using ACTUAL lags available at date t
  x_new <- ar_frame %>%
    filter(date == t) %>%
    select(all_of(paste0("y_L", 1:p_star)))
  
  # If t is within data, x_new will exist and is built from actual history
  y_hat <- as.numeric(predict(fit, newdata = x_new))
  
  # 3) Store actual y(t) and the forecast
  y_act <- ar_frame$y[ar_frame$date == t]
  
  fcst_list[[i]] <- data.frame(
    date   = t,
    y_hat  = y_hat,
    y_act  = y_act
  )
}

fcst <- bind_rows(fcst_list) %>%
  mutate(error = y_act - y_hat)

# evaluation metrics
rmse <- sqrt(mean(fcst$error^2, na.rm = TRUE))
mae  <- mean(abs(fcst$error), na.rm = TRUE)

list(head_fcst = head(fcst, 5), tail_fcst = tail(fcst, 5), RMSE = rmse, MAE = mae)

# OOS R² (relative to mean model)
train_mean <- mean(ar_frame$y[ar_frame$date <= train_end], na.rm = TRUE)

SSE_model <- sum((fcst$y_act - fcst$y_hat)^2, na.rm = TRUE)
SSE_naive <- sum((fcst$y_act - train_mean)^2, na.rm = TRUE)

R2_oos <- 1 - SSE_model / SSE_naive
R2_oos

# subsample oos R^2 (mean reversion & random walk)
# Compute training mean (used as the mean forecast baseline)
train_mean <- mean(ar_frame$y[ar_frame$date <= train_end], na.rm = TRUE)

# Compute random-walk forecast baseline (y_rw = y_{t-1})
fcst <- fcst %>%
  mutate(
    year = format(date, "%Y"),
    y_rw = dplyr::lag(y_act, 1)
  )

# helper functions
subsample_r2_mean <- function(df) {
  SSE_model <- sum((df$y_act - df$y_hat)^2, na.rm = TRUE)
  SSE_mean  <- sum((df$y_act - train_mean)^2, na.rm = TRUE)
  1 - SSE_model / SSE_mean
}

subsample_r2_rw <- function(df) {
  SSE_model <- sum((df$y_act - df$y_hat)^2, na.rm = TRUE)
  SSE_rw    <- sum((df$y_act - df$y_rw)^2,  na.rm = TRUE)
  1 - SSE_model / SSE_rw
}

# Compute yearly R² values
yearly_r2 <- fcst %>%
  group_by(year) %>%
  summarise(
    n_obs  = n(),
    R2_mean = subsample_r2_mean(cur_data()),
    R2_rw   = subsample_r2_rw(cur_data())
  ) %>%
  arrange(year)

yearly_r2
