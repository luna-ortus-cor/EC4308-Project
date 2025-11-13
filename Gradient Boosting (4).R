# setwd("C:/Users/Grace/Desktop/EC4308 Project")
library(readr)
library(dplyr)
library(lubridate)
library(zoo)

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

# select variables to lag (all columns other than sasdate)
vars_to_lag <- setdiff(names(data), "sasdate")

# helper function to create lags
make_lags <- function(df, vars, K = 4) {
  for (v in vars) {
    for (k in 1:K) {
      df[[paste0(v, "_L", k)]] <- dplyr::lag(df[[v]], k)
    }
  }
  return(df)
}

# lag dataset (4 lags for each variable)
data_lagged <- make_lags(data, vars_to_lag, K = 4)

# drop first 4 rows to remove missing values
data_lagged <- data_lagged %>% 
  filter(!is.na(INDPRO_growth_L4))%>%
  select(-INDPRO_growth_L4) # drop this so that comparable to AR(3)   

# keep only lagged predictors & target variable
model_data <- data_lagged %>%
  select(sasdate, INDPRO_growth, ends_with("_L1"), ends_with("_L2"), ends_with("_L3"), ends_with("_L4"))

# define training and oos forecast windows
train_end = "2015-12-01"
oos_start = "2016-01-01"
oos_end   = "2025-06-01"

# define target 
y <- model_data$INDPRO_growth

# define predictor matrix (exclude date and y)
X <- model_data %>%
  select(-sasdate, -INDPRO_growth)

train_index <- which(model_data$sasdate <= as.Date("2015-12-01"))
oos_index   <- which(model_data$sasdate >= as.Date("2016-01-01") &
                       model_data$sasdate <= as.Date("2025-06-01"))

# create train & test sets
X_train <- X[train_index, ]
y_train <- y[train_index]

X_oos <- X[oos_index, ]
y_oos <- y[oos_index]

###############################################################################

###############################################################################
# hyperparameter tuning
library(xgboost)
library(parallel)

set.seed(1)
ts_cv_xgb <- function(X, y, params, nrounds=350, early_stopping_rounds=20,
                      initial=NULL, assess=3, step=3) {
  n <- nrow(X)
  if (is.null(initial)) initial <- max(120, floor(0.7*n))
  starts <- seq(initial, n - assess, by = step)
  rmses <- numeric(length(starts))
  best_iters <- integer(length(starts))
  
  dfull <- xgb.DMatrix(as.matrix(X), label = y)
  
  for (i in seq_along(starts)) {
    tr_end <- starts[i]
    tr_idx <- 1:tr_end
    va_idx <- (tr_end+1):(tr_end+assess)
    
    dtr <- xgboost::slice(dfull, tr_idx)
    dva <- xgboost::slice(dfull, va_idx)
    
    fit <- xgb.train(
      params = params,
      data = dtr,
      nrounds = nrounds,
      watchlist = list(train = dtr, val = dva),
      early_stopping_rounds = early_stopping_rounds,
      verbose = 0
    )
    preds <- predict(fit, dva, ntreelimit = fit$best_iteration)
    rmses[i] <- sqrt(mean((y[va_idx] - preds)^2))
    best_iters[i] <- fit$best_iteration
  }
  list(cv_rmse = mean(rmses), best_nrounds = round(median(best_iters)))
}

gr <- expand.grid(
  eta = c(0.2, 0.25),
  max_depth = 2,           # shallow trees
  min_child_weight = c(5, 10),  # fewer splits
  subsample = c(0.5, 0.6),
  colsample_bytree = c(0.4, 0.5),
  KEEP.OUT.ATTRS = FALSE
)

res <- lapply(seq_len(nrow(gr)), function(i){
  p <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    seed = 1,
    nthread = max(1, detectCores() - 1),
    tree_method = "hist",
    max_bin = 128,
    eta = gr$eta[i],
    max_depth = gr$max_depth[i],
    min_child_weight = gr$min_child_weight[i],
    subsample = gr$subsample[i],
    colsample_bytree = gr$colsample_bytree[i]
  )
  out <- ts_cv_xgb(
    X_train, y_train, p,
    nrounds = 350, early_stopping_rounds = 20,
    initial = max(120, floor(0.7*nrow(X_train))),
    assess = 3, step = 3
  )
  data.frame(gr[i,], cv_rmse = out$cv_rmse, best_nrounds = out$best_nrounds)
})

results <- do.call(rbind, res)
best <- results[which.min(results$cv_rmse), ]
best

###############################################################################
# final evaluation
# Build params 
params_best <- list(
  objective        = "reg:squarederror",
  eval_metric      = "rmse",
  seed             = 1,
  tree_method      = "hist",
  max_bin          = 128,
  eta              = best$eta,
  max_depth        = best$max_depth,
  min_child_weight = best$min_child_weight,
  subsample        = best$subsample,
  colsample_bytree = best$colsample_bytree
)

# Train once on the full training sample
dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)
xgb_fit <- xgb.train(
  params  = params_best,
  data    = dtrain,
  nrounds = best$best_nrounds,
  verbose = 0
)

# Static OOS predictions
y_hat_oos <- predict(xgb_fit, as.matrix(X_oos))

# Metrics
mse   <- mean((y_oos - y_hat_oos)^2)
mae   <- mean(abs(y_oos - y_hat_oos))
R2_oos <- 1 - sum((y_oos - y_hat_oos)^2) / sum((y_oos - mean(y_train))^2)

cat(sprintf("Static OOS MSE  : %.6f\n", mse))
cat(sprintf("Static OOS MAE  : %.6f\n", mae))
cat(sprintf("Static OOS R^2  : %.6f\n", R2_oos))

plot(model_data$sasdate[oos_index], y_oos,
     type = "l", col = "black", lwd = 1.2,
     ylab = "Industrial Production Growth (%)",
     xlab = "Date",
     xaxt = "n",
     main = "Static Grdient Boosting Out-of-Sample Forecasts")
axis(1, at = seq(from = as.Date("2016-01-01"),
                 to   = as.Date("2025-06-01"),
                 by   = "2 years"),
     labels = format(seq(as.Date("2016-01-01"),
                         as.Date("2025-06-01"),
                         by = "2 years"), "%Y"))

lines(model_data$sasdate[oos_index], y_hat_oos, col = "red", lwd = 1.2)

legend("topleft",
       legend = c("Actual", "Predicted"),
       col = c("black", "red"),
       lty = 1, lwd = 1.5,
       bty = "n",        
       cex = 0.7,        
       seg.len = 0.6,
       x.intersp = 0.3,
       y.intersp = 0.4,
       inset = c(0.02, 0.02))

