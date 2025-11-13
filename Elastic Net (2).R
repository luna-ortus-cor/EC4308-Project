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
# create elastic net dataframe

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

# lag dataset (4 lags for each variable) for elastic net
data_lagged <- make_lags(data, vars_to_lag, K = 4)

# drop first 4 rows to remove missing values
data_lagged <- data_lagged %>% 
  filter(!is.na(INDPRO_growth_L4))%>%
  select(-INDPRO_growth_L4) # drop this so that comparable to AR(3)

# keep only lagged predictors & target variable
model_data <- data_lagged %>%
  select(sasdate, INDPRO_growth, ends_with("_L1"), ends_with("_L2"), ends_with("_L3"), ends_with("_L4"))

#scale all predictors (including lags of INDPRO_growth but not INDPRO_growth)
# Columns to scale (exclude date and target)
cols_to_scale <- setdiff(names(model_data), c("sasdate", "INDPRO_growth"))

# Define training window end (same as your earlier split)
train_end <- as.Date("2015-12-01")

# Identify which rows are in-sample for computing scaling parameters
train_mask <- model_data$sasdate <= train_end

# Compute training-only means and standard deviations
mu  <- sapply(model_data[train_mask, cols_to_scale, drop = FALSE], mean, na.rm = TRUE)
sdv <- sapply(model_data[train_mask, cols_to_scale, drop = FALSE], sd,   na.rm = TRUE)
sdv[!is.finite(sdv) | sdv == 0] <- 1  # prevent divide-by-zero

# Apply same scaling parameters (mu/sd) to ALL rows (train + OOS)
X_scaled <- sweep(sweep(as.matrix(model_data[, cols_to_scale, drop = FALSE]), 2, mu, "-"), 2, sdv, "/")

# Recombine scaled predictors with unscaled target and date
model_data_scaled <- bind_cols(
  model_data %>% select(sasdate, INDPRO_growth),
  as.data.frame(X_scaled)
)

###############################################################################
# elastic net

#define variables
# ensure dataset is ordered in time
model_data_scaled <- model_data_scaled %>% arrange(sasdate)

# define target 
y <- model_data_scaled$INDPRO_growth

# define predictor matrix (exclude date and y)
X <- model_data_scaled %>%
  select(-sasdate, -INDPRO_growth) %>%
  as.matrix()

# define training and oos forecast windows
train_end = "2015-12-01"
oos_start = "2016-01-01"
oos_end   = "2025-06-01"  

train_index <- which(model_data_scaled$sasdate <= as.Date("2015-12-01"))
oos_index   <- which(model_data_scaled$sasdate >= as.Date("2016-01-01") &
                       model_data_scaled$sasdate <= as.Date("2025-06-01"))

# create train & test sets
X_train <- X[train_index, ]
y_train <- y[train_index]

X_oos <- X[oos_index, ]
y_oos <- y[oos_index]

# implement elastic net to select alpha
# Time-series-safe alpha selection for elastic net (forward-chaining inner CV)
library(glmnet)

set.seed(1)

alphas <- seq(0, 1, by = 0.1) # alpha grid
val_len <- 60 # no. of 1-step-ahead forecasts inside the training set to evaluate each α
n       <- nrow(X_train) # no. of obs in training data

t_start <- n - val_len - 1          # expanding window starts here
# we fit on 1..t and validate using t+1
t_end   <- n - 1                    # predict t+1 each step
# last usable t to train t+1. last t+1 = n

kfolds_inner <- 5                   # forward-chaining splits inside [1..t]

alpha_mse <- numeric(length(alphas)) # stores mse for each alpha

# loop over alpha
for (i in seq_along(alphas)) {
  a <- alphas[i]
  step_mse <- numeric(0)
  
  for (t in t_start:t_end) {
    # 1) define inner forward-chaining splits within 1..t
    #    split [1..t] into k contiguous blocks; for block j,
    #    train = 1..(start_j-1), validate = that block (no look-ahead).
    brks   <- floor(quantile(1:t, probs = seq(0, 1, length.out = kfolds_inner + 1)))
    brks[1] <- 1
    # ensure strictly increasing
    brks <- unique(brks)
    
    # 2) get a candidate lambda grid from a fit on 1..t (no CV)
    fit0 <- glmnet(x = X_train[1:t, , drop = FALSE],
                   y = y_train[1:t],
                   alpha = a,
                   standardize = FALSE,
                   family = "gaussian")
    lam_grid <- fit0$lambda
    
    # 3) evaluate each lambda by forward-chaining validation
    lam_loss <- rep(0, length(lam_grid))
    lam_n    <- rep(0, length(lam_grid))
    
    for (j in seq_len(length(brks) - 1)) {
      val_idx   <- brks[j]: (brks[j + 1] - 1)
      trn_end   <- min(val_idx) - 1
      if (trn_end < 5) next  # skip tiny train segments
      
      # fit on strictly earlier data
      fit_j <- glmnet(x = X_train[1:trn_end, , drop = FALSE],
                      y = y_train[1:trn_end],
                      alpha = a,
                      lambda = lam_grid,     # reuse same grid
                      standardize = FALSE,
                      family = "gaussian")
      
      pred_j <- predict(fit_j, newx = X_train[val_idx, , drop = FALSE], s = lam_grid)
      # pred_j is |val_idx| x |lam_grid|
      errs   <- sweep(pred_j, 1, y_train[val_idx], FUN = "-")^2
      lam_loss <- lam_loss + colSums(errs, na.rm = TRUE)
      lam_n    <- lam_n + length(val_idx)
    }
    
    if (sum(lam_n) == 0) next
    lam_mse <- lam_loss / lam_n
    lam_star <- lam_grid[ which.min(lam_mse) ]
    
    # 4) refit on 1..t with lam_star, then 1-step-ahead forecast at t+1
    fit_t <- glmnet(x = X_train[1:t, , drop = FALSE],
                    y = y_train[1:t],
                    alpha = a,
                    lambda = lam_star,
                    standardize = FALSE,
                    family = "gaussian")
    
    y_hat <- as.numeric(predict(fit_t, newx = X_train[t + 1, , drop = FALSE]))
    step_mse <- c(step_mse, (y_train[t + 1] - y_hat)^2)
  }
  
  alpha_mse[i] <- mean(step_mse, na.rm = TRUE)
}

alpha_star <- alphas[ which.min(alpha_mse) ]
list(alpha_star = alpha_star, alpha_mse = setNames(alpha_mse, alphas))

# plot alpha vs MSE diagram
plot(alphas, alpha_mse, type = "b", pch = 16,
     xlab = expression(alpha), ylab = "Mean OOS MSE",
     main = "Elastic Net: Alpha vs Validation Error")
abline(v = alpha_star, lty = 2)
points(alpha_star, min(alpha_mse), pch = 19, cex = 1.2)

###############################################################################
 # create block folds for lambda selection

# alpha_star = 0.6
k <- 5

# get a lambda grid from only training dataset
fit0 <- glmnet(as.matrix(X_train), y_train, alpha = alpha_star, standardize = FALSE, family = "gaussian")
lam_grid <- fit0$lambda

#forward-chaining CV over k contiguous blocks
brks <- floor(quantile(1:n, probs = seq(0, 1, length.out = k + 1)))
brks[1] <- 1; brks <- unique(brks)

lam_loss <- rep(0, length(lam_grid))
lam_n    <- rep(0, length(lam_grid))

for (j in seq_len(length(brks) - 1)) {
  val_idx <- brks[j]:(brks[j + 1] - 1)
  trn_end <- min(val_idx) - 1
  if (trn_end < 5) next
  
  fit_j <- glmnet(X[1:trn_end, , drop = FALSE], y[1:trn_end],
                  alpha = alpha_star, lambda = lam_grid,
                  standardize = FALSE, family = "gaussian")
  
  pred  <- predict(fit_j, newx = X[val_idx, , drop = FALSE], s = lam_grid)
  errs  <- sweep(pred, 1, y[val_idx], FUN = "-")^2
  lam_loss <- lam_loss + colSums(errs, na.rm = TRUE)
  lam_n    <- lam_n + length(val_idx)
}

lam_mse  <- lam_loss / lam_n
lambda_star_final <- lam_grid[which.min(lam_mse)]
###############################################################################
# generate oos forecasts with α = 0.6, λ = 0.0837

alpha_star  <- 0.6
lambda_star <- lambda_star_final 

fc_oos <- numeric(length(oos_index))

for (i in seq_along(oos_index)) {
  t <- oos_index[i] - 1                 # last in-sample index for this forecast
  # Fit on strictly past data 1..t
  fit_t <- glmnet(
    x = X[1:t, , drop = FALSE],
    y = y[1:t],
    alpha = alpha_star,
    lambda = lambda_star,               # fixed lambda
    standardize = FALSE,
    family = "gaussian"
  )
  # Forecast the next observation at t+1 (i.e., the OOS row)
  fc_oos[i] <- as.numeric(predict(fit_t, newx = X[oos_index[i], , drop = FALSE]))
}

mse <- function(a, b) mean((a - b)^2)
mae <- function(a, b) mean(abs(a - b))

oos_mse <- mse(y_oos, fc_oos)
oos_mae <- mae(y_oos, fc_oos)

oos_mse; oos_mae

# OOS R^2 vs a flat (training-mean) benchmark
train_mean <- mean(y[train_index])
R2_oos <- 1 - sum((y_oos - fc_oos)^2) / sum((y_oos - train_mean)^2)

###############################################################################
plot(model_data_scaled$sasdate[oos_index], y_oos,
     type = "l", col = "black", lwd = 1.2,
     ylab = "Industrial Production Growth (%)",
     xlab = "Date",
     xaxt = "n",
     main = "Elastic Net Out-of-Sample Forecasts")
axis(1, at = seq(from = as.Date("2016-01-01"),
                 to   = as.Date("2025-06-01"),
                 by   = "2 years"),
     labels = format(seq(as.Date("2016-01-01"),
                         as.Date("2025-06-01"),
                         by = "2 years"), "%Y"))

lines(model_data_scaled$sasdate[oos_index], fc_oos, col = "red", lwd = 1.2)

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


# compute R^2 excluding COVID
covid_idx <- which(model_data_scaled$sasdate[oos_index] >= as.Date("2020-01-01") &
                     model_data_scaled$sasdate[oos_index] <= as.Date("2020-12-01"))

R2_excl_covid <- 1 - sum((y_oos[-covid_idx] - fc_oos[-covid_idx])^2) /
  sum((y_oos[-covid_idx] - mean(y_train))^2)
R2_excl_covid

mse_excl_covid <- mse(y_oos[-covid_idx], fc_oos[-covid_idx])
mae_excl_covid <- mae(y_oos[-covid_idx], fc_oos[-covid_idx])

mse_excl_covid
mae_excl_covid

