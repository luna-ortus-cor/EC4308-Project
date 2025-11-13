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
###############################################################################

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
library(randomForest)

# select optimal nodesize
# Validation period: last 24 months within training window
val_index <- which(model_data$sasdate > as.Date("2014-01-01") &
                     model_data$sasdate <= as.Date("2015-12-01"))

train_index_sub <- which(model_data$sasdate <= as.Date("2013-12-01"))

# Train subset (early period)
X_train_sub <- X[train_index_sub, ]
y_train_sub <- y[train_index_sub]

# Validation subset (recent part of training)
X_val <- X[val_index, ]
y_val <- y[val_index]

for (nsize in c(2, 5, 10, 20)) {
  fit <- randomForest(x=X_train, y=y_train,
                      ntree=500, mtry=floor(sqrt(ncol(X_train))),
                      nodesize=nsize)
  preds <- predict(fit, newdata=X_val)
  mse <- mean((y_val - preds)^2)
  print(c(nodesize=nsize, MSE=mse))
}

#select optimal ntree
rf_fit <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 1000,
  mtry = floor(sqrt(ncol(X_train))),
  nodesize = 2,
  importance = TRUE
)

# Plot OOB (out-of-bag) error vs number of trees
plot(rf_fit)
###############################################################################
# static random forest
set.seed(1)
rf_fit <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 500,
  mtry = floor(sqrt(ncol(X_train))),  # typical default
  nodesize = 2,
  importance = TRUE
)

# OOS predictions
y_hat_oos <- predict(rf_fit, newdata = X_oos)

# Simple OOS metrics
mse <- function(a,b) mean((a-b)^2)
mae  <- function(a,b) mean(abs(a-b))

rf_mse <- mse(y_oos, y_hat_oos)
rf_mae  <- mae(y_oos, y_hat_oos)

R2_oos <- 1 - sum((y_oos - y_hat_oos)^2) / sum((y_oos - mean(y_train))^2)

cat(sprintf("RF OOS MSE  = %.4f\nRF OOS MAE  = %.4f\nRF OOS R^2  = %.4f\n",
            rf_mse, rf_mae, R2_oos))

plot(model_data$sasdate[oos_index], y_oos,
     type = "l", col = "black", lwd = 1.2,
     ylab = "Industrial Production Growth (%)",
     xlab = "Date",
     xaxt = "n",
     main = "Static Random Forest Out-of-Sample Forecasts")
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
