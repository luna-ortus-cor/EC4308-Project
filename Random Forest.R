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

# lag dataset (4 lags for each variable) for elastic net
data_lagged <- make_lags(data, vars_to_lag, K = 4)

# drop first 4 rows to remove missing values
data_lagged <- data_lagged %>% 
  filter(!is.na(INDPRO_growth_L4))   

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

set.seed(1)
rf_fit <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 500,
  mtry = floor(sqrt(ncol(X_train))),  # typical default
  nodesize = 5,
  importance = TRUE
)

# OOS predictions
y_hat_oos <- predict(rf_fit, newdata = X_oos)

# Simple OOS metrics
rmse <- function(a,b) sqrt(mean((a-b)^2))
mae  <- function(a,b) mean(abs(a-b))

rf_rmse <- rmse(y_oos, y_hat_oos)
rf_mae  <- mae(y_oos, y_hat_oos)
#RF OOS RMSE = 1.6838
#RF OOS MAE  = 0.7315

cat(sprintf("RF OOS RMSE = %.4f\nRF OOS MAE  = %.4f\n", rf_rmse, rf_mae))


###############################################################################
# rolling window cross validation

# install.packages("ranger")  # if needed
library(ranger)

set.seed(1)

# --- indices & data frames ---
dates_train <- model_data$sasdate[train_index]
Xtr <- as.data.frame(X_train)
ytr <- as.numeric(y_train)

# --- rolling CV setup (expanding window, 1-step ahead) ---
n <- nrow(Xtr)
initial_window <- max(120, ceiling(0.6 * n))  # ~10y or 60% of train, whichever larger
fold_starts <- seq(from = initial_window, to = n - 1, by = 1)  # each step predicts t+1

# --- small, fast grid ---
p <- ncol(Xtr)
grid <- expand.grid(
  mtry = unique(round(c(sqrt(p), 2*sqrt(p)))),
  min.node.size = c(5, 10),
  num.trees = c(300)  # keep small for CV speed
)

# helper: one-step ahead fit/predict with x/y interface (avoids your earlier error)
one_step_pred <- function(tr_end, pars) {
  tr_idx <- 1:tr_end
  val_idx <- tr_end + 1
  fit <- ranger(
    x = Xtr[tr_idx, , drop = FALSE],
    y = ytr[tr_idx],
    num.trees = pars$num.trees,
    mtry = pars$mtry,
    min.node.size = pars$min.node.size,
    sample.fraction = 0.8,          # faster, adds robustness
    respect.unordered.factors = "order",
    write.forest = TRUE,           # faster during CV
    num.threads = max(1, parallel::detectCores() - 1),
    seed =1
  )
  pred <- predict(fit, data = Xtr[val_idx, , drop = FALSE])$predictions
  c(y_true = ytr[val_idx], y_hat = pred)
}

# --- run rolling CV ---
cv_results <- lapply(seq_len(nrow(grid)), function(g) {
  pars <- grid[g, ]
  preds <- matrix(NA_real_, nrow = length(fold_starts), ncol = 2,
                  dimnames = list(NULL, c("y_true","y_hat")))
  k <- 1
  for (t in fold_starts) {
    pr <- one_step_pred(t, pars)
    preds[k, ] <- pr
    k <- k + 1
  }
  data.frame(
    mtry = pars$mtry,
    min.node.size = pars$min.node.size,
    num.trees = pars$num.trees,
    rmse = rmse(preds[,"y_true"], preds[,"y_hat"])
  )
})

cv_summary <- do.call(rbind, cv_results)
cv_summary <- cv_summary[order(cv_summary$rmse), ]
print(cv_summary)
best <- cv_summary[1, ]

# --- refit ONCE on full training with best params, then evaluate OOS ---
rf_fit_ts <- ranger(
  x = as.data.frame(X_train),
  y = as.numeric(y_train),
  num.trees = 500,                        # stronger final forest
  mtry = best$mtry,
  min.node.size = best$min.node.size,
  sample.fraction = 0.8,
  importance = "permutation",
  num.threads = max(1, parallel::detectCores() - 1)
)

y_hat_oos <- predict(rf_fit_ts, data = as.data.frame(X_oos))$predictions
rf_rmse <- rmse(y_oos, y_hat_oos)
rf_mae  <- mean(abs(y_oos - y_hat_oos))
cat(sprintf("TS-CV tuned RF — OOS RMSE = %.4f | OOS MAE = %.4f\n", rf_rmse, rf_mae))

# quick insight: importance
imp <- rf_fit_ts$variable.importance
print(sort(imp, decreasing = TRUE)[1:20])

