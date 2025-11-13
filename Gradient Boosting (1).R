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

###############################################################################
# hyperparameter tuning

set.seed(1)
# create rolling time-series CV folds
n <- nrow(X_train)
k <- 5  # number of folds
fold_size <- floor(n / (k+1))

folds <- list()
for(i in 1:k){
  train_end <- fold_size * i
  valid_start <- train_end + 1
  valid_end <- min(train_end + fold_size, n)
  folds[[i]] <- valid_start:valid_end
}

# # define search grid
# etas <- c(0.05, 0.1)
# depths <- c(2, 3, 4)
# min_child <- c(3, 5)
# subsamps <- c(0.6, 0.8)
# colsamps <- c(0.3, 0.5)

depths <- c(3, 4)          # allow deeper trees
min_child <- c(3, 5)       # allow smaller leaf size
subsamps <- c(0.6, 0.8)  # allow more variance
etas <- c(0.03, 0.07)      # allow deeper learning
colsamps <- c(0.3, 0.5)

# rolling cv grid search
library(xgboost)

dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)

results <- data.frame()

for (eta in etas){
  for (depth in depths){
    for (mc in min_child){
      for (ss in subsamps){
        for (cs in colsamps){
          
          params <- list(
            objective = "reg:squarederror",
            eta = eta,
            max_depth = depth,
            min_child_weight = mc,
            subsample = ss,
            colsample_bytree = cs,
            eval_metric = "rmse"
          )
          
          cv <- xgb.cv(
            params = params,
            data = dtrain,
            nrounds = 500,
            folds = folds,          # time-series folds
            early_stopping_rounds = 20,
            verbose = FALSE
          )
          
          results <- rbind(results, data.frame(
            eta = eta,
            max_depth = depth,
            min_child_weight = mc,
            subsample = ss,
            colsample_bytree = cs,
            best_rmse = cv$evaluation_log$test_rmse_mean[cv$best_iteration],
            best_nrounds = cv$best_iteration
          ))
        }
      }
    }
  }
}

best <- results[which.min(results$best_rmse), ]
best


# fit oos
params_best <- list(
  objective = "reg:squarederror",
  eta = best$eta,
  max_depth = best$max_depth,
  min_child_weight = best$min_child_weight,
  subsample = best$subsample,
  colsample_bytree = best$colsample_bytree,
  eval_metric = "rmse"
)

xgb_fit <- xgb.train(
  params = params_best,
  data = dtrain,
  nrounds = best$best_nrounds
)

y_hat_oos <- predict(xgb_fit, as.matrix(X_oos))

rmse <- sqrt(mean((y_oos - y_hat_oos)^2))
mae  <- mean(abs(y_oos - y_hat_oos))
rmse; mae

R2_in  <- 1 - sum((y_train - predict(xgb_fit, as.matrix(X_train)))^2) / sum((y_train - mean(y_train))^2)
R2_oos <- 1 - sum((y_oos   - y_hat_oos)^2)/ sum((y_oos   - mean(y_train))^2)

R2_in
R2_oos
