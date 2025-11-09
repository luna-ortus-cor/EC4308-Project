# set working directory
#setwd("C:/Users/Grace/Desktop/EC4308 Project")

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

# na_share[na_share > 0]
# CMRMTSPLx        PERMIT      PERMITNE      PERMITMW       PERMITS       PERMITW        ACOGNO 
# 0.00125       0.01500       0.01500       0.01500       0.01500       0.01500       0.49750 
# ANDENOx       BUSINVx      ISRATIOx      NONREVSL        CONSPI S&P div yield  S&P PE ratio 
# 0.13625       0.00125       0.00125       0.00125       0.00125       0.00250       0.00125 
# CP3Mx     COMPAPFFx TWEXAFEGSMTHx      UMCSENTx   DTCOLNVHFNM      DTCTHFNM       VIXCLSx 
# 0.00125       0.00125       0.21000       0.19250       0.00125       0.00125       0.05250 

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
  filter(!is.na(INDPRO_L4))   

# keep only lagged predictors & target variable
model_data <- data_lagged %>%
  select(sasdate, INDPRO_growth, ends_with("_L1"), ends_with("_L2"), ends_with("_L3"), ends_with("_L4"))

#scale all predictors (including lags of INDPRO_growth but not INDPRO_growth)
# columns to scale
cols_to_scale <- setdiff(names(model_data), c("sasdate", "INDPRO_growth"))

model_data_scaled <- model_data %>%
  mutate(across(all_of(cols_to_scale), ~ as.numeric(scale(.x))))

#check for missing values
model_data_scaled %>% 
  filter(if_any(everything(), is.na))

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
library(glmnet)

set.seed(1)

alphas    <- c(0, 0.05, 0.1)
val_len   <- 60                      # size of rolling validation zone
n_train   <- nrow(X_train)

t_start   <- n_train - val_len - 1   # -1 so that t+1 exists
t_end     <- n_train - 1

cv_errors <- numeric(length(alphas))

for (i in seq_along(alphas)) {
  a <- alphas[i]
  mse_vec <- c()
  
  for (t in t_start:t_end) {
    # blocked folds inside each expanding fit to respect time order
    kfolds <- 5
    foldid <- cut(seq_len(t), breaks = kfolds, labels = FALSE)
    
    cvfit <- cv.glmnet(
      x = X_train[1:t, , drop = FALSE],
      y = y_train[1:t],
      alpha = a,
      foldid = foldid,         # contiguous blocks
      nfolds = kfolds,
      standardize = FALSE,     # you already standardized
      family = "gaussian"
    )
    
    pred <- as.numeric(predict(cvfit,
                               newx = X_train[t+1, , drop = FALSE],
                               s = "lambda.min"))
    mse_vec <- c(mse_vec, (y_train[t+1] - pred)^2)
  }
  
  cv_errors[i] <- mean(mse_vec)
}

alpha_star <- alphas[which.min(cv_errors)]
alpha_star
cv_errors

###############################################################################
 # create block folds for alpha selection

K <- 5   
foldid <- cut(seq_len(nrow(X_train)), breaks = K, labels = FALSE)

# run ridge cv
ridge_cv <- cv.glmnet(
  X_train, y_train,
  alpha = 0,              # <-- Ridge
  foldid = foldid,        # <-- blocked CV (no randomness)
  standardize = FALSE     # <-- we already scaled ourselves
)

# extract lambda
lambda_min  <- ridge_cv$lambda.min   # lambda that minimizes CV error
lambda_1se  <- ridge_cv$lambda.1se   # simpler (more shrinkage), but nearly as good

#lambda_min = 1.934788 (use this for oos prediction)
#lambda_1se = 193.4788

###############################################################################
# generate oos forecasts with ridge (α = 0, λ = 1.93)

alpha_star  <- 0
lambda_star <- lambda_min 

# Initialize vector to store OOS forecasts
fc_oos <- numeric(length(oos_index))

# Loop through each OOS point
for (i in seq_along(oos_index)) {
  
  # Determine how many observations are available at this forecast point
  t <- oos_index[i] - 1  # last in-sample index for this forecast
  
  # Training sample expands each iteration (recursive window)
  X_sub <- X[1:t, , drop = FALSE]
  y_sub <- y[1:t]
  
  # Fit ridge model with selected lambda
  fit_ridge <- glmnet(
    X_sub, y_sub,
    alpha = alpha_star,
    lambda = lambda_star,
    standardize = FALSE
  )
  # Forecast next observation
  fc_oos[i] <- as.numeric(predict(fit_ridge, X[oos_index[i], , drop = FALSE]))
}

# evaluate forecast accuracy
# actual OOS target values
y_real <- y[oos_index]

# RMSE
rmse_ridge <- sqrt(mean((y_real - fc_oos)^2))
rmse_ridge

# MAE
mae_ridge <- mean(abs(y_real - fc_oos))
mae_ridge

fcst_ridge <- data.frame(
  date  = model_data_scaled$sasdate[oos_index],
  y_act = y[oos_index],
  y_hat = fc_oos
)

train_mean <- mean(y_train, na.rm = TRUE)

fcst_ridge <- fcst_ridge %>%
  mutate(y_rw = dplyr::lag(y_act, 1),
         year = format(date, "%Y"))

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

yearly_r2_ridge <- fcst_ridge %>%
  group_by(year) %>%
  summarise(
    n_obs   = n(),
    R2_mean = subsample_r2_mean(cur_data()),
    R2_rw   = subsample_r2_rw(cur_data())
  ) %>%
  arrange(year)

yearly_r2_ridge

