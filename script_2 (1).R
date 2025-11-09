library(readr)
library(dplyr)
library(zoo)
library(tidyr)
library(MASS)
library(car)

raw_data <- read.csv("2025-09-MD.csv")

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

#test correlation 
# Identify predictor columns (exclude dependent variable and date)
predictor_cols <- setdiff(names(data), c("sasdate", "INDPRO", "INDPRO_growth"))

# Compute correlation with 1-month ahead INDPRO_growth
lead_cor <- sapply(data[, predictor_cols], function(x) {
  cor(dplyr::lead(data$INDPRO_growth, 1), x, use = "pairwise.complete.obs")
})

# Pick top 10 predictors by absolute correlation
top_predictors <- names(sort(abs(lead_cor), decreasing = TRUE))[1:10]
top_predictors

# Remove previous lag columns first
data <- data %>% select(-matches("_lag"))

# -----------------------------
# Create lags for predictors
# -----------------------------
max_lag <- 4
predictor_cols <- setdiff(names(data), c("sasdate", "INDPRO", "INDPRO_growth"))

# Compute correlation of each lag with 1-month-ahead INDPRO_growth
best_lag_per_var <- sapply(predictor_cols, function(var) {
  cor_lags <- sapply(1:max_lag, function(lag_i) {
    cor(lead(data$INDPRO_growth, 1), lag(data[[var]], lag_i), use = "pairwise.complete.obs")
  })
  which.max(abs(cor_lags))  # choose lag with highest absolute correlation
})

# -----------------------------
# Build lagged dataset with only best lags
# -----------------------------
for (var in predictor_cols) {
  best_lag <- best_lag_per_var[var]
  lag_name <- paste0(var, "_lag", best_lag)
  data[[lag_name]] <- dplyr::lag(data[[var]], best_lag)
}

# create AR lags for INDPRO_growth
for (i in 1:max_lag) {
  data[[paste0("INDPRO_growth_lag", i)]] <- dplyr::lag(data$INDPRO_growth, i)
}

# Remove rows with NAs from lagging
data_lagged <- drop_na(data)

# -----------------------------
# Build full model with best lags
# -----------------------------
lagged_cols <- grep("_lag", names(data_lagged), value = TRUE)
full_formula <- as.formula(paste("INDPRO_growth ~", paste(lagged_cols, collapse = " + ")))
full_model <- lm(full_formula, data = data_lagged)


# -----------------------------
# Stepwise selection using AIC/BIC
# -----------------------------
n <- nrow(data_lagged)

# Stepwise selection using AIC
step_model_aic <- stepAIC(full_model, direction = "both", trace = TRUE)

# Stepwise selection using BIC
step_model_bic <- stepAIC(full_model, direction = "both", k = log(n), trace = TRUE)

### Check 

summary(step_model_aic)
summary(step_model_bic)
vif(step_model_aic)  # check multicollinearity for AIC-selected model
vif(step_model_bic)
