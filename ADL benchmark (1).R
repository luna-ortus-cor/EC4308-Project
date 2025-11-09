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

df <- data %>%
  arrange(sasdate) %>%
  mutate(
    BAA_AAA = BAA - AAA,
    T10Y3M = T10YFFM - TB3MS
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

# choose by BIC (more conservative) — or AIC if you prefer
p_star <- ics$p[which.min(ics$BIC)]
p_star


# prepare lagged indicators
# add lagged Xs (1..4) to the AR frame
add_x_lags <- function(dat, var) {
  if (!(var %in% names(df))) return(dat)
  dat %>%
    mutate(
      !!paste0(var,"_L1") := lag_k(df[[var]],1),
      !!paste0(var,"_L2") := lag_k(df[[var]],2),
      !!paste0(var,"_L3") := lag_k(df[[var]],3),
      !!paste0(var,"_L4") := lag_k(df[[var]],4)
    )
}

adl_base <- df %>%
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
  add_x_lags("BAA_AAA") %>%
  add_x_lags("HOUST") %>%
  add_x_lags("T10Y3M")%>%
  add_x_lags("UMCSENTx")%>%
  # drop rows until max lag available (ensures SAME sample for all p <= 6)
  filter(!is.na(y_L6))


adl_train <- adl_base %>% filter(date <= train_end)

# base AR(p*) IC on same training sample
rhs_ar <- paste0("y_L", 1:p_star, collapse = " + ")
fit_ar <- lm(as.formula(paste0("y ~ ", rhs_ar)), data = adl_train)
base_ic <- get_ic(fit_ar, "BIC")

# candidate indicators present in your data
cands <- c("BAA_AAA","HOUST","T10Y3M", "UMCSENTx")
cands <- cands[cands %in% names(df)]  # drop missing ones

# evaluate each indicator alone with k=1..4 lags
x_lag_max <- 4
best_by_var <- list()
for (v in cands) {
  scores <- c()
  for (k in 1:x_lag_max) {
    x_terms <- paste0(v, "_L", 1:k, collapse = " + ")
    fml <- as.formula(paste0("y ~ ", rhs_ar, " + ", x_terms))
    fit <- lm(fml, data = adl_train)
    scores[k] <- get_ic(fit, "BIC")
  }
  k_star_v <- which.min(scores)
  if (scores[k_star_v] + 1e-8 < base_ic) {
    best_by_var[[v]] <- list(k = k_star_v, IC = scores[k_star_v])
  }
}
best_by_var


