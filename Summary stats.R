library(readr)
library(dplyr)
library(lubridate)
library(zoo)
library(ggplot2)
library(tidyr)

raw_file <- read_csv("2025-09-MD.csv")

# extract transformation code
tcode_row <- raw_file[1, ]

data_raw  <- raw_file[-1, ]

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


summary(data)

# Sd & missing values
sapply(data %>% select(-sasdate), sd, na.rm = TRUE)
sapply(data %>% select(-sasdate), function(x) sum(is.na(x)))



ggplot(data, aes(x = sasdate, y = INDPRO_growth)) +
  geom_line(color = "steelblue") +
  geom_smooth(method = "loess", span = 0.2, color = "red", se = FALSE) +
  labs(title = "Industrial Production Growth Over Time",
       x = "Date", y = "Monthly Growth (%)") +
  theme_minimal()



predictors <- c("BAA_AAA", "HOUST", "T10Y3M", "UMCSENTx") 

data_long <- data %>%
  select(sasdate, all_of(predictors)) %>%
  pivot_longer(cols = -sasdate, names_to = "variable", values_to = "value")

ggplot(data_long, aes(x = sasdate, y = value, color = variable)) +
  geom_line() +
  facet_wrap(~variable, scales = "free_y", ncol = 1) +
  labs(title = "Selected Macroeconomic Predictors Over Time",
       x = "Date", y = "Value") +
  theme_minimal() +
  theme(legend.position = "none")



ggplot(data, aes(x = INDPRO_growth)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Monthly Industrial Production Growth",
       x = "INDPRO Growth (%)", y = "Frequency") +
  theme_minimal()



ggplot(data, aes(y = INDPRO_growth, x = "")) +
  geom_boxplot(fill = "steelblue", outlier.color = "red") +
  labs(title = "Boxplot of Monthly Industrial Production Growth",
       y = "INDPRO Growth (%)") +
  theme_minimal()


library(dplyr)

key_vars <- c("INDPRO_growth", "BAA_AAA", "HOUST", "T10Y3M", "UMCSENTx")

data %>%
  select(all_of(key_vars)) %>%
  summary()


desc_stats <- data %>%
  select(all_of(key_vars)) %>%
  summarise(across(everything(), list(
    Mean = ~ mean(.x, na.rm = TRUE),
    SD   = ~ sd(.x, na.rm = TRUE),
    Min  = ~ min(.x, na.rm = TRUE),
    Max  = ~ max(.x, na.rm = TRUE)
  ))) %>%
  pivot_longer(everything(),
               names_to = c("Variable", ".value"),
               names_sep = "_")

desc_stats

library(dplyr)

# outliers 
flag_outliers <- function(x, na.rm = TRUE) {
  Q1 <- quantile(x, 0.25, na.rm = na.rm)
  Q3 <- quantile(x, 0.75, na.rm = na.rm)
  IQR_val <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR_val
  upper <- Q3 + 1.5 * IQR_val
  x < lower | x > upper
}

# BAA_AAA outliers
baa_outliers <- data %>%
  filter(flag_outliers(BAA_AAA)) %>%
  select(sasdate, BAA_AAA)

# HOUST outliers
hou_outliers <- data %>%
  filter(flag_outliers(HOUST)) %>%
  select(sasdate, HOUST)


cat("Number of BAA_AAA outliers:", nrow(baa_outliers), "\n")
print(baa_outliers)

cat("Number of HOUST outliers:", nrow(hou_outliers), "\n")
print(hou_outliers)

library(ggplot2)
library(forecast)
library(zoo)

data_ts <- ts(data$INDPRO_growth, start = c(1978, 1), frequency = 12)


autoplot(data_ts) + 
  ggtitle("Industrial Production Growth") +
  ylab("Monthly Growth (%)") +
  xlab("Year")


decomp <- stl(data_ts, s.window = "periodic")
autoplot(decomp) + ggtitle("STL Decomposition of INDPRO Growth")


seasonal <- decomp$time.series[, "seasonal"]
autoplot(seasonal) + 
  ggtitle("Seasonal Component of INDPRO Growth") +
  ylab("Seasonal Effect") +
  xlab("Year")


acf(data_ts, main = "ACF of INDPRO Growth")
pacf(data_ts, main = "PACF of INDPRO Growth")

