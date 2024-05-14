# PACKAGES AND DIRECTORY ---- 
rm(list = ls())
filepath = rstudioapi::getSourceEditorContext()$path
dirpath = dirname(rstudioapi::getSourceEditorContext()$path)
setwd(dirpath)
dirpath

library(rnassqs)
library(rnassqs)
library(ncdf4)
library(dplyr)
library(tidyr)
library(stringr)
library(data.table)
library(ncdf4)
library(stringr)
library(FedData)
library(terra)
library(stars)
library(tigris)
library(lubridate)
library(sf)
library(daymetr)
library(httr)
library(tictoc)
library(purrr)
library(sf)
library(ggplot2)
library(gridExtra)
library(raster)
library(gganimate)
library(transformr)
library(reshape2)
library(gridGraphics)
library(tseries)
if (!require("raster")) install.packages("raster", dependencies = TRUE)
if (!require("sf")) install.packages("sf")



# Loading the data and processing it for future plots -----

# in this section the data is manipulated not only for analysis but 
# many are used also in the plot section.

# daily_illinois = read.csv("writen data and checkpoints/daily_Weather_Data_illinois.csv")
weekly_illinois = read.csv("writen data and checkpoints/weekly_Weather_Data_ILLINOIS.csv")
# daily_iowa = read.csv("writen data and checkpoints/daily_Weather_Data_IOWA.csv")
weekly_iowa = read.csv("writen data and checkpoints/weekly_Weather_Data_iowa.csv")
field_obs_eda =read.csv("writen data and checkpoints/field_obs.csv")
field_obs= read.csv("writen data and checkpoints/field_obs.csv")
field_obs = field_obs %>% filter(year> 1979) 

# Bounding Boxes Per County ----
# This part explains: 
# 1. How boundign boxes are computes 
# 2. Gathers longitude and latitude data used later as a feature 
# 3. Long and latt are gathered from the central point of the bounding box

counties <- tigris::counties(state = c("IA", "IL"), resolution = "20m", cb = TRUE) %>%
  st_as_sf()%>%
  st_transform(crs = 4326)

# Exapmple of box
x = st_bbox(dplyr::filter(counties, NAME == "Jones" )) # one example of a county
print(x)

# Generate bounding box geometries
counties$bbox_geometry <- lapply(st_geometry(counties), st_bbox) %>%
  lapply(st_as_sfc) %>%  
  do.call(c, .)  # from geometry coulumn  to bounding box wiht 4 cordinates

# Convert list back to an sf column
counties$bbox_geometry <- st_sfc(counties$bbox_geometry, crs = st_crs(counties))

# Calculate the center of bounding boxes
counties <- counties %>%
  mutate(center = lapply(st_geometry(.), function(g) {
    bbox <- st_bbox(g)
    list(lon = (bbox$xmin + bbox$xmax) / 2, lat = (bbox$ymin + bbox$ymax) / 2)
  })) %>%
  mutate(center = st_sfc(lapply(center, function(coord) {
    st_point(c(coord[[1]], coord[[2]]), dim = "XY")
  }), crs = st_crs(counties)))

# Check the result
Long_lat_data <- counties %>%
  st_set_geometry(NULL) %>%  # Remove geometry to simplify DataFrame operations
  transmute(
    CountyName = NAME,
    State_Name = STATE_NAME,
    Longitude = st_coordinates(center)[,1],  # Extract longitude from the 'center' POINT
    Latitude = st_coordinates(center)[,2]    # Extract latitude from the 'center' POINT
  )

# View the resulting DataFrame
print(Long_lat_data) 


### Processing NC files------
# Below is the pipeline that transforms NC files of 1985 to a data-frame
# This is crucial since it is used Iowa later 
# to generate heat maps of the area and weather features
# thus allowing us to cross check the download process.


base_dir  <- "C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r/weathernc" 
year <- "1985"
variable <- "tmax"
pattern <- sprintf(".*_%s_%s\\.nc$", variable, year)

# List all files that match this pattern for the given year and variable across all counties
nc_files <- list.files(path = base_dir, pattern = pattern, full.names = TRUE)
print(nc_files)

# List all files that match this pattern for the given year and variable across all counties
nc_files <- list.files(path = base_dir, pattern = pattern, full.names = TRUE)
print(nc_files)

process_nc_file_to_df <- function(nc_file) {
  # Load the NetCDF file
  temp_data <- rast(nc_file)  # Load the raster data
  
  # If the CRS isn't set or appears incorrect, manually set it
  if (is.na(crs(temp_data)) || crs(temp_data) == "") {
    crs(temp_data) <- "+proj=lcc +lat_0=41.878 +lon_0=-93.0977 +lat_1=41 +lat_2=43 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
  }
  
  # Project the data to geographic coordinates (WGS84)
  temp_data_ll <- project(temp_data, "+proj=longlat +datum=WGS84")
  
  # Convert the reprojected raster to a data frame for plotting
  temp_df <- as.data.frame(temp_data_ll, xy = TRUE)
  names(temp_df) <- c("Longitude", "Latitude", paste("Day", 1:nlyr(temp_data_ll), sep = "_"))
  
  # Convert to long format for animation
  temp_df_long <- reshape2::melt(temp_df, id.vars = c("Longitude", "Latitude"), variable.name = "Time", value.name = "Temperature")
  
  # Add a column for the source file (county)
  temp_df_long$County <- basename(nc_file)
  
  return(temp_df_long)
}

# listing and applying function 
nc_files <- list.files(path = base_dir, pattern = pattern, full.names = TRUE)
all_data_1985_plot <- do.call(rbind, lapply(nc_files, process_nc_file_to_df))
all_data_filtered_day1 <- all_data_1985_plot[all_data_1985_plot$Time == "Day_1", ]

gc()

# Creating a date column for all data to use in Gifs and plots
# This comes as a consequence of the creation of the data set itself 
# from processing nc files function in the download step the date misses and 
# its important to have for plots so here is created... 

# Convert 'Time' to numeric days (stripping off 'Day_' and converting to integer)
all_data_1985_plot$DayNumber <- as.integer(gsub("Day_", "", all_data_1985_plot$Time))

# Define the start date (first day of Week 18 in 1985) # INFORMATION FROM SUMMARY OF FINAL DATA THAT MIN = WEEK 18 
start_date <- as.Date("1985-01-01") + days(which(weekdays(as.Date("1985-01-01") + 0:364) == "Monday")[18] - 1)

all_data_1985_plot$Date <- start_date + days(all_data_1985_plot$DayNumber - 1)

gc()
# Modeling Field Observations ----- 

## Plant Progress ---- 

### The chosen strategy for progress : 

# ''' Two ideas : IDEA 1 WINS SINCE IT FULLY DESCRIBES THE DATA 
# 1. Create a stage column an overlapping stage column and a percentage column
#  and fill them like this :
# 
# week planted blooming     stage      overlapping      percentage      percentage overlap
# 14    20        0       planted     none                20              0
# 15    80        20      planted     bloming             80              20
# 16    100       40      planted     bloming             100             40
# 17     0        60      bloming     none                60              0
# 
# 
# 2. Run two regressions on the data first regression 
# 
# Stage = alpha + week + state + county
# we let it fully fit the data and use to create stage columns
# percentage = alpha + week + state + county + stage 
# percentage column 
# 
# Strategy one is used it is derived directly from the data and seems logical. 
# '''

# -------------> Data arranged in python due to run time efficiency # <<<>>>>
field_obs= read.csv("writen data and checkpoints/Field_obs_prog_fix2.csv")

field_obs <- field_obs %>%
  mutate(
    overlapping = if_else(overlapping == "none", NA_character_, overlapping)
  )

## Plant Condition ----

###  Condition contained 5 columns with percentages amounting to 100 percent so 
# week pct.excellent pct.good ..... pct.very poor
# 15      20          30                10       = total plus other classes 100
# I decided to represent this using an variable called condition class which is 
# computed as the weighted average of the condition scores and outputs a score 
# leaning or not towards a class , than this scores are divided into quartiles 
# and divided into 4 classes so that multidisciplinary is not present in the model
# finally the non presence of such score is considered as none.

# Weighted average condition score <---

field_obs <- field_obs %>%
  mutate(
    condition_score = (PCT.EXCELLENT * 5 + PCT.GOOD * 4 + PCT.FAIR * 3 + PCT.POOR * 2 + PCT.VERY.POOR * 1) / 
      (PCT.EXCELLENT + PCT.GOOD + PCT.FAIR + PCT.POOR + PCT.VERY.POOR),
    condition_score = ifelse(is.nan(condition_score), NA, condition_score)  # Handle cases where all PCT values are NA
  )

# Illustration of score distribution 
ggplot(field_obs, aes(x = week , y = condition_score))+
  geom_point()

# Dividing into quantiles 
quantiles <- quantile(field_obs$condition_score, probs = c(0.2, 0.4, 0.6, 0.8), na.rm = TRUE)

# Put into class one left out 

field_obs <- field_obs %>%
  mutate(
    condition_class = case_when(
      is.na(condition_score) ~ NA_character_,        # Explicitly handle NA values
      condition_score <= quantiles[1] ~ "VERY.POOR", # Lowest 20%
      condition_score <= quantiles[2] ~ "POOR",      # Next 20%
      condition_score <= quantiles[3] ~ "FAIR",      # Middle 20%
      condition_score <= quantiles[4] ~ "GOOD",      # Next 20%
      TRUE ~ "EXCELLENT"                             # Top 20%
    )
  )


# Check the distribution NA counts
table(field_obs$condition_class, useNA = "ifany")

# Plot distrubution (EDA POTENTIAL USE)
ggplot(field_obs, aes(x = condition_score, fill = condition_class)) +
  geom_histogram(bins = 30, alpha = 0.6) +
  labs(title = "Distribution of Condition Scores by Class",
       x = "Condition Score",
       y = "Count",
       fill = "Condition Class") +
  theme_minimal()

field_obs = field_obs %>%
  dplyr::select(-c("PCT.EXCELLENT","PCT.FAIR","PCT.GOOD" ,"PCT.POOR",
            "PCT.VERY.POOR"))%>%
  mutate(county_name = if_else(
      state_name %in% c("ILLINOIS", "IOWA"),
      paste(county_name, tolower(state_name), sep = "_"),
      county_name))

# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# Checkpoint condition and progress ---- 
# write.csv(field_obs, file = "writen data and checkpoints/field_obs_final.csv", row.names = FALSE)
# field_obs= read.csv("writen data and checkpoints/field_obs_final.csv")
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>

# Cleaning environment...
all_objects <- ls()

# Find names that contain the word 'plot'
objects_to_remove <- grep("plot", all_objects, value = TRUE, ignore.case = TRUE)
if (length(objects_to_remove) > 0) {
  rm(list = objects_to_remove, envir = .GlobalEnv)
}

graphics.off()
gc()
print("Enviroment clean moving to next step")

# <<<>>>><<<>>>><<<>>>>Crucial Step<<<>>>><<<>>>><<<>>>><<<>>>>
# Yield de-trend ----- 

yield_trend = field_obs %>%
  dplyr::select(c(YIELD , year, county_name, state_name))%>%
  distinct(YIELD, county_name, year)%>%
  pivot_wider(
    names_from = county_name,    
    values_from = YIELD,         
    values_fill = list(YIELD = NA)  
    )%>%
  arrange(year, decreasing = FALSE)%>%
    dplyr::select(-c("OTHER COUNTIES_iowa","OTHER COUNTIES_illinois"))

detrend_data <- function(yield_data, window_size = 20) {
  n <- length(yield_data)
  detrended_yields <- rep(NA, n) 
  
  # Ensure we start calculating where we have a full window
  for (i in seq_len(n - window_size)) {
    start_index <- i
    end_index <- i + window_size - 1  # End at the window_size to include exactly window years of data
    
    # Extract the local data
    local_data <- yield_data[start_index:end_index]
    
    # Continue only if there are sufficient non-NA data points in the window
    if (length(na.omit(local_data)) < 2) {
      # If there are no valid data points, skip this iteration
      next
    } else {
      time_seq <- seq_along(local_data)
      model <- lm(local_data ~ time_seq, na.action = na.exclude)
      # Predict for the next year after the window ends
      if (!is.na(yield_data[end_index + 1])) {
        predicted_yield <- predict(model, newdata = data.frame(time_seq = window_size + 1))
        detrended_yields[end_index + 1] <- yield_data[end_index + 1] - predicted_yield
      }
      # If the actual yield is NA, it remains NA in the detrended data
    }
  }
  
  return(detrended_yields)
}

yield_trend_c= yield_trend

print("Detrending procces Start")
county_columns <- names(yield_trend)[-1]  # exclude 'year'
yield_trend[ , county_columns] <- map_df(yield_trend[ , county_columns], detrend_data)
yield_trend3 = yield_trend # Hold out coppies for future
yield_trend = yield_trend %>% filter(year >= 1980)
yield_trend2 = yield_trend 

print("Detrending procces End")

# One county analysis of de-trending <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# Example visualization after de-trending (EDA POTENTIAL)
ggplot(yield_trend, aes(x = year, y = CLAY_iowa)) +
  geom_line() +
  geom_point() +
  labs(title = "Detrended Yield Over Years for BUENA VISTA, Iowa",
       x = "Year", y = "Detrended Yield") +
  theme_minimal()

# Autocorrelation plot of example
acf_results <- acf(yield_trend$CLAY_iowa, lag.max = 20, plot = TRUE)

# Stationarity test using Augmented Dickey-Fuller test
adf_test_result <- adf.test(yield_trend$CLAY_iowa, alternative = "stationary")
print(adf_test_result)
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>


detrended_data = yield_trend %>%
  pivot_longer(
    cols = -year,  # Exclude the year column from the pivot (keep it as it is)
    names_to = "county_name",  # Name of the new column for county names
    values_to = "YIELD"        # Name of the new column for yield values
  )%>%
  na.omit()


X= field_obs %>% dplyr::select(-c("YIELD"))
detrended_data = detrended_data %>%
  left_join(X , by = c("county_name", "year"))%>%
  filter(year>=1980)
rm(X)


# Dealing with Missing Values ----
# Data is not missing at random where data is missing is considered informative
# since it is missing by construct for example there is no planting at week 
# 40 which is normal since that is harvesting stage. 

detrended_data$condition_class[is.na(detrended_data$condition_class)] <- "NONE"
detrended_data$overlapping[is.na(detrended_data$overlapping)] <- "NONE"
detrended_data$condition_score[is.na(detrended_data$condition_score)]<- 0
anyNA(detrended_data)

# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# EDA Potential : Yield distribution before and after de-trending
# residuals should have a mean of 0 and normally distributed around the mean 

detrended_datax <- detrended_data %>%
  dplyr::select(county_name, year, YIELD) %>%
  distinct(county_name, year, .keep_all = TRUE)


field_obsx= field_obs%>%
  dplyr::select(county_name, year, YIELD) %>%
  distinct(county_name, year, .keep_all = TRUE)%>%
  filter(year>=1980)

p1 <- ggplot(detrended_datax, aes(x = YIELD)) +
  geom_histogram(stat = "bin", fill = "steelblue", color = "black") +
  ggtitle("Detrended Data Histogram") +
  theme_minimal()

p2 <- ggplot(field_obsx, aes(x = YIELD)) +
  geom_histogram(stat = "bin", fill = "firebrick", color = "black") +
  ggtitle("Field Observations Histogram") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 2)
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# EDA Potential: distribution of yield per 12 randomly choosen counties.
data_field = detrended_data
x = data_field %>%
  dplyr::select(c(year , YIELD, county_name))%>%
  distinct(year, YIELD , county_name)%>%
  pivot_wider(names_from = county_name , values_from = YIELD)

county_columns <- names(x)[-1]
samples = sample(county_columns, 12)

temp = x %>% dplyr:: select(c(year, all_of(samples)))
str(temp)

temp <- temp %>%
  pivot_longer(cols = -year, names_to = "county", values_to = "value")

p <- ggplot(temp, aes(x = value)) +
  geom_histogram(bins = 20, fill = "blue", color = "black") +  # Adjust number of bins as necessary
  facet_wrap(~county, scales = "free_y") +  # Free y scales if count scale varies significantly
  theme_minimal() +
  labs(x = "Value", y = "Frequency", title = "Distribution of Values by County") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Improving x axis label readability

# Print the plot
print(p)
rm(x , p , temp , acf_results , adf_test_result)
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>



### Merging weather with field observations -----
# For now weekly weather observations are chosen but later maybe there can 
# be a shift to daily observations.... 

weekly_illinois$state_name = "ILLINOIS"
weekly_iowa$state_name = "IOWA"
combined_data = bind_rows(weekly_illinois, weekly_iowa)

combined_data = combined_data %>%
  mutate(county_name = tolower(county_name))%>%
  mutate(
      county_name = if_else(
        state_name %in% c("ILLINOIS", "IOWA"),
        paste(county_name, tolower(state_name), sep = "_"),
        county_name)
      )

data_field = data_field %>%
  mutate(county_name = tolower(county_name))%>%
  mutate(year = as.integer(year),
         week = as.integer(week))

unique(combined_data$county_name)

final_data <- inner_join(combined_data, data_field,
                         by = c("county_name", "year", "week","state_name"))

final_data = final_data %>%
  dplyr::select(-c("country_code"))

names(final_data)
final_data = final_data %>%
  dplyr::select(c("year" , "county_name" ,"state_name","AREA.PLANTED"
                  ,"PRODUCTION","AREA.HARVESTED","YIELD", "week", 
                  "stage","percentage" ,"overlapping","percentage_overlap", 
                  "condition_class" ,"condition_score","weekly_dayl",       
                  "weekly_prcp","weekly_srad", "weekly_swe","weekly_tmax",       
                 "weekly_tmin","weekly_vp"))

Long_lat_data <- Long_lat_data %>%
  mutate(
    county_name = tolower(paste(CountyName, State_Name, sep = "_"))  # Convert to lower case and concatenate
  )%>%
  dplyr::select(-c(State_Name, CountyName))

final_data <- final_data %>%
  left_join(Long_lat_data, by = "county_name")



# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# Chekpoint 
write.csv(final_data, file = "C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r/writen data and checkpoints/df_finalv1.csv", row.names = FALSE)
summary(final_data)

### Missing value analysis -----
# This is non relevant here and quick stop SKIP NOT IMPORTANT 
# left as a section just for future potential if needed 
data_summary_plot <- final_data %>%
  group_by(county_name, year) %>%
  summarize(weeks_of_data = n_distinct(week), .groups = 'drop')
summary(final_data)





# Model provisional---- 

# This was build just to test the different split and conclude that this
# should be carried forward in python where a forecasting approach will be chosen
# this allowed me to see different inherent problems : 
# 1. Geographical split captures only the systematic portion of the variance
# or it naturally assume that counties are the same. Would be more interesting
# if we would investigate the similarities between counties. But for our objective
# which is to predict the next yield temporal split is a must because in a geo 
# split we have seen the full range of the year in one county and thus it is 
# inaccurate to predict next year when in training we have such temporal data
# we aim to incorporate such geo nature by longitude and latitude variables
# 2. There is a problem with the length of the prediction , we must predict
# one number at the end for one year using all of the weekly data not weekly
# predictions. 


library(randomForest)

set.seed(123)  # for reproducibility

unique_counties <- unique(data_field$county_name)

# Shuffle and split the unique counties into training, validation, and testing sets
set.seed(123) # For reproducibility
train_counties <- sample(unique_counties, size = round(0.6 * length(unique_counties)))
valid_test_counties <- setdiff(unique_counties, train_counties)
validation_counties <- sample(valid_test_counties, size = round(0.5 * length(valid_test_counties)))
test_counties <- setdiff(valid_test_counties, validation_counties)

# Create the actual training, validation, and testing sets based on the county splits
train_set <- data_field %>% filter(county_name %in% train_counties)
validation_set <- data_field %>% filter(county_name %in% validation_counties)
test_set <- data_field %>% filter(county_name %in% test_counties)

# Data leakage test : 
unique_train_counties = unique(train_set$county_name) 
unique_test_counties= unique(test_set$county_name)
common_counties <- intersect(unique_train_counties, unique_test_counties)
length(common_counties)



# Train Random Forest Model on the training sethttp://127.0.0.1:14183/graphics/24ca6984-6607-453c-8233-a7d1cbc1753a.png
model <- randomForest(YIELD ~ ., data = train_set, ntree = 5)

# Check model performance on the validation set
predictions_validation <- predict(model, validation_set)

# Assuming 'validation_set$YIELD' holds the actual yield values
actuals_validation <- validation_set$YIELD

# Calculate Mean Squared Error (MSE) for validation set
mse_validation <- mean((predictions_validation - actuals_validation)^2)
print(paste("Validation Mean Squared Error: ", mse_validation))

# Calculate Mean Absolute Error (MAE) for validation set
mae_validation <- mean(abs(predictions_validation - actuals_validation))
print(paste("Validation Mean Absolute Error: ", mae_validation))

# Calculate R-squared for validation set
ss_res_validation <- sum((predictions_validation - actuals_validation)^2)
ss_tot_validation <- sum((actuals_validation - mean(actuals_validation))^2)
r_squared_validation <- 1 - (ss_res_validation / ss_tot_validation)
print(paste("Validation R-squared: ", r_squared_validation))

# Predicting test set 
predictions_test <- predict(model, test_set)
actuals_test <- test_set$YIELD

# Test set performance 

ss_res_test <- sum((predictions_test - actuals_test)^2)
ss_tot_test <- sum((actuals_test - mean(actuals_test))^2)
r_squared_test <- 1 - (ss_res_test / ss_tot_test)

# Feature Importance (based on trained model)
importance <- importance(model)
varImpPlot(model)  # Plotting feature importance
library(ggplot2)

# Data frame for plotting
plot_data <- data.frame(Actual = actuals_test, Predicted = predictions_test)
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "blue") +  # Set point color to blue
  geom_smooth(method = lm, color = "red", formula = y ~ x) +  # Regression line in red
  ggtitle("Actual vs. Predicted Yields") +
  xlab("Actual Yield") +
  ylab("Predicted Yield") +
  theme_minimal()
head(plot_data)

plot_data_long <- data.frame(Value = c(actuals_test, predictions_test),
                             Type = rep(c("Actual", "Predicted"), each = length(actuals_test)))

# Creating the plot
ggplot(plot_data_long, aes(x = Value, color = Type)) +
  geom_density(alpha = 0.5) +  # Using density plot to show distribution
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue")) +  # Set colors for actual and predicted
  ggtitle("Actual vs. Predicted Yields") +
  xlab("Yield") +
  ylab("Density") +
  theme_minimal() +
  theme(legend.title = element_blank())  


# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# EDA analysis ---- 
names_int = c("year","YIELD" , "PCT.PLANTED"  , "PCT.EMERGED","PCT.BLOOMING",
              "PCT.SETTING.PODS","PCT.COLORING" ,"PCT.DROPPING.LEAVES", "PCT.HARVESTED",     
              "PCT.EXCELLENT","PCT.FAIR", "PCT.GOOD", "PCT.POOR" ,"PCT.VERY.POOR", "county_name", "week")

names_int2 = c("year","YIELD" , "PCT.PLANTED"  , "PCT.EMERGED","PCT.BLOOMING",
               "PCT.SETTING.PODS","PCT.COLORING" ,"PCT.DROPPING.LEAVES", "PCT.HARVESTED",     
               "county_name", "week")

test = field_obs_eda%>%
  dplyr::select(names_int2)%>%
  filter(year >= 1980)



# Representing progress Visually

long_data <- test %>%
  dplyr::select(county_name, week, starts_with("PCT.")) %>%
  pivot_longer(cols = starts_with("PCT."), names_to = "stage", values_to = "percentage") %>%
  filter(percentage > 0) %>%
  group_by(county_name, stage) %>%
  mutate(start_week = min(week[percentage > 0]), 
         end_week = max(week[percentage > 0])) %>%
  ungroup()

set.seed(123) 
selected_counties <- sample(unique(long_data$county_name), 15)

filtered_data <- long_data %>% 
  filter(county_name %in% selected_counties)%>%
  group_by(county_name, stage, week) %>%
  summarise(percentage = max(percentage), .groups = "drop")

# Function to filter out weeks after the first occurrence of the maximum percentage is reached
filter_after_max <- function(data) {
  max_value <- max(data$percentage)  # Find the maximum percentage in the data
  first_max_index <- which(data$percentage == max_value)[1]  # Get the index of the first occurrence of this max value
  data <- data[1:first_max_index, ]  # Keep only up to the first occurrence of max
  return(data)
}

# Apply the function to each group of county and stage
cleaned_data <- filtered_data %>%
  group_by(county_name, stage) %>%
  group_modify(~ filter_after_max(.x)) %>%
  ungroup()

filtered_data = cleaned_data

# Create a single faceted plot for the selected counties
combined_plot <- ggplot(filtered_data, aes(x = as.factor(week), y = percentage, fill = stage)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ tolower(county_name), scales = "fixed", nrow = 5) + # Adjust nrow for desired layout
  scale_fill_brewer(palette = "Set3",
                    breaks = c("PCT.PLANTED" , "PCT.EMERGED" , "PCT.BLOOMING", 
                               "PCT.SETTING.PODS" , "PCT.COLORING", "PCT.DROPPING.LEAVES", "PCT.HARVESTED")) +
  labs(x = "Week of the Year", y = "Percentage Complete", fill = "Stages") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 7),
        strip.text.y = element_text(angle = 0),
        legend.text = element_text(size = 8),  # Smaller text size for legend
        legend.key.size = unit(0.5, "cm"))

# Print the combined plot
print(combined_plot)


# The area of study : 
p1 <- ggplot(data = counties) +
  geom_sf(aes(fill = NAME)) +  # Fill by county name
  ggtitle("Counties without Bounding Boxes") +
  theme_minimal() +
  theme(legend.position = "none")  # Remove legend to clean up the plot

p2 <- ggplot(data = counties) +
  geom_sf(aes(fill = NAME)) +  # Original county shapes
  geom_sf(data = counties, aes(geometry = bbox_geometry), color = "red", fill = NA, lwd = 0.7) +  # Bounding boxes
  ggtitle("Counties with Bounding Boxes") +
  theme_minimal() +
  theme(legend.position = "none")

grid.arrange(p1, p2, ncol = 2)

ggplot(data = counties) +
  geom_sf() +  # This uses the geometry column by default
  theme_minimal() +
  labs(title = "Map of Counties in IA and IL")

# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>

# Missing value plots # NONE IN THIS CONFIGURATION WE HAVE 
# ggplot(data_summary_plot, aes(x = year, y = weeks_of_data, fill = county_name)) +
#   geom_bar(stat = "identity", position = "dodge") +
#   labs(title = "Number of Distinct Weeks of Data per County and Year",
#        x = "Year",
#        y = "Distinct Weeks of Data") +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Improve x-axis label readability


# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>

# NEEDS DATA FROM DAILY LEVEL WHICH IS FILTERED AT LINE 148 
# One day heat map of Iowa counties maximum temperature 

# plot1 <- ggplot(all_data_filtered_day1, aes(x = Longitude, y = Latitude)) +
#   geom_point(aes(color = Temperature), size = 1) + 
#   scale_color_viridis_c() + 
#   coord_fixed(ratio = 1) +
#   labs(title = "Point Plot of Temperature Data",
#        x = "Longitude", y = "Latitude", color = "Temperature") +
#   xlim(c(-96.64, -90.09)) +  # Set x limits based on Longitude summary
#   ylim(c(40.33, 43.54)) +    # Set y limits based on Latitude summary
#   theme_minimal()
# 
# print(plot1)
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>


### Seasonality of temperature for both counties  -----
# This serves as a critical example to show the nature of the data and 
# it is proof of the data being correct 

daily_iowa = read.csv("writen data and checkpoints/daily_Weather_Data_IOWA.csv")

# Convert date to Date type
daily_iowa$date <- as.Date(daily_iowa$date)

# Create month and year variables for monthly aggregation
daily_iowa <- daily_iowa %>%
  mutate(
    Month = month(date, label = TRUE, abbr = TRUE),  # Extract month as an abbreviated character
    year = year(date),  # Extract year
    Week = week(date)   # Extract week number
  )

# Aggregate data to monthly average temperature
monthly_temp_plot <- daily_iowa %>%
  group_by(year, Month) %>%
  summarise(Avg_Temp = mean((tmax + tmin) / 2, na.rm = TRUE)) %>%
  ungroup()

# Aggregate data to weekly average temperature
weekly_temp_plot <- daily_iowa %>%
  group_by(year, Week) %>%
  summarise(Avg_Temp = mean((tmax + tmin) / 2, na.rm = TRUE)) %>%
  ungroup()


ggplot(monthly_temp_plot, aes(x = Month, y = Avg_Temp, group = year, color = as.factor(year))) +
  geom_line() +
  labs(title = "Monthly Average Temperature by Year",
       x = "Month",
       y = "Average Temperature (°C)",
       color = "Year") +
  theme_minimal()


ggplot(weekly_temp_plot, aes(x = Week, y = Avg_Temp, group = year, color = as.factor(year))) +
  geom_line() +
  labs(title = "Weekly Average Temperature by Year",
       x = "Week of the Year",
       y = "Average Temperature (°C)",
       color = "Year") +
  theme_minimal()


if (!inherits(daily_iowa$date, "Date")) {
  daily_iowa$date <- as.Date(daily_iowa$date, format = "%Y-%m-%d")  # Adjust the format as per your data
}

ggplot(daily_iowa, aes(x = date, y = tmax)) +
  geom_line() +  # Add a line plot
  scale_x_date(date_breaks = "2 years",  # Set breaks every two years
               date_labels = "%Y",  # Format labels to show only the year
               limits = as.Date(c("1980-01-01", "2024-12-31"))) +  
  labs(title = "Daily Maximum Temperature from 1980 to 2024",
       x = "Date",
       y = "Temperature (°C)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
## De-trent For illustration only but not actual de trend -----

data_county = yield_trend_c %>% dplyr::select(c(year,CLAY_iowa))

p1 = ggplot(data = data_county , aes(x = year, y = CLAY_iowa)) +
  geom_line() +  
  geom_point() + 
  geom_smooth(method = "lm", se = FALSE, color = "blue") + 
  theme_minimal() + 
  scale_x_continuous(breaks = seq(min(data_county$year), max(data_county$year), by = 5))+
  labs(title = "Yield Over The Years Pre De-trending Procedure",
       x = "Year",
       y = "Yield") +
  theme(plot.title = element_text(hjust = 0.5))

acf(data_county$CLAY_iowa , lag.max = 20, main = "Autocorrelation Function")

acf_results <- acf(data_county$CLAY_iowa, lag.max = 20, plot = FALSE)
acf_data <- data.frame(
  Lag = as.numeric(acf_results$lag[-1]),  # remove the first element to exclude lag 0
  ACF = acf_results$acf[-1]
)

# Plot using ggplot2
p2 =  ggplot(acf_data, aes(x = Lag, y = ACF)) +
  geom_bar(stat = "identity", fill = "gray", width = 0.1 ) +  
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", size = 0.8) +
  geom_hline(yintercept = 0.2, linetype = "dashed", color = "steelblue", size = 1)+
  geom_hline(yintercept = -0.2, linetype = "dashed", color = "steelblue", size = 1)+
  scale_x_continuous(breaks = seq(0,20))+
  labs(title = "Autocorrelation of Yields Pre De-trending Procedure (Excluding Lag 0)",
       x = "Lag", y = "Autocorrelation") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))


data_county$CLAY_iowa_detrended = yield_trend3$CLAY_iowa
data_countyx = data_county %>% filter(year >= 1980)
p3 = ggplot(data = data_countyx, aes(x = year, y =CLAY_iowa_detrended )) +
  geom_line() +  
  geom_point() + 
  geom_smooth(method = "lm", se = FALSE, color = "blue") + 
  theme_minimal() + 
  scale_x_continuous(breaks = seq(1980, max(data_county$year), by = 5))+
  labs(title = "Yield Over The Years Post De-trending Procedure",
       x = "Year",
       y = "Yield") +
  theme(plot.title = element_text(hjust = 0.5))
p3
# Autocorrelation after de-trend 

acf_results <- acf(data_countyx$CLAY_iowa_detrended , lag.max = 20, plot =TRUE)
acf_data <- data.frame(
  Lag = as.numeric(acf_results$lag[-1]),  # remove the first element to exclude lag 0
  ACF = acf_results$acf[-1]
)
# Plot using ggplot2
p4 =  ggplot(acf_data, aes(x = Lag, y = ACF)) +
  geom_bar(stat = "identity", fill = "gray", width = 0.1 ) +  # Use geom_col() if you prefer
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", size = 0.8) +
  geom_hline(yintercept = 0.2, linetype = "dashed", color = "steelblue", size = 1)+
  geom_hline(yintercept = -0.2, linetype = "dashed", color = "steelblue", size = 1)+
  scale_x_continuous(breaks = seq(0,20))+
  labs(title = "Autocorrelation of Yields Pre De-trending Procedure (Excluding Lag 0)",
       x = "Lag", y = "Autocorrelation") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))

p1 <- p1 + theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))
p2 <- p2 + theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))
p3 <- p3 + theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))
p4 <- p4 + theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))
grid.arrange(p1, p2, p3, p4, ncol = 2)


library(tseries)
adf_test_result <- adf.test(data_countyx$CLAY_iowa_detrended , alternative = "stationary")
print(adf_test_result )


# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>

### Heatmap for different days (ram intense for more than 3 days) ---- 
# gc()
# Correct the days vector
days = c("Day_1", "Day_2", "Day_125")

# Function to create plots
plot_days <- function(day_data) {
  ggplot(day_data, aes(x = Longitude, y = Latitude)) +
    geom_point(aes(color = Temperature), size = 1) +
    scale_color_viridis_c() +
    coord_fixed(ratio = 1) +
    labs(title = paste("Point Plot of Temperature Data for", unique(day_data$Time)),
         x = "Longitude", y = "Latitude", color = "Temperature") +
    theme_minimal()
}

# Create a list to store the plots
plot_list <- list()

# Loop to create a plot for each day and store it
for (d in days) {
  day_data = all_data_1985_plot[all_data_1985_plot$Time == d, ]
  plot_list[[d]] <- plot_days(day_data)
}

# Arrange all plots in a grid
do.call(grid.arrange, c(plot_list, ncol = 2))



### Animation does not render because of ram problems ----- 
# gc()
# 
# all_data_1985_plot1= all_data_1985_plot
# all_data_1985_plot1$month = month(all_data_1985_plot1$Date)
# all_data_1985_plot1 = all_data_1985_plot1 %>% filter(month < 7)
# anim_plot <- ggplot(all_data_1985_plot1, aes(x = Longitude, y = Latitude, fill = Temperature)) +
#   geom_tile() +  # Use geom_tile or geom_point depending on your data layout and visual preference
#   scale_fill_viridis_c(option = "C") +  # Color scale
#   labs(title = 'Temperature on Day {frame_time}', x = 'Longitude', y = 'Latitude') +
#   coord_fixed(ratio = 1) +  # Keep aspect ratio
#   theme_minimal() +
#   transition_time(Date) +  
#   ease_aes('linear') 
# 
# anim <- animate(anim_plot, duration = 2, fps = 10, width = 800, height = 600, renderer = gifski_renderer())
# anim_save("temperature_animation.gif", animation = anim)

# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
# <<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>><<<>>>>
### EDA Finish ----
print('EDA Finished removing data and cleaning enviroment')


# List all objects in the environment
all_objects <- ls()
gc()
# Find names that contain the word 'plot'
objects_to_remove <- grep("plot", all_objects, value = TRUE, ignore.case = TRUE)

# Remove these objects
if (length(objects_to_remove) > 0) {
  rm(list = objects_to_remove, envir = .GlobalEnv)
}

graphics.off()
print(ls())
rm(x , p1 , p2 , variable , filepath , 
   all_data_filtered_day1 , all_objects,
   year, start_date , objects_to_remove , pattern, nc_files )
gc()
print("Enviroment clean moving to next step")











## Detrend EDA POTENTIAL -----
detrend_data_with_fit <- function(yield_data, window_size = 20) {
  n <- length(yield_data)
  detrended_yields <- numeric(n)
  fitted_yields <- numeric(n)  # To store fitted values
  
  for (i in seq_len(n)) {
    start_index <- max(1, i - window_size + 1)  # Correct the window's start index
    end_index <- i  # Use data up to the current index
    
    if (end_index - start_index + 1 < window_size) {
      # Not enough data points to form a full window
      detrended_yields[i] <- NA
      fitted_yields[i] <- NA
      next
    }
    
    local_data <- yield_data[start_index:end_index]
    if (length(na.omit(local_data)) < 2) {
      # If there are fewer than 2 valid data points, skip this iteration
      detrended_yields[i] <- NA
      fitted_yields[i] <- NA
      next
    }
    
    time_seq <- seq(start_index, end_index)
    model <- lm(local_data ~ time_seq, na.action = na.exclude)
    predicted_yield <- predict(model, newdata = data.frame(time_seq = i))
    detrended_yields[i] <- yield_data[i] - predicted_yield
    fitted_yields[i] <- predicted_yield  # Store the fitted values
  }
  
  return(list(residuals = detrended_yields, fits = fitted_yields))
}


# Assuming 'yield_trend' is your dataset and 'CARROLL_iowa' is the column of interest
results <- detrend_data_with_fit(yield_trend_c$`BUENA VISTA_iowa`, window_size = 20)
yield_data <- yield_trend_c$`BUENA VISTA_iowa`
time_points <- seq_along(yield_data)  # Assuming a simple sequence for plotting


# Assuming 'yield_data' starts from a specific year, for example, 1980
start_year <- 1951
years <- seq(start_year, by = 1, length.out = length(yield_data))

# Using the modified 'years' vector for plotting
ggplot() +
  geom_point(aes(x = years, y = yield_data), color = "blue") +  # Actual data points
  geom_line(aes(x = years, y = results$fits), color = "red") +  # Fitted trend line
  labs(title = "Fitting of Rolling Regression Model WINDOW SIZE 20",
       x = "Year",
       y = "Yield") +
  scale_x_continuous(breaks = seq(start_year, max(years), by = 4)) +  # Set x-axis breaks every 5 years
  theme_minimal()


# Clean the environment ----

all_objects <- ls()

# Find names that contain the word 'plot'
objects_to_remove <- grep("plot", all_objects, value = TRUE, ignore.case = TRUE)
if (length(objects_to_remove) > 0) {
  rm(list = objects_to_remove, envir = .GlobalEnv)
}

graphics.off()
gc()
print("Enviroment clean moving to next step")

