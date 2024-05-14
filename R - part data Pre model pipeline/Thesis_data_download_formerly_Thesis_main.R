
# Setting directory ----
rm(list = ls())
filepath = rstudioapi::getSourceEditorContext()$path
dirpath = dirname(rstudioapi::getSourceEditorContext()$path)
setwd(dirpath)
dirpath

library(rnassqs)
library(tidyverse)
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


## API KEY ----
NASSQS_TOKEN <- "A873465F-173C-37A4-9FBC-76BB5C316984"

nassqs_auth(key = NASSQS_TOKEN) # Authentication 

## Processing county data -----
process_ag_data <- function(params) {
  # Fetch data using the provided parameters
  data <- nassqs(params)
  
  # Process area harvested or planted
  df_area_harvester <- data %>%
    filter(statisticcat_desc == "AREA HARVESTED" | statisticcat_desc == "AREA PLANTED") %>%
    filter(county_name != "OTHER (COMBINED) COUNTIES") %>%
    pivot_wider(names_from = statisticcat_desc, values_from = Value,
                id_cols = c(state_name, year, county_name, country_code))
  
  # Process production
  df_production <- data %>%
    filter(statisticcat_desc == "PRODUCTION") %>%
    filter(county_name != "OTHER (COMBINED) COUNTIES") %>%
    pivot_wider(names_from = statisticcat_desc, values_from = Value,
                id_cols = c(state_name, year, county_name, country_code))
  
  # Process yield
  df_yield <- data %>%
    filter(statisticcat_desc == "YIELD") %>%
    filter(county_name != "OTHER (COMBINED) COUNTIES") %>%
    pivot_wider(names_from = statisticcat_desc, values_from = Value,
                id_cols = c(state_name, year, county_name, country_code))
  
  # Combine all the data
  combined_data <- df_yield %>%
    left_join(df_production, by = c("year", "county_name", "country_code", "state_name")) %>%
    left_join(df_area_harvester, by = c("year", "county_name", "country_code", "state_name"))
  
  # Returning the combined data and the unique values for exploration
  return(combined_data)
}

params <- list(
  source_desc = "SURVEY",
  sector_desc = "CROPS",
  group_desc = "FIELD CROPS",
  commodity_desc = "SOYBEANS",
  agg_level_desc = "COUNTY",
  state_name = c("IOWA")
)



## Processing state level progress -----

# Main function to fetch, process, and prepare growth stages data for a given state
prepare_growth_stages_data <- function(state_name) {
  # Function to define parameters for fetching data
  paramsf <- function(paramet) {
    params <- list(
      source_desc = "SURVEY",
      sector_desc = "CROPS",
      group_desc = "FIELD CROPS",
      commodity_desc = "SOYBEANS",
      statisticcat_desc = paramet,
      agg_level_desc = "STATE",
      state_name = state_name
    )
    return(params)
  }
  
  # Fetch the data
  data <- nassqs(paramsf("PROGRESS"))
  
  # Process growth stages
  process_growth_stages <- function(data) {
    unique_stages <- unique(data$unit_desc)
    stage_data_list <- list()
    
    for (stage in unique_stages) {
      stage_data <- filter(data, unit_desc == stage)
      weeks <- sort(unique(as.numeric(sub("WEEK #", "", stage_data$reference_period_desc))))
      
      stage_weekly_data <- stage_data %>%
        mutate(week_number = as.numeric(sub("WEEK #", "", reference_period_desc))) %>%
        pivot_wider(names_from = week_number, values_from = Value,
                    names_prefix = "Week_", id_cols = c(state_name, year, unit_desc))
      
      stage_data_list[[stage]] <- stage_weekly_data
    }
    
    return(stage_data_list)
  }
  
  s <- process_growth_stages(data)
  
  transform_dataset <- function(df, stage_name) {
    # Convert from wide to long format
    df_long <- pivot_longer(df, 
                            cols = starts_with("Week_"),
                            names_prefix = "Week_",
                            names_to = "week",
                            values_to = "value") %>%
      mutate(week = as.numeric(week), # Convert week numbers to numeric
             stage = stage_name) %>%  # Add a column for the growth stage
      select(-unit_desc) # Remove the unit_desc column if it's redundant now
    
    return(df_long)
  }
  
  # Transform each dataset
  s_transformed <- lapply(names(s), function(stage_name) {
    df <- s[[stage_name]]
    transform_dataset(df, stage_name)
  })
  
  # Helper function to transform dataset
  transform_dataset <- function(df, stage_name) {
    df_long <- pivot_longer(df, 
                            cols = starts_with("Week_"),
                            names_prefix = "Week_",
                            names_to = "week",
                            values_to = "value") %>%
      mutate(week = as.numeric(week),
             stage = stage_name) %>%
      select(-unit_desc)
    
    return(df_long)
  }
  
  # Combine all transformed datasets into a single dataframe
  combined_df <- bind_rows(s_transformed)
  
  # Prepare the wide_df with desired structure and cleaning
  wide_df <- combined_df %>%
    unite("year_week", year, week, sep = "_") %>%
    pivot_wider(names_from = stage, values_from = value, values_fill = list(value = NA)) %>%
    separate(year_week, into = c("year", "week"), sep = "_") %>%
    arrange(desc(as.numeric(year)), as.numeric(week)) %>%
    select(state_name, year, week, 
           `PCT PLANTED`, `PCT EMERGED`, `PCT BLOOMING`, `PCT SETTING PODS`, 
           `PCT COLORING`, `PCT DROPPING LEAVES`, `PCT HARVESTED`) %>%
    filter(rowSums(!is.na(select(., `PCT PLANTED`:`PCT HARVESTED`))) > 0) %>%
    mutate(across(`PCT PLANTED`:`PCT HARVESTED`, ~replace_na(., 0)))
  
  return(wide_df)
}

# CONDITION -----


# Function to fetch and process condition data for a given state
fetch_process_condition_data <- function(state_name) {
  # Define parameters for fetching condition data
  paramsf <- function(paramet) {
    params <- list(
      source_desc = "SURVEY",
      sector_desc = "CROPS",
      group_desc = "FIELD CROPS",
      commodity_desc = "SOYBEANS",
      statisticcat_desc = paramet,
      agg_level_desc = "STATE",
      state_name = state_name
    )
    return(params)
  }
  
  # Fetch condition data
  data <- nassqs(paramsf("CONDITION"))
  
  # Process quality stages
  process_quality_stages <- function(data) {
    unique_stages <- unique(data$unit_desc)
    stage_data_list <- list()
    
    for (stage in unique_stages) {
      stage_data <- filter(data, unit_desc == stage)
      weeks <- sort(unique(as.numeric(sub("WEEK #", "", stage_data$reference_period_desc))))
      
      # Pivot data to have one column per week
      stage_weekly_data <- stage_data %>%
        mutate(week_number = as.numeric(sub("WEEK #", "", reference_period_desc))) %>%
        pivot_wider(names_from = week_number, values_from = Value,
                    names_prefix = "Week_", id_cols = c(state_name, year, unit_desc))
      
      # Add the processed data frame to the list
      stage_data_list[[stage]] <- stage_weekly_data
    }
    
    return(stage_data_list)
  }
  
  s <- process_quality_stages(data)
  
  # Helper function to transform dataset
  transform_dataset <- function(df, stage_name) {
    df_long <- pivot_longer(df, 
                            cols = starts_with("Week_"),
                            names_prefix = "Week_",
                            names_to = "week",
                            values_to = "value") %>%
      mutate(week = as.numeric(week),
             stage = stage_name) %>%
      select(-unit_desc)
    
    return(df_long)
  }
  
  # Transform each dataset
  s_transformed <- lapply(names(s), function(stage_name) {
    transform_dataset(s[[stage_name]], stage_name)
  })
  
  # Helper function to transform dataset
  transform_dataset <- function(df, stage_name) {
    df_long <- pivot_longer(df, 
                            cols = starts_with("Week_"),
                            names_prefix = "Week_",
                            names_to = "week",
                            values_to = "value") %>%
      mutate(week = as.numeric(week),
             stage = stage_name) %>%
      select(-unit_desc)
    
    return(df_long)
  }
  
  # Combine all transformed datasets into a single dataframe
  combined_df <- bind_rows(s_transformed)
  
  # Prepare the wide_df with desired structure and cleaning
  wide_df <- combined_df %>%
    unite("year_week", year, week, sep = "_") %>%
    pivot_wider(names_from = stage, values_from = value, values_fill = list(value = NA)) %>%
    separate(year_week, into = c("year", "week"), sep = "_") %>%
    arrange(desc(as.numeric(year)), as.numeric(week))%>% #  
    filter(rowSums(!is.na(select(., starts_with("PCT")))) > 0)
  
  return(wide_df)
}


# Getting and merging field observation data -----

# Main function to process and combine agricultural data for given states
process_and_combine_ag_data <- function(state_names) {
  combined_data_list <- lapply(state_names, function(state_name) {
    # Fetch and process county-based variables
    params <- list(
      source_desc = "SURVEY",
      sector_desc = "CROPS",
      group_desc = "FIELD CROPS",
      commodity_desc = "SOYBEANS",
      agg_level_desc = "COUNTY",
      state_name = state_name
    )
    data_county_level <- process_ag_data(params)
    
    # Fetch and prepare state-level progression data
    state_level_progression_data <- prepare_growth_stages_data(state_name)
    
    # Merge county data with state progression
    data_final <- data_county_level %>%
      left_join(state_level_progression_data, by = c("year", "state_name"))
    
    # Fetch condition of plants at the state level
    condition_plant_data <- fetch_process_condition_data(state_name)
    
    # Merge with condition data
    data_final <- data_final %>%
      arrange(desc(year)) %>%
      left_join(condition_plant_data, by = c("year", "state_name", "week"))
    
    return(data_final)
  })
  
  # Combine data for all specified states
  data_combined <- bind_rows(combined_data_list)
  return(data_combined)
}

# Define the states you want to process and run 
states_to_process <- c("IOWA", "ILLINOIS")

data_combined <- process_and_combine_ag_data(states_to_process)


# the end goal will be : 

# {
#   "Iowa": {
#     "2022": {
#       "Yield": 57.2,
#       "Production": 8208000,
#       "Progress": {
#         "Week 15": {
#           "Condition" : ["PCT PLANTED": "50","PCT_Emerged":"100" ....],
#           "Temperature": [{"date": "2022-04-10", "value": 15.2}, ...],
#           "Moisture": [...]
#         },
#         ...
#       }
#     }
#   }
# }





#WEATHER DATA DAYMET ----

# Function to construct Daymet URL for a variable
construct_daymet_url <- function(var, bbox, year, region) {
  paste0("https://thredds.daac.ornl.gov/thredds/ncss/grid/ornldaac/2129/daymet_v4_daily_",
         region, "_", var, "_", year, ".nc",
         "?var=", var,
         "&north=", bbox$ymax, "&west=", bbox$xmin, "&east=", bbox$xmax, "&south=", bbox$ymin,
         "&horizStride=1",
         "&time_start=", year, "-04-04T00:00:00Z", # start week 14 
         "&time_end=", year, "-12-31T12:00:00Z",
         "&timeStride=1&accept=netcdf")
}

# Function to download data for a single variable, year, and county
download_variable_data <- function(county_name, bbox, var, year, region, base_dir) {
  url <- construct_daymet_url(var, bbox, year, region)
  file_path <- file.path(base_dir, paste0(county_name, "_", var, "_", year, ".nc"))
  GET(url, write_disk(file_path, overwrite = TRUE))
  return(file_path)
}

# Function to process and extract averaged data from a single NetCDF file
process_nc_file <- function(file_path) {
  parts <- strsplit(basename(file_path), "_")[[1]]
  county_name <- parts[1]
  nc_data <- nc_open(file_path)
  var_name <- str_extract(file_path, "(tmax|tmin|srad|vp|swe|prcp|dayl)")
  time <- ncvar_get(nc_data, "time")
  var_data <- ncvar_get(nc_data, var_name)
  dates <- as.Date(time, origin = "1950-01-01")
  
  daily_averages <- apply(var_data, MARGIN = c(3), FUN = mean, na.rm = TRUE)
  
  nc_close(nc_data)
  
  data.frame(
    date = dates,
    variable = var_name,
    value = daily_averages,
    county_name = county_name
  )
}

# Main function to orchestrate downloading and processing workflow
orchestrate_download <- function(counties, variables, start_year, end_year, region, base_dir) {
  results <- list()
  counter = 0 
  for (county in counties$NAME) {
    bbox <- st_bbox(filter(counties, NAME == county))
    
    for (year in start_year:end_year) {
      for (var in variables) {
        file_path <- download_variable_data(county, bbox, var, year, region, base_dir)
        print(counter)
        counter = counter+1
      }
    }
  }
}

# DOWNLOADING WEATHER DATA IOWA-----


start_year <- 1980
end_year <- 2022
variables <- c("tmax", "tmin", "srad", "vp", "swe", "prcp", "dayl")
region <- "na"
base_dir  <- "C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r/weathernc" 

counties <- tigris::counties(state = c("IA"), resolution = "20m", cb = TRUE) %>%
  st_as_sf()

#:::MAJOR::: HOW COUNTIES BOXES ARE BEING GENERATED -------- 
x = st_bbox(filter(counties, NAME == "Jones" ))
x
plot(counties)

orchestrate_download(counties, variables, start_year, end_year, region, base_dir)

# FAIL SAFE FOR RE-INITATING WHERE IS LEFT IOWA -----

nc_files <- list.files(base_dir, full.names = TRUE, pattern = "\\.nc$")
county_names <- str_extract(nc_files, "(?<=weathernc/)[^_]+")
unique_county_names <- unique(county_names)
print(unique_county_names)

counties = counties %>%
  filter(!(NAME %in% unique_county_names))

# counties =  counties%>%
#   filter(NAME == "Delaware" )



# Downloading weather data illinois: ----

start_year <- 1980
end_year <- 2022
variables <- c("tmax", "tmin", "srad", "vp", "swe", "prcp", "dayl")
region <- "na"
base_dir <- "C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r/weathernc_Illinois"

counties <- tigris::counties(state = c("IL"), resolution = "20m", cb = TRUE) %>%
  st_as_sf()

orchestrate_download(counties, variables, start_year, end_year, region, base_dir)


# FAIL SAFE FOR RE-INITATING WHERE IS LEFT -----
nc_files <- list.files(base_dir, full.names = TRUE, pattern = "\\.nc$")
county_names <- str_extract(nc_files, "(?<=weathernc_Illinois/)[^_]+")
unique_county_names <- unique(county_names)
print(unique_county_names)
counties <- tigris::counties(state = c("IL"), resolution = "20m", cb = TRUE) %>%
  st_as_sf() 
counties = counties %>%
  filter(!(NAME %in% unique_county_names))
# If one county is left at the middle 
counties =  counties%>%
  filter(NAME == "Logan" )






# PROCESSING DOWNLOADED DATA ----- 


# Proccessing for Iowa and checkpoint ------ 
base_dir = "C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r/weathernc"
nc_files <- list.files(base_dir, full.names = TRUE, pattern = "\\.nc$")


process_in_chunks <- function(nc_files, batch_size = 100) {  # Adjust batch_size as needed
  combined_data_list <- list()  # Initialize an empty list to store combined data
  x= 0
  # Process files in batches
  for (i in seq(1, length(nc_files), by = batch_size)) {
    chunk_files <- nc_files[i:min(i + batch_size - 1, length(nc_files))]
    chunk_data_list <- lapply(chunk_files, process_nc_file)
    combined_data_list[[length(combined_data_list) + 1]] <- do.call(rbind, chunk_data_list)
    print(x)
    x = x+1
  }
  
  # Combine all batched data frames into a single data frame
  all_data <- do.call(rbind, combined_data_list)
  return(all_data)
}

# Assuming batch_size is defined or adjust it in the function call
all_data <- process_in_chunks(nc_files, 100)  # Adjust batch_size as per your system's memory capacity
final_dataset = all_data
summary(final_dataset)
wide_data <- final_dataset %>%
  pivot_wider(
    names_from = variable,  # Use 'variable' column to define new column names
    values_from = value  # Fill the new columns with values from the 'value' column
  )

final_dataset1 <- final_dataset %>%
  mutate(week = isoweek(date), year = year(date)) %>%
  group_by(county_name, year, week, variable) %>%
  summarize(weekly_avg = mean(value, na.rm = TRUE), .groups = 'drop') %>%
  pivot_wider(
    names_from = variable,
    values_from = weekly_avg,
    names_prefix = "weekly_"
  )
gc()



# Chek for date if correct 
date_ranges <- wide_data %>%
  gather(variable, value, -date, -county_name) %>% # Transform to long format
  group_by(county_name, variable) %>%
  summarize(start_date = min(date), end_date = max(date)) %>%
  ungroup()

daily_iowa = wide_data
weekly_iowa = final_dataset1
field_obs = data_combined

rm(all_data, data_combined, final_dataset ,final_dataset1 , wide_data)

# Writen data check point ---- 
write.csv(weekly_iowa , file = "weekly_Weather_Data_iowa.csv", row.names = FALSE)
write.csv(daily_iowa, file = "daily_Weather_Data_IOWA.csv", row.names = FALSE)
write.csv(data_combined, file = "field_obs.csv", row.names = FALSE)
write.csv(counties, file = "counties_boundary_and_names.csv", row.names = FALSE)



# Processing for illinois and check point ----- 

base_dir = "C:/Users/user/Desktop/Thesis Data science/Manegment_Data_r/weathernc_Illinois"
nc_files <- list.files(base_dir, full.names = TRUE, pattern = "\\.nc$")


process_in_chunks <- function(nc_files, batch_size = 100) {  # Adjust batch_size as needed
  combined_data_list <- list()  # Initialize an empty list to store combined data
  x= 0
  # Process files in batches
  for (i in seq(1, length(nc_files), by = batch_size)) {
    chunk_files <- nc_files[i:min(i + batch_size - 1, length(nc_files))]
    chunk_data_list <- lapply(chunk_files, process_nc_file)
    combined_data_list[[length(combined_data_list) + 1]] <- do.call(rbind, chunk_data_list)
    print(x)
    x = x+1
  }
  
  # Combine all batched data frames into a single data frame
  all_data <- do.call(rbind, combined_data_list)
  return(all_data)
}

# Assuming batch_size is defined or adjust it in the function call
all_data_ILL <- process_in_chunks(nc_files, 100)  # Adjust batch_size as per your system's memory capacity
final_dataset_ILL = all_data_ILL

wide_data <- final_dataset_ILL %>%
  pivot_wider(
    names_from = variable,  # Use 'variable' column to define new column names
    values_from = value  # Fill the new columns with values from the 'value' column
  )

final_dataset_ILL1 <- final_dataset_ILL %>%
  mutate(week = isoweek(date), year = year(date)) %>%
  group_by(county_name, year, week, variable) %>%
  summarize(weekly_avg = mean(value, na.rm = TRUE), .groups = 'drop') %>%
  pivot_wider(
    names_from = variable,
    values_from = weekly_avg,
    names_prefix = "weekly_"
  )
gc()

daily_illinois = wide_data
weekly_illinois = final_dataset_ILL1 

rm(wide_data, final_dataset_ILL1, all_data_ILL ,final_dataset_ILL, params)

# Writen data check point ---- 
write.csv(weekly_illinois, file = "weekly_Weather_Data_ILLINOIS.csv", row.names = FALSE)
write.csv(daily_illinois, file = "daily_Weather_Data_illinois.csv", row.names = FALSE)



gc()
