# wur_thesis_2024
This is the repository for the MSc thesis of Jialin He at Wageningen University & Research. 

## Usage Instructions

ams_data_explore.ipynb:
input: 
1. Households in Amsterdam PV and demand data.xlsx

output: 
1. ams_data_energy.csv
2. ams_data_pv.csv
3. ams_data_con.csv
4. ams_data_net.csv

ams_weather_explore.ipynb:

input: 
1. ams_weather_data.txt
   
output: 
1. ams_data_weather_processed.csv

ams_data_merge.ipynb:

input: 
1. ams_data_energy.csv
2. ams_data_weather_processed.csv

output: 
1. ams_data_merged_unprocessed.csv


ams_data_preprocessing.ipynb:

input: 
1. ams_data_merged_unprocessed.csv

output: 
1. ams_data_merged_preprocessed.csv
2. ams_pv_capacity_from_pv_profiles.csv

capacity_estimation.ipynb:

input:
1. ams_data_merged_preprocessed.csv
2. ams_pv_capacity_from_pv_profiles.csv

output: 
1. capacity_error_rate_part1.npy
2. capacity_error_rate_part2.npy

ams_point_regression.ipynb:

input: 
1. ams_data_merged_preprocessed.csv

output: 
1. point_regression_results


ams_pob_regression_lgb_two_parts12.ipynb:
ams_pob_regression_lgb_two_parts21.ipynb:

input:
1. ams_data_merged_preprocessed_part1.csv
2. ams_data_merged_preprocessed_part2.csv
3. capacity_error_rate_part1.npy
4. capacity_error_rate_part2.npy
   
output: