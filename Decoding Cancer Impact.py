import pandas as pd
import os

# Load the dataset (replace this with your actual dataset path if needed)
file_path = r'C:\Users\IoannisZografakis-Re\Downloads\cancer.csv'  # Update this to the correct path of your dataset
cancer_data = pd.read_csv(file_path)

# Apply the filters based on the given conditions:
# - The 'age_adjusted_death_rate' must be greater than 100
# - The 'recent_trend' should be either 'stable' or 'rising'
filtered_data = cancer_data[
    (cancer_data['age_adjusted_death_rate'] > 100) &
    (cancer_data['recent_trend'].isin(['stable', 'rising']))
]

# Select the specified columns
selected_columns = [
    'county', 'fips', 'met_objective', 'age_adjusted_death_rate',
    'lower_confidence_interval_death_rate', 'upper_confidence_interval_death_rate',
    'average_deaths_per_year', 'recent_trend', 'recent_5year_trend_death_rates',
    'lower_confidence_interval_trend', 'upper_confidence_interval_trend'
]

# Filter the data based on selected columns
filtered_data = filtered_data[selected_columns]

# Specify the output path (make sure to update this to a valid path on your local system)
output_file_path = r'C:\path_to_save\filtered_cancer_research_data.csv'  # Update this to your desired save path

# Ensure the directory exists before saving the file
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Export the filtered data to a CSV file
filtered_data.to_csv(output_file_path, index=False)

print(f"Filtered data has been saved to: {output_file_path}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os



# Convert 'age_adjusted_death_rate' column to numeric and handle missing values
filtered_data['age_adjusted_death_rate'] = pd.to_numeric(filtered_data['age_adjusted_death_rate'], errors='coerce')

# Drop rows where 'age_adjusted_death_rate' is missing
filtered_data.dropna(subset=['age_adjusted_death_rate'], inplace=True)

# Calculate the average and standard deviation of 'age_adjusted_death_rate'
avg_death_rate = np.mean(filtered_data['age_adjusted_death_rate'])
std_death_rate = np.std(filtered_data['age_adjusted_death_rate'])

print(f"Average Age Adjusted Death Rate: {avg_death_rate:.2f}")
print(f"Standard Deviation of Death Rate: {std_death_rate:.2f}")

plt.figure(figsize=(10, 6))
sns.regplot(x='average_deaths_per_year', y='age_adjusted_death_rate', data=filtered_data, scatter_kws={'s': 50}, line_kws={"color":"red"})

# Add a horizontal line for the average death rate
plt.axhline(y=avg_death_rate, color='blue', linestyle='--', label=f'Average Death Rate: {avg_death_rate:.2f}')

# Customize the plot
plt.title('Scatter Plot of Average Deaths per Year vs Age Adjusted Death Rate')
plt.xlabel('Average Deaths per Year')
plt.ylabel('Age Adjusted Death Rate')
plt.legend()

# Display the plot
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_data['average_deaths_per_year'], filtered_data['age_adjusted_death_rate'])

# Print the slope, intercept, and R-squared value
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.2f}")
print(f"R-squared Value: {r_value**2:.4f}")


# Step 1: Data Preprocessing
# Convert 'age_adjusted_death_rate' and 'recent_5year_trend_death_rates' columns to numeric and handle missing values
filtered_data['age_adjusted_death_rate'] = pd.to_numeric(filtered_data['age_adjusted_death_rate'], errors='coerce')
filtered_data['recent_5year_trend_death_rates'] = pd.to_numeric(filtered_data['recent_5year_trend_death_rates'], errors='coerce')

# Drop rows where 'age_adjusted_death_rate' or 'recent_5year_trend_death_rates' are missing
filtered_data.dropna(subset=['age_adjusted_death_rate', 'recent_5year_trend_death_rates'], inplace=True)

# Step 2: Perform Calculations with NumPy
# Calculate the mean, median, and standard deviation of 'age_adjusted_death_rate'
mean_death_rate = np.mean(filtered_data['age_adjusted_death_rate'])
median_death_rate = np.median(filtered_data['age_adjusted_death_rate'])
std_death_rate = np.std(filtered_data['age_adjusted_death_rate'])

# Step 3: Print the Calculations
mean_death_rate, median_death_rate, std_death_rate


# Step 4: Create a Combined Histogram and KDE Plot

plt.figure(figsize=(12, 6))

# Histogram and KDE for 'age_adjusted_death_rate'
sns.histplot(filtered_data['age_adjusted_death_rate'], kde=True, color='skyblue', label='Age Adjusted Death Rate', bins=20)

# Histogram and KDE for 'recent_5year_trend_death_rates'
sns.histplot(filtered_data['recent_5year_trend_death_rates'], kde=True, color='orange', label='Recent 5-Year Trend Death Rates', bins=20)

# Step 5: Highlight Key Metrics on the Plot
# Add vertical lines for the mean and median of 'age_adjusted_death_rate'
plt.axvline(mean_death_rate, color='blue', linestyle='--', label=f'Mean Death Rate: {mean_death_rate:.2f}')
plt.axvline(median_death_rate, color='green', linestyle='--', label=f'Median Death Rate: {median_death_rate:.2f}')

# Step 6: Customize the Plot
plt.title('Histogram and KDE Plot of Age Adjusted Death Rate and Recent 5-Year Trend Death Rates')
plt.xlabel('Death Rates')
plt.ylabel('Frequency')
plt.legend()

# Display the plot
plt.show()



import plotly.express as px

# Step 4: Create a 3D Scatter Plot using Plotly
fig = px.scatter_3d(
    filtered_data,
    x='average_deaths_per_year',
    y='age_adjusted_death_rate',
    z='recent_5year_trend_death_rates',
    color='recent_5year_trend_death_rates',
    color_continuous_scale='Viridis',
    title='3D Scatter Plot of Average Deaths per Year, Age Adjusted Death Rate, and Recent 5-Year Trend',
    labels={
        'average_deaths_per_year': 'Average Deaths per Year',
        'age_adjusted_death_rate': 'Age Adjusted Death Rate',
        'recent_5year_trend_death_rates': 'Recent 5-Year Trend Death Rates'
    }
)

# Step 5: Customize and display the plot
fig.update_layout(scene = dict(
                    xaxis_title='Average Deaths per Year',
                    yaxis_title='Age Adjusted Death Rate',
                    zaxis_title='Recent 5-Year Trend Death Rates'
                ))

fig.show()


import plotly.express as px
import pandas as pd
import numpy as np

# Step 1: Data Preprocessing
# Convert 'age_adjusted_death_rate', 'recent_5year_trend_death_rates', and 'average_deaths_per_year' to numeric
filtered_data['average_deaths_per_year'] = pd.to_numeric(filtered_data['average_deaths_per_year'], errors='coerce')

# Drop rows where any of the specified columns are missing
filtered_data.dropna(subset=['age_adjusted_death_rate', 'recent_5year_trend_death_rates', 'average_deaths_per_year'], inplace=True)

# Step 2: Perform Calculations with NumPy
# Calculate the maximum, minimum, and range of 'age_adjusted_death_rate'
max_death_rate = np.max(filtered_data['age_adjusted_death_rate'])
min_death_rate = np.min(filtered_data['age_adjusted_death_rate'])
range_death_rate = max_death_rate - min_death_rate

# Step 4: Create a 3D Scatter Plot using Plotly
fig = px.scatter_3d(
    filtered_data,
    x='average_deaths_per_year',
    y='age_adjusted_death_rate',
    z='recent_5year_trend_death_rates',
    color='recent_5year_trend_death_rates',
    color_continuous_scale='Viridis',
    title='3D Scatter Plot of Average Deaths per Year, Age Adjusted Death Rate, and Recent 5-Year Trend',
    labels={
        'average_deaths_per_year': 'Average Deaths per Year',
        'age_adjusted_death_rate': 'Age Adjusted Death Rate',
        'recent_5year_trend_death_rates': 'Recent 5-Year Trend Death Rates'
    }
)

# Step 5: Customize and display the plot
fig.update_layout(scene = dict(
                    xaxis_title='Average Deaths per Year',
                    yaxis_title='Age Adjusted Death Rate',
                    zaxis_title='Recent 5-Year Trend Death Rates'
                ))

fig.show()
