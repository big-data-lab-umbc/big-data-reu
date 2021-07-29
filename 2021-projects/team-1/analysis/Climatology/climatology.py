#Climatology: This code can be used to calculate climatologies and use them to find anomalies for any variable in our monthly data sets.
# Note: This example code uses the varable of sea ice extent and created anomalies for the year 2012, however one can edit# the code to analyze another variable such as T2m and observe any other year in the data by changing the file read in and# variables parsed from the data.
# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in cvs file through pandas. File is located in our monthly data directory.
monthly_ice = pd.read_csv("/umbc/xfs1/cybertrn/reu2021/team1/research/analysis/monthly_data/monthly_ice.csv")
# Parce data to extract sea ice extent for years 1979-2018  and reshape the data to fit our data format.
monthly_ice = np.array(monthly_ice.iloc[:, 1])
monthly_ice = monthly_ice.reshape(480, 448, 304)

# Take the average of each months sea ice extent.
avg_ice = np.nanmean(monthly_ice, axis = (1,2))

# Create date values to be added to the data in order to be able to group values by months.
date = pd.date_range(start='1979-01-10', end='2018-12-10', periods=480)
# Combine Date values with average sea ice values as a pandas dataframe.
df = pd.DataFrame({'datetime': date,
                                 'ice': avg_ice,})

# Convert Date column into a datetime object with date and time formatting.
df.index = pd.to_datetime(df['datetime'],format='%Y-%m-%d %H:%M:%S')
# Group all months from 1979-2018 by their months and take the average of these months values in order to create a
# climatology holding 12 values to represent the total average of sea ice extent for specific months over the 40 year time span.
month_avg = df.groupby(df['datetime'].dt.month).mean().values

# We can then extract any specific years monthly sea ice values by finding the range in whcih a year exists (2012 for
# example begins at the 397th row and ends at the 409th row of data).
ice2012 = df[397:409].values
ice2012 = ice2012[:,1]

# Anomaly 
# Create an array holding zeros that is 12 values in length. We then run a for loop with the same length subtracting our
# calculated climatology values for monthly sea ice from a specific year (example is 2012) to create anomalies for each
# month.
anom12 = np.zeros(ice2012.size)
x=0
for x in range(len(ice2012)):
    anom12[x] = ice2012[x]-month_avg[x]

# Plot
# We then plot the calculated anomalies as the y-axis with dates by month on the x-axis and save the plot as a .png.
plt.plot(anom12)
plt.xlabel('Month')
plt.ylabel('Sea Ice Extent')
plt.title('Monthly Sea Ice Extent Anomalies for 2012')
positions = (0,2,4,6,8,10)
labels = ("Jan", "Mar", "May", "Jul", "Sep", "Nov")
plt.xticks(positions, labels)
plt.savefig('anom12.png', bbox_inches='tight')

