# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:50:08 2024

@author: ssath
"""
#%reset -f

import os
import scipy.io as sio
import datetime
import numpy as np
# import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
# from pathlib import Path


def find_rainfall_events(rainfall_data):
    events = []
    start_points = []
    end_points = []

    # Initialize variables for current event
    event_start = None
    event_end = None
    event_sum = 0
    event_count = 0

    # Iterate through the rainfall data
    for i, value in enumerate(rainfall_data):
        # Check if the current value is part of an event
        if value > 0:
            event_sum += value
            event_count += 1

            # If this is the start of a new event, record the start point
            if event_start is None:
                event_start = i

            # If this is the end of the array, record the end point of the event
            if i == len(rainfall_data) - 1:
                event_end = i
                events.append(event_sum / event_count)
                start_points.append(event_start)
                end_points.append(event_end)

        # If the current value is 0 and we were in an event, record the end point
        elif event_start is not None:
            event_end = i - 1
            events.append(event_sum / event_count)
            start_points.append(event_start)
            end_points.append(event_end)

            # Reset variables for the next event
            event_start = None
            event_end = None
            event_sum = 0
            event_count = 0

    return events, start_points, end_points

# file_path = r'C:\Users\ssath\Desktop\NRSC\Jodhpur\mrrave_Jodhpur_20220928'
file_path = r"C:\Users\ssath\Desktop\NRSC\Tezpur"
# file_path = r"C:\Users\ssath\Desktop\NRSC\testdata"
# file_path = r"C:\Users\ssath\Desktop\NRSC\Jodhpur\2023"
outpath=r"C:\Users\ssath\Desktop\NRSC\test1\Tezpur"
files = os.listdir(file_path)
files_ave = [file for file in files if file.endswith('.mat')]


location = 'Tezpur'
path_out=r'C:\Users\ssath\Desktop\NRSC\Tezpur'
day_data=[]
mean_withzero=[]
avg_date=[]
avg_height=[]
max_val_rr=[]
max_height=[]
max_date_time=[]
sum_data=[]
sum_date=[]
for reqfile in files_ave:
    # reqfile=files_ave[309]
    # datafile = os.path.join(file_path,reqfile)
    # filename3 = read_ave_file(datafile) 
    #filename3='mrrave_Jodhpur_20220608.mat'
    filename4=os.path.join(path_out,reqfile)
    
    mrrdata=sio.loadmat(filename4)
    mrr_height=mrrdata['dataheight']
    mrr_rainrate=mrrdata['datarainrate']
    # mrr_rainrate[mrr_rainrate<0.25]=np.nan
    mean_rainrate=[]
    time_rr=mrrdata['datadate']
    mrr_time=[datetime.datetime.fromtimestamp(ts) for ts in time_rr.flatten()]
    date2_exp=np.tile(np.array(mrr_time)[:,np.newaxis],(1,30))
    
    i=0
    for i in range(0,len(mrr_rainrate)):
        val_rr1 = mrr_rainrate[i,:]
        mean_rr=np.nanmean(val_rr1)
        
        if mean_rr <= 0.25 :
            mean_rainrate.append(np.nan)
        else:
            mean_rainrate.append(mean_rr)
        
        i+=1
        
    events, start_points, end_points = find_rainfall_events(mean_rainrate)
    # del i,val_rr1
    # for k in range(0,mrr_rainrate.shape[1]):       
    #     val_rr2 = mrr_rainrate[:,k] 
    
    #meltinglayer code mrr_rainrate    
    test=[]
    for k in range(0,len(start_points)):      
        val_rr2 = mrr_rainrate[start_points[k]:end_points[k],:]
        req=np.nanmean(val_rr2,axis=0)
        test.append(req)
    test = np.array(test)
    # days.append(test)
    #sum of tha mrr_data for height in 200m
    add=0
    try:
        val_r1=mrr_rainrate[:,0]
        add=sum(val_r1)/120
        sum_data.append(add)   
        sum_date.append(date2_exp[0,0])
    except IndexError:
        continue
    #max of mean of rainrate
    max_val=[]
    index_maxval=[]
    for m in range(0,len(test)):
        max_val1= max(test[m,:])
        max_val.append(max_val1)
        index_val=np.argmax(test[m,:])
        index_maxval.append(index_val*200)
    #mean value of max_val of rr
    max_val = np.array(max_val)
    #max_mean=np.nanmean(max_val)
    max_val = max(max_val) if max_val.size != 0 else np.nan
    max_val_rr.append(max_val)
    #mean value of index
    #index_mean=np.nanmean(index_maxval)
    index_maxval = np.array(index_maxval) 
    index_maxval = max(index_maxval) if index_maxval.size != 0 else 0
    max_height.append(index_maxval)

    max_time=[]
    max_time1=[]
    t=0
    for t in range(0,len(start_points)):
        time=date2_exp[start_points[t],:]
        time2=time
        time1=time[0]
        max_time.append(time1)
        max_time1.append(time2)
    max_time = np.array(max_time) 
    max_time1=np.array(max_time1)
    
    if max_time.size != 0:
        max_time= max(max_time)
        max_time1=max_time1[0,:]
    else:
        max_time  = date2_exp[t,0]
        max_time1=date2_exp[t,:]
    max_date_time.append(max_time)
    avg_date.append(max_time1) 
    
    #mean of all 30 values without 0
    mean_avgtest=[]
    avgheight=[]
    try:
        for test1 in range(test.shape[1]):
            avg_test=test[:,test1]        
            mean_test=np.nanmean(avg_test)
            
            if mean_test == 0:
                mean_test = np.nan
            mean_avgtest.append(mean_test)      
        mean_avgtest=np.array(mean_avgtest)
        avg_height.append(mrr_height[test1])
        day_data.append(mean_avgtest)
    except IndexError:
        sample=[]
        for z in range(0,30):
            sample.append(np.nan)
        sample=np.array(sample)
        day_data.append(sample)
        avg_height.append(mrr_height[0])
        # avg_date.append(date2_exp[test])
    # del test1,avg_test
    mean_sum=[]
    sum_height=[]
    
#  the  graph for profile for weekly and monthly
daydata1=np.array(day_data)
avg_date=np.array(avg_date)
avg_height=np.array(avg_height)
# Create a DataFrame
monthly_val=[]
monthly_date=[]
monthly_height=[]

for mon in range(daydata1.shape[1]):      
    mon_val=daydata1[:,mon]
    mon_date=avg_date[:,mon]
    mon_height=avg_height[:,mon]
    mon_date= pd.to_datetime(mon_date)
    df = pd.DataFrame({'date': mon_date, 'value': mon_val, 'height': mon_height})
    df.set_index('date', inplace=True)
    # monthly
    monthly_max1 = df.resample('M').max()
    
    monthly_date1 = monthly_max1.index.date
    monthly_values = monthly_max1['value'].values
    monthly_height1 = monthly_max1['height'].values
    monthly_val.append(monthly_values)
    monthly_date.append(monthly_date1)
    monthly_height.append(monthly_height1)
    
#monthly
monthly_val=np.array(monthly_val)
monthly_date=np.array(monthly_date)
monthly_height=np.array(monthly_height)
monthly_val ,monthly_date,monthly_height= monthly_val.T ,monthly_date.T,monthly_height.T


num_months = monthly_val.shape[0]

# Determine the number of rows and columns for the subplots
ncols = 3  # Number of columns
nrows = (num_months + ncols - 1) // ncols  # Calculate number of rows needed

# Creating subplots
fig, axs = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Loop over each month to create a subplot
for g in range(num_months):
    if np.any(np.isnan(monthly_val[g])) or np.any(np.isnan(monthly_height[g])):
        continue  # Skip this month if there are NaN values
        
    current_date = monthly_date[g, 0]
    axs[g].plot(monthly_val[g], monthly_height[g], label=current_date.strftime('%B %Y'))
    axs[g].set_xlabel('Maximum Rain Rate - mm/hr')
    axs[g].set_ylabel('MRR Height(m)')
    axs[g].set_title(current_date.strftime('%Y-%m '), fontsize=12, fontweight='bold')
    axs[g].legend()

# Remove any empty subplots
for ax in axs[num_months:]:
    fig.delaxes(ax)
# Adjust layout and show plot
plt.tight_layout()
plt.savefig(os.path.join(outpath, 'monthly_profile.png'))
plt.show()
    

   

#  conver the  average data into weekly base and shown in the graph
avg_test=[]
for n in range(0,len(day_data)):
    avg_rr=np.nanmean(day_data[n])
    avg_test.append(avg_rr)
avg_test=np.array(avg_test)


datetime_series = pd.to_datetime(max_date_time)

# Create a DataFrame
avg_data = {
    'datetime': datetime_series,
    'value': avg_test,
    'height':max_height
}

df1 = pd.DataFrame(avg_data)
df1.set_index('datetime', inplace=True)


# weekly data
weekly_avg = df1.resample('W').mean()

weekly_avg_dt = weekly_avg.index.date
weekly_avg_values = weekly_avg['value'].values
weekly_avg_height = weekly_avg['height'].values
# plat graph of the mrr rainrate  melting layer graph

# monthly data
monthly_avg = df1.resample('M').mean()

monthly_avg_dt = monthly_avg.index.date
monthly_avg_values = monthly_avg['value'].values
monthly_avg_height = monthly_avg['height'].values


plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'

fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.plot(weekly_avg_dt,weekly_avg_values ,color='black', linestyle='--',label='avg_rainrate')
ax1.set_ylim(0,100)
# line plot
ax2 = ax1.twinx()
ax2.plot(weekly_avg_dt ,weekly_avg_height , color='blue', linestyle='-', label='avg_height')
ax2.set_ylim(0,6000)
# Display legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines + lines2 , labels + labels2 , loc='upper left', prop={'size': 8})

# Set labels and title
xlab='Time'
ptitle=f'{location} AVG Rainfall '
plt.title(ptitle,fontsize=14, fontweight='bold')
ax1.set_xlabel( xlab,fontsize=12, fontweight='bold')
ax1.set_ylabel('Rain Rate - mm/hr', color='black',fontsize=12, fontweight='bold')
ax2.set_ylabel('MRR Height(m)', color='blue',fontsize=12, fontweight='bold')

ax1.tick_params(axis='both', which='major', labelsize=10)
ax2.tick_params(axis='both', which='major', labelsize=10)


plt.xticks(rotation=45)  # Rotate x-axis labels for readability
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='x', linestyle='--', alpha=0.8)
# Format the x-axis to display dates nicely

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
plt.legend(fontsize=10)
# Display the plot
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outpath, 'weekly_avg.png'))
plt.show()       

# monthly avg_graph 
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'

fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.plot(monthly_avg_dt,monthly_avg_values ,color='black', linestyle='--',label='avg_rainrate')
ax1.set_ylim(0,100)
# line plot
ax2 = ax1.twinx()
ax2.plot(monthly_avg_dt ,monthly_avg_height , color='blue', linestyle='-',label='avg_height')
ax2.set_ylim(0,6000)
# Display legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines + lines2 , labels + labels2 , loc='upper right', prop={'size': 8})

# Set labels and title
xlab='Time'
ptitle=f'{location} AVG Rainfall '
plt.title(ptitle,fontsize=14, fontweight='bold')
ax1.set_xlabel( xlab,fontsize=12, fontweight='bold')
ax1.set_ylabel('Rain Rate - mm/hr', color='black',fontsize=12, fontweight='bold')
ax2.set_ylabel('MRR Height(m)', color='blue',fontsize=12, fontweight='bold')

ax1.tick_params(axis='both', which='major', labelsize=10)
ax2.tick_params(axis='both', which='major', labelsize=10)


plt.xticks(rotation=45)  # Rotate x-axis labels for readability
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='x', linestyle='--', alpha=0.8)
# Format the x-axis to display dates nicely

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.legend(fontsize=10)
# Display the plot
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outpath, 'monthly_avg.png'))
plt.show()       
    




# convert the data into a weekly base and shown in the graph
datetime_series = pd.to_datetime(max_date_time)

# Create a DataFrame
data = {
    'datetime': datetime_series,
    'value': max_val_rr,
    'height':max_height
}

df = pd.DataFrame(data)
df.set_index('datetime', inplace=True)


# weekly data
weekly_max = df.resample('W').max()

weekly_max_dt = weekly_max.index.date
weekly_max_values = weekly_max['value'].values
weekly_max_height = weekly_max['height'].values

# monthly data
monthly_max = df.resample('M').max()

monthly_max_dt = monthly_max.index.date
monthly_max_values = monthly_max['value'].values
monthly_max_height = monthly_max['height'].values






#ploat graph for weekly melting layer
    
# Create the plot
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'

fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.plot(weekly_max_dt,weekly_max_values ,color='black', linestyle='--',label='max_rainrate')
ax1.set_ylim(0,200)
# line plot
ax2 = ax1.twinx()
ax2.plot(weekly_max_dt ,weekly_max_height , color='blue', linestyle='-',label='max_height')
ax2.set_ylim(0,6000)
# Display legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines + lines2 , labels + labels2 , loc='upper right', prop={'size': 8})

# Set labels and title
xlab='Time'
ptitle=f'{location} MAX Rainfall'
plt.title(ptitle,fontsize=14, fontweight='bold')
ax1.set_xlabel( xlab,fontsize=12, fontweight='bold')
ax1.set_ylabel('Rain Rate - mm/hr', color='black',fontsize=12, fontweight='bold')
ax2.set_ylabel('MRR Height(m)', color='blue',fontsize=12, fontweight='bold')

ax1.tick_params(axis='both', which='major', labelsize=10)
ax2.tick_params(axis='both', which='major', labelsize=10)


plt.xticks(rotation=45)  # Rotate x-axis labels for readability
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='x', linestyle='--', alpha=0.8)
# Format the x-axis to display dates nicely

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
plt.legend(fontsize=10)
# Display the plot
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outpath, 'weekly_max.png'))
plt.show()       
    



# plot the graph for the monthly melting layer  for max 
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'

fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.plot(monthly_max_dt,monthly_max_values ,color='black', linestyle='--',label='max_rainrate')
ax1.set_ylim(0,200)
# line plot
ax2 = ax1.twinx()
ax2.plot(monthly_max_dt ,monthly_max_height , color='blue', linestyle='-',label='max_height')
ax2.set_ylim(0,6000)
# Display legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines + lines2 , labels + labels2 , loc='upper right', prop={'size': 8})

# Set labels and title
xlab='Time'
ptitle=f'{location} MAX RainFall '
plt.title(ptitle,fontsize=14, fontweight='bold')
ax1.set_xlabel( xlab,fontsize=12, fontweight='bold')
ax1.set_ylabel('Rain Rate - mm/hr', color='black',fontsize=12, fontweight='bold')
ax2.set_ylabel('MRR Height(m)', color='blue',fontsize=12, fontweight='bold')

ax1.tick_params(axis='both', which='major', labelsize=10)
ax2.tick_params(axis='both', which='major', labelsize=10)


plt.xticks(rotation=45)  # Rotate x-axis labels for readability
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='x', linestyle='--', alpha=0.8)
# Format the x-axis to display dates nicely

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.legend(fontsize=10)
# Display the plot
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outpath, 'monthly_max.png'))
plt.show()  



sum_data=np.array(sum_data)
sum_date==np.array(sum_date)

datetime_series = pd.to_datetime(sum_date)

# Create a DataFrame
data1 = {
    'datetime': sum_date,
    'value': sum_data,
}

df3 = pd.DataFrame(data1)
df3.set_index('datetime', inplace=True)

# monthly data
monthly_sum_values = df3['value'].resample('M').sum()
monthly_sum_values_divided = monthly_sum_values 


monthly_sum_dates= monthly_sum_values.index.date


plt.figure(figsize=(10, 6))
plt.plot(monthly_sum_dates, monthly_sum_values_divided, label='Monthly sum Value')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Monthly sum Values')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(outpath, 'weekly_sum.png'))
plt.show()

# weekly data
weekly_sum_values = df3['value'].resample('W').sum()
weekly_sum_dates= weekly_sum_values.index.date

plt.figure(figsize=(10, 6))
plt.plot(weekly_sum_dates, weekly_sum_values, label='Weekly sum Value')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Weekly sum Values')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(outpath, 'monthly_sum.png'))
plt.show()

