import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from tabulate import tabulate

# *****************************************************************
# Font dictionaries
# ******************************************************************

title_font = {'fontname': 'Courier New', 'size': '16', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
label_font = {'fontname': 'Courier New', 'size': '8', 'color': 'red'}
axis_font = {'fontname': 'Courier New', 'size': '10'}

# *****************************************************************
# Load historical data
# ******************************************************************

symbol = 'TSCO'
data = pd.read_csv('yahoo_pattern.csv')  # data = get_google_data(symbol='TSCO', period=6000, window=10000, exch='LON')
data['Date'] = pd.to_datetime(data['Date'])
# Setup search

reference = data['SPY.Close']  # reference = data['CLOSE']
n = len(reference)
query = reference[(n - 90):n]  # last 90 records
reference = reference[0:(n - 90)]  # the rest

n_query = len(query)
n_reference = len(reference)

# *****************************************************************
# Compute euclidian distances
# ******************************************************************

dist = np.repeat(np.nan, n_reference)
query_normalized = np.array(query - np.mean(query)) / np.std(query)  # apply normalisation to dataset

for i in range(n_query, n_reference):
    window = reference[(i - n_query): i]  # lookback query window length for each i
    window_normalized = np.array(window - np.mean(window)) / np.std(window)  # normalise
    dist[i] = np.linalg.norm(query_normalized - window_normalized)

# *****************************************************************
# Find matches
# ******************************************************************

min_index = []
n_match = 10

temp = copy.copy(dist)  # careful when making a copy, a = b simply assigns reference!!!
temp[temp > np.nanmean(dist)] = np.NaN  # replace distances > mean with NaN's, mean ignoring nans

# remove n_query, points to the left/right of the minimums

for i in range(0, n_match):
    if any(~np.isnan(temp)):  # iterate until while any are not NaN
        index = np.nanargmin(temp)  # index of first minimum value
        min_index.append(index)
        temp[max(0, index - 2*n_query) : min(n_reference, (index + n_query))] = np.NaN  # remove points left/right of minimus

n_match = len(min_index)
print(min_index)

# *****************************************************************
# Plot matches 1) Euclidian distances and starts, 2) Main price series highlight
# ******************************************************************

dates = pd.to_datetime(data['Date'][0:len(dist)])

fig1, ax1 = plt.subplots()
ax1.plot(dates, dist, '-m', color='gray', linewidth = 0.5)
ax1.axhline(np.nanmean(dist), color='gray', linestyle='dashed', linewidth=0.5)
ax1.plot(dates[min_index], dist[min_index], marker='s', linestyle=None, color='red', linewidth=0)
for i, xy in enumerate(zip(dates[min_index], dist[min_index])):
    ax1.annotate('%s' % str(i+1), xy=xy, textcoords='data')
ax1.set_title('Top Matches')
ax1.set_xlabel('Date')
ax1.set_ylabel('Eucledian Distance')
plt.plot()

fig2, ax2 = plt.subplots()
ax2.plot(data['Date'], data['SPY.Close'], '-m', color='gray', linewidth = 0.5, label = 'Price')
ax2.plot(data['Date'][-90:], data['SPY.Close'][-90:], '-m', color='blue', linewidth = 0.5, label = 'Pattern')
for i in range(0, n_match):
    ax2.plot(data['Date'][(min_index[i]-n_query + 1):min_index[i]], data['SPY.Close'][(min_index[i]-n_query + 1):min_index[i]], '-m', color='red', linewidth=0.5, label = 'Match')
for i, xy in enumerate(zip(dates[min_index], data['SPY.Close'][min_index])):
    ax2.annotate('%s' % str(i+1), xy=xy, textcoords='data', color='black')
ax2.set_title(symbol)
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend(loc='upper left')
plt.plot()

# *****************************************************************
# Overlay matches with relative period
# ******************************************************************

matches = np.empty([n_match + 1, (3 * n_query)])  # match + 1 for current series and 3 periods total (1 back 1 current 1 forward)
matches_not_nan = ~np.isnan(matches)  # messy - all I am doing is creating a matrix of Nans...
matches[matches_not_nan] = np.NaN
# 90 nulls + reference (the rest, 2430 records) + query (last 90 records)
# i.e. add 90 nulls to front!! This ensures a 2 window forward period is always available!!
temp = pd.Series(np.repeat(np.nan, n_query)).append(reference).append(query)
for i in range(0, n_match):  # set value for each match 1-10 as slice first index of match - 1 window and + 2 windows
    matches[i, ] = np.array(temp[(min_index[i] - n_query):(min_index[i] + 2 * n_query)])
# Add query pattern (last 90 points)
query_series = pd.Series(temp[(len(temp) - 2 * n_query):(len(temp) + n_query - 1)])  # this is messy
print(np.shape(query_series))
matches[n_match, ] = query_series.append(pd.Series(np.repeat(np.nan, (n_query * 3) - len(query_series))))  # add nulls

for i in range(0, n_match+1):  # calculate returns last div first (this is a form of normalisation)
    matches[i, ] = matches[i, ] / matches[i, n_query]

# *****************************************************************
# Plot overlay
# ******************************************************************

temp = 100 * (np.transpose(matches[:, n_query:]) - 1)  # all indexes except 1-90 (-ve), -1 and *100 to get into %
fig3, ax3 = plt.subplots()
ax3.plot(temp, '-m', color='gray', linewidth = 0.5)
ax3.plot(temp[:, n_match], '-m', color='black', linewidth = 1)  # color last column
ax3.plot(np.repeat(n_query*2 - 1, n_match+1), temp[n_query*2 - 1, :],  marker='o', linestyle=None, color='grey', markersize = 2, linewidth=0)


# Annotate with stats
def stats_labels(x, df, dispatcher, color='red'): # apply list of functions to df at given x value
    """Given x, y inputs, iterates a list of functions on y values and outputs points/annotates to plot """
    for key, element in dispatcher.items():
        y = dispatcher[key](df)
        ax3.plot(x, y, marker='o', linestyle=None, color=color, markersize=2, linewidth=0)
        ax3.annotate('%s %s%%' % (key, np.round(y,1)), xy=(x, y), textcoords='data', **label_font)

dispatcher = {'Min': (lambda x: np.min(x)[0]), 'Max': (lambda x: np.max(x)[0]), 'Med' : (lambda x: np.median(x)),
                'Bot 25%': (lambda x: np.percentile(x, 25)), 'Top 75%' : (lambda x: np.percentile(x, 75))}
ref_ends = pd.DataFrame(temp[2*n_query - 1, 0:n_match])  # reference ends
stats_labels(n_query*2 - 1, ref_ends, dispatcher, color='red')
stats_labels(n_query-1, temp[n_query - 1, n_match], dispatcher={'Current': lambda x: x}, color='red')

#  axis and fonts
plt.xticks(**axis_font)
plt.yticks(**axis_font)
ax3.set_title(symbol + ' pattern prediction with 10 neighbours', **title_font)
ax3.set_xlabel('Period', **axis_font)
ax3.set_ylabel('Returns %', **axis_font)

# *****************************************************************
# Table with predictions
# ******************************************************************

#  initialise empty table
temp = pd.DataFrame(np.nan, index=range(0, n_match), columns=['match no.', 'start', 'end', 'return', '5 period', '20 period', '60 period']) #  n_match, 0 indexed. Remember python doesnt include last value in a:b!!
temp = temp.append(pd.DataFrame(np.nan, index=['current', 'min', 'average', 'max'], columns=['match no.', 'start', 'end', 'return', '5 period', '20 period', '60 period'])) #  plus 4 stats, 0 indexed

#  populate table, 2) takes latest return value div 90th return value (i.e. current position)
# 0 is -90th value. 89 is -1st value. 90 is 1st value. 179 is +90th value. 269 is +180th value
temp.loc[range(0, n_match), 'match no.'] = range(1, 11)
temp.loc[range(0, n_match), 'return'] = ((matches[0:n_match, 3*n_query-1] / matches[0:n_match, 2*n_query-1]) - 1) * 100  # iloc is indexed based e.g. (1,2,3),(2,3) and loc is name based
temp.loc['current', 'return'] = ((matches[n_match, 3*n_query-1] / matches[n_match, 2*n_query-1]) - 1) * 100
temp.loc[range(0, n_match), '5 period'] = ((matches[0:n_match, (2*n_query-1)+6] / matches[0:n_match, 2*n_query-1]) - 1) * 100  # 'current'+5 days div 'current' level. 0 indexed hence 6...
temp.loc[range(0, n_match), '20 period'] = ((matches[0:n_match, (2*n_query-1)+21] / matches[0:n_match, 2*n_query-1]) - 1) * 100  # 'current'+5 days div 'current' level. 0 indexed hence 6...
temp.loc[range(0, n_match), '60 period'] = ((matches[0:n_match, (2*n_query-1)+61] / matches[0:n_match, 2*n_query-1]) - 1) * 100  # 'current'+5 days div 'current' level. 0 indexed hence 6...

# compute average returns
index = ['return', '5 period', '20 period', '60 period']
temp.loc['min', index] = np.min(temp.loc[range(0, n_match), index])
temp.loc['average', index] = np.mean(temp.loc[range(0, n_match), index])
temp.loc['max', index] = np.max(temp.loc[range(0, n_match), index])
temp.loc['current', 'start'] = data['Date'][(n - n_query)].format('%Y-%m-%d')
temp.loc['current', 'end'] = data['Date'][n-1].format('%Y-%m-%d')

for i in range(0, n_match):
    temp.loc[i, 'start'] = data['Date'][(min_index[i]-n_query + 1)].format('%Y-%m-%d')
    temp.loc[i, 'end'] = data['Date'][min_index[i]-1].format('%Y-%m-%d')

print(tabulate(temp, headers='keys', tablefmt='psql'))
