import os
import matplotlib.pyplot as plt

m_values = range(3,21)
dsatur_no_colours = range(3,21)
my_avg_colours = [3,3,3,3,3,3,3,3.08,3,3,3,3.08,3.17,3.17,3.25,3.17,3.33,3.5]
my_min_colours = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
min_error = [avg-min_val for min_val,avg in zip(my_min_colours, my_avg_colours)]
my_max_colours = [3,3,3,3,3,3,3,4,3,3,3,4,4,4,4,4,4,5]
max_error = [max_val-avg for max_val,avg in zip(my_max_colours, my_avg_colours)]

fig = plt.figure(figsize=(12,4.5))

plt.errorbar(m_values, dsatur_no_colours, ls='', marker='s', ms=5, c='orange', label='DSATUR')
# plt.scatter(m_values, dsatur_no_colours, c='orange', marker='D', s=30, label='DSATUR')
#             label = legend_labels[ind], color=line_colour, linewidth=.5)

plt.errorbar(m_values, my_avg_colours, yerr=[min_error, max_error], ls='', marker='D', ms=4, c='blue', label='ReLCol')

plt.title('Comparison of DSATUR and ReLCol on Spinrad graphs', fontsize=14)
plt.xlabel("m", fontsize=12)
plt.ylabel("No. of Colours Used", fontsize=12)
plt.xticks(ticks=range(3,21), fontsize=10)
plt.yticks(ticks=[0,5,10,15,20], fontsize=10)
plt.ylim(0, 22)
plt.legend(loc='upper left', framealpha=.75, fontsize=12)

fig.savefig(os.path.join('outputs', 'graphs', 'spinrad_comparison'))