import os
import matplotlib.pyplot as plt

graph_sizes = [25,50,75,100,150,200]
dsatur_no_colours = [5.352, 8.248, 10.6, 13.42, 15.236, 17.568]
random_avg_colours = [6.52244, 9.9258, 12.93552, 16.0742, 18.93708, 21.01888]
my_avg_colours = [5.444, 8.43633333333333, 11.1063333333333, 14.473, 17.827, 20.5853333333333]
# my_min_colours = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
# min_error = [avg-min_val for min_val,avg in zip(my_min_colours, my_avg_colours)]
# my_max_colours = [3,3,3,3,3,3,3,4,3,3,3,4,4,4,4,4,4,5]
# max_error = [max_val-avg for max_val,avg in zip(my_max_colours, my_avg_colours)]

fig = plt.figure(figsize=(12,4.5))

plt.errorbar(graph_sizes, random_avg_colours, ls='', marker='o', ms=5, c='green', label='Random')
# plt.errorbar(graph_sizes, my_avg_colours, yerr=[min_error, max_error], ls='', marker='D', ms=4, c='blue', label='ReLCol')
plt.errorbar(graph_sizes, my_avg_colours, ls='', marker='D', ms=4, c='blue', label='ReLCol')
plt.errorbar(graph_sizes, dsatur_no_colours, ls='', marker='s', ms=5, c='orange', label='DSATUR')

plt.title('Comparison of DSATUR, ReLCol and Random on different sized graphs', fontsize=14)
plt.xlabel("n", fontsize=12)
plt.ylabel("No. of Colours Used", fontsize=12)
plt.xticks(ticks=[0,25,50,75,100,125,150,175,200], fontsize=10)
plt.xlim(20, 205)
plt.yticks(ticks=[0,5,10,15,20], fontsize=10)
plt.ylim(0, 22)
plt.legend(loc='upper left', framealpha=.75, fontsize=12)

fig.savefig(os.path.join('outputs', 'graphs', 'larger_graphs_comparison'))