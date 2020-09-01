'''
@file       plotCreator.py
@date       2020/08/31
@brief      Script to generate plots for the Annotator benchmarking results
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read in all relevant CSV files
results1 = pd.read_csv('Results/Annotator_Statistics_1090_95.csv').to_numpy().T
results2 = pd.read_csv('Results/Annotator_Statistics_2080_95_1.csv').to_numpy().T
results3 = pd.read_csv('Results/Annotator_Statistics_3070_95.csv').to_numpy().T
results4 = pd.read_csv('Results/Annotator_Statistics_2080_80_2.csv').to_numpy().T
results5 = pd.read_csv('Results/Annotator_Statistics_2080_82_1.csv').to_numpy().T
results6 = pd.read_csv('Results/Annotator_Statistics_2080_85_2.csv').to_numpy().T
results7 = pd.read_csv('Results/Annotator_Statistics_2080_90_2.csv').to_numpy().T

'''CSV column indices
1 Active Learning Iteration
2 Labeled Database Size
3 Unlabeled Database Size
4 Precision (Tag)
5 Recall (Tag)
6 F1 Score (Tag)
7 Hamming Loss
8 Accuracy (Tag List)
'''

'''
# Annotator_Statistics_2080_95_1 plots
iteration = results2[1]
labeledSize = results2[2]
unlabeledSize = results2[3]
precisison = results2[4]
recall = results2[5]
fscore = results2[6]
hammingLoss = results2[7]
accuracy = results2[8]

fig = plt.figure(figsize=(14.4, 4.8))
sp1 = fig.add_subplot(1,2,1)
sp1.plot(iteration, unlabeledSize, c='r', label='Unlabeled Database Size')
sp1.plot(iteration, labeledSize, c='b', label='Labeled Database Size')

plt.title('Database Sizes')
plt.xlabel('Active Learning Iteration')
plt.ylabel('Database Size (Samples)')
plt.legend()

sp2 = fig.add_subplot(1,2,2)
sp2.plot(iteration, precisison, c='b', label='Precision (Tag)')
sp2.plot(iteration, recall, c='r', label='Recall (Tag)')
sp2.plot(iteration, fscore, c='g', label='F1 Score (Tag)')
sp2.plot(iteration, np.ones(8) - hammingLoss, c='m', label='Inverse Hamming Loss')
sp2.plot(iteration, accuracy, c='k', label='Accuracy (Tag List)')

plt.title('Performance Metric Comparison')
plt.xlabel('Active Learning Iteration')
plt.ylabel('Normalized Score (0 to 1)')
plt.legend()
'''

'''
# Database comparison plots (Confidence Calculation: Average)
iteration = results1[1]
labeledSize1 = results1[2]
unlabeledSize1 = results1[3]
fscore1 = results1[6]
labeledSize2 = results2[2]
unlabeledSize2 = results2[3]
fscore2 = results2[6]
labeledSize3 = results3[2]
unlabeledSize3 = results3[3]
fscore3 = results3[6]

fig = plt.figure(figsize=(14.4, 4.8))
sp1 = fig.add_subplot(1,2,1)
sp1.plot(iteration, unlabeledSize1, c='r', ls=':', label='Unlabeled Database (10-90)')
sp1.plot(iteration, labeledSize1, c='b', ls=':', label='Labeled Database (10-90)')
sp1.plot(iteration, unlabeledSize2, c='r', ls='--', label='Unlabeled Database (20-80)')
sp1.plot(iteration, labeledSize2, c='b', ls='--', label='Labeled Database (20-80)')
sp1.plot(iteration, unlabeledSize3, c='r', ls='-', label='Unlabeled Database (30-70)')
sp1.plot(iteration, labeledSize3, c='b', ls='-', label='Labeled Database (30-70)')

plt.title('Database Sizes')
plt.xlabel('Active Learning Iteration')
plt.ylabel('Database Size (Samples)')
plt.legend()

sp2 = fig.add_subplot(1,2,2)
sp2.plot(iteration, fscore1, c='g', ls=':', label='F1 Score (10-90)')
sp2.plot(iteration, fscore2, c='g', ls='--', label='F1 Score (20-80)')
sp2.plot(iteration, fscore3, c='g', ls='-', label='F1 Score (30-70)')

plt.title('F1 Score Comparison')
plt.xlabel('Active Learning Iteration')
plt.ylabel('Normalized Score (0 to 1)')
plt.legend()
'''


# Database comparison plots (Confidence Calculation: Average)
iteration = results4[1]
labeledSize1 = results4[2]
unlabeledSize1 = results4[3]
fscore1 = results4[6]
labeledSize2 = results5[2]
unlabeledSize2 = results5[3]
fscore2 = results5[6]
labeledSize3 = results6[2]
unlabeledSize3 = results6[3]
fscore3 = results6[6]
labeledSize4 = results7[2]
unlabeledSize4 = results7[3]
fscore4 = results7[6]

fig = plt.figure(figsize=(14.4, 4.8))
sp1 = fig.add_subplot(1,2,1)
sp1.plot(iteration, unlabeledSize1, c='r', ls=':', label='ULDB (70)')
sp1.plot(iteration, labeledSize1, c='b', ls=':', label='LDB (70)')
sp1.plot(iteration, unlabeledSize2, c='r', ls='-.', label='ULDB (80)')
sp1.plot(iteration, labeledSize2, c='b', ls='-.', label='LDB (80)')
sp1.plot(iteration, unlabeledSize3, c='r', ls='--', label='ULDB (85)')
sp1.plot(iteration, labeledSize3, c='b', ls='--', label='LDB (85)')
sp1.plot(iteration, unlabeledSize4, c='r', ls='-', label='ULDB (90)')
sp1.plot(iteration, labeledSize4, c='b', ls='-', label='LDB (90)')

plt.title('Database Sizes')
plt.xlabel('Active Learning Iteration')
plt.ylabel('Database Size (Samples)')
plt.legend()

sp2 = fig.add_subplot(1,2,2)
sp2.plot(iteration, fscore1, c='g', ls=':', label='F1 Score (70 Threshold)')
sp2.plot(iteration, fscore2, c='g', ls='-.', label='F1 Score (80 Threshold)')
sp2.plot(iteration, fscore3, c='g', ls='--', label='F1 Score (85 Threshold)')
sp2.plot(iteration, fscore4, c='g', ls='-', label='F1 Score (90 Threshold)')

plt.title('F1 Score Comparison')
plt.xlabel('Active Learning Iteration')
plt.ylabel('Normalized Score (0 to 1)')
plt.legend()



# Display all plots
plt.show()




