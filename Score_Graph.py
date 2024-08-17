import pandas as pd
from scipy.stats import ranksums



base_data_read_path = "enter excel data read path here"
num_of_problems = 50
HVdata = {}
FRdata = {}
timedata = {}



for x in range(1, num_of_problems + 1):
    prob = 'RWMOP' + str(x)
    tempdata = pd.read_csv(base_data_read_path + prob + '_HV.csv')
    HVdata[prob] = tempdata
    tempdata = pd.read_csv(base_data_read_path + prob + '_Feasible_rate.csv')
    FRdata[prob] = tempdata
    tempdata = pd.read_csv(base_data_read_path + prob + '_runtime.csv')
    timedata[prob] = tempdata
    
    
    
def is_less(x, y, alpha=0.05):
    stat, p = ranksums(x, y)
    score = 'temp'
    if p < alpha:
        if stat < 0:
            score = 'less'
        else:
            score = 'greater'
    else:
        score = 'insignificant'
    return score



problems = list(HVdata.keys())
algos = list(HVdata['RWMOP1'].keys())
HV_scores = {}
FR_scores = {}
time_scores = {}
tot_scores = {}


for algo in algos:
    tot_scores[algo] = 0
    HV_scores[algo] = 0
    FR_scores[algo] = 0
    time_scores[algo] = 0



for prob in problems:
    for algo1 in algos:
        x = list(HVdata[prob][algo1])
        x2 = list(FRdata[prob][algo1])
        x3 = list(timedata[prob][algo1])
        
        for algo2 in algos:
            if algo2 == algo1:
                continue
            else:
                current=0
                y = list(HVdata[prob][algo2])
                score = is_less(x, y)    
                if score == 'less':
                    current = current + 1
                    HV_scores[algo2] += 1
                elif score == 'insignificant':
                    y2 = list(FRdata[prob][algo2])
                    score2 = is_less(x2, y2)
                    if score2 == 'less':
                        FR_scores[algo2] += 1
                        current = current + 1
                    elif score2 == 'insignificant':
                        y3 = list(timedata[prob][algo2])
                        score3 = is_less(x3, y3)
                        if score3 == 'greater':
                            time_scores[algo2] += 1
                            current = current + 1
                        elif score3 == 'insignificant':
                            continue
                else:
                    continue
                tot_scores[algo1] += current/(len(algos) - 1)
      
for x in tot_scores:
    tot_scores[x] = tot_scores[x]/(num_of_problems)
  

    
HVMean = {}
FRMean = {}
TimeMean = {}
HVMeanMax = {}
FRMeanMax = {}
TimeMeanMax = {}



for x in HVdata:
    averages = HVdata[x].mean(axis=0)
    HVMean[x] = pd.DataFrame(averages.values.reshape(1, len(averages)), columns=averages.index)
    HVMeanMax[x] = averages.idxmax()

for x in FRdata:
    averages = FRdata[x].mean(axis=0)
    FRMean[x] = pd.DataFrame(averages.values.reshape(1, len(averages)), columns=averages.index)
    FRMeanMax[x] = averages.idxmax()

for x in timedata:
    averages = timedata[x].mean(axis=0)
    TimeMean[x] = pd.DataFrame(averages.values.reshape(1, len(averages)), columns=averages.index)
    TimeMeanMax[x] = averages.idxmax()
    
    
    
def flatten_nested_df(data_dict):
    flattened_df = pd.concat([df for df in data_dict.values()], ignore_index=True)
    return flattened_df

flat_HVMean = flatten_nested_df(HVMean).T
flat_HVMean = flat_HVMean.rename(columns=lambda x: 'RWCMOP' + str(x + 1))

flat_FRMean = flatten_nested_df(FRMean).T
flat_FRMean = flat_FRMean.rename(columns=lambda x: 'RWCMOP' + str(x + 1))

flat_TimeMean = flatten_nested_df(TimeMean).T
flat_TimeMean = flat_TimeMean.rename(columns=lambda x: 'RWCMOP' + str(x + 1))

problems21 = problems[:21]
problems29 = problems[21:29]
problems35 = problems[29:35]
problems50 = problems[35:50]


flat_HVMean_21 = flat_HVMean.iloc[:,:21]
flat_HVMean_29 = flat_HVMean.iloc[:,21:29]
flat_HVMean_35 = flat_HVMean.iloc[:,29:35]
flat_HVMean_50 = flat_HVMean.iloc[:,35:50]


flat_FRMean_21 = flat_FRMean.iloc[:,:21]
flat_FRMean_29 = flat_FRMean.iloc[:,21:29]
flat_FRMean_35 = flat_FRMean.iloc[:,29:35]
flat_FRMean_50 = flat_FRMean.iloc[:,35:50]


flat_TimeMean_21 = flat_TimeMean.iloc[:,:21]
flat_TimeMean_29 = flat_TimeMean.iloc[:,21:29]
flat_TimeMean_35 = flat_TimeMean.iloc[:,29:35]
flat_TimeMean_50 = flat_TimeMean.iloc[:,35:50]






import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import wilcoxon, friedmanchisquare

plt.figure(figsize=(18, 12))
colors = plt.cm.tab20(np.linspace(0, 1, 25))
for i, algorithm in enumerate(algos):
    plt.plot(problems21, flat_HVMean_21.loc[algorithm], linestyle='--', marker='o', label=algorithm, color=colors[i])
plt.title('HV Values of Algorithms over RWCMOP 1-21')
plt.xlabel('Problems')
plt.ylabel('HV Values')
plt.xticks(ticks=np.arange(0, 21), labels=[f'RWCMOP{i+1}' for i in range(21)], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\hv\\line_21.png")
plt.show()


plt.figure(figsize=(18, 12))
sns.heatmap(flat_HVMean_21, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Performance'}, linewidths=0.5)
plt.title('HV Values of Algorithms over RWCMOP 1-21')
plt.xlabel('Problems')
plt.ylabel('Algorithms')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\hv\\heatmap_21.png")
plt.show()



plt.figure(figsize=(18, 12))
colors = plt.cm.tab20(np.linspace(0, 1, 25))
for i, algorithm in enumerate(algos):
    plt.plot(problems29, flat_HVMean_29.loc[algorithm], linestyle='--', marker='o', label=algorithm, color=colors[i])
plt.title('HV Values of Algorithms over RWCMOP 22-29')
plt.xlabel('Problems')
plt.ylabel('HV Values')
plt.xticks(ticks=np.arange(0, 8), labels=[f'RWCMOP{i+22}' for i in range(8)], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\hv\\line_29.png")
plt.show()

plt.figure(figsize=(18, 12))
sns.heatmap(flat_HVMean_29, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Performance'}, linewidths=0.5)
plt.title('HV Values of Algorithms over RWCMOP 22-29')
plt.xlabel('Problems')
plt.ylabel('Algorithms')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\hv\\heatmap_29.png")
plt.show()



# Plot scatter points with dotted lines
plt.figure(figsize=(18, 12))
colors = plt.cm.tab20(np.linspace(0, 1, 25))
for i, algorithm in enumerate(algos):
    plt.plot(problems35, flat_HVMean_35.loc[algorithm], linestyle='--', marker='o', label=algorithm, color=colors[i])
plt.title('HV Values of Algorithms over RWCMOP 30-35')
plt.xlabel('Problems')
plt.ylabel('HV Values')
plt.xticks(ticks=np.arange(0, 6), labels=[f'RWCMOP{i+30}' for i in range(6)], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\hv\\line_35.png")
plt.show()

plt.figure(figsize=(18, 12))
sns.heatmap(flat_HVMean_35, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Performance'}, linewidths=0.5)
plt.title('HV Values of Algorithms over RWCMOP 30-35')
plt.xlabel('Problems')
plt.ylabel('Algorithms')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\hv\\heatmap_35.png")
plt.show()



# Plot scatter points with dotted lines
plt.figure(figsize=(18, 12))
colors = plt.cm.tab20(np.linspace(0, 1, 25))
for i, algorithm in enumerate(algos):
    plt.plot(problems50, flat_HVMean_50.loc[algorithm], linestyle='--', marker='o', label=algorithm, color=colors[i])
plt.title('HV Values of Algorithms over RWCMOP 36-50')
plt.xlabel('Problems')
plt.ylabel('HV Values')
plt.xticks(ticks=np.arange(0, 15), labels=[f'RWCMOP{i+36}' for i in range(15)], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\hv\\line_50.png")
plt.show()

plt.figure(figsize=(18, 12))
sns.heatmap(flat_HVMean_50, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Performance'}, linewidths=0.5)
plt.title('HV Values of Algorithms over RWCMOP 36-50')
plt.xlabel('Problems')
plt.ylabel('Algorithms')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\hv\\heatmap_50.png")
plt.show()



friedman_stat, friedman_p = friedmanchisquare(*flat_HVMean_21.values)
print(f'Friedman test statistic: {friedman_stat}, p-value: {friedman_p}')

# Plot the critical diagram
sp.plot_cd(flat_HVMean_21.values, alpha='0.05', title='Critical Difference Diagram')
plt.show()

nemenyi = sp.posthoc_nemenyi_friedman(flat_HVMean_21.T.values)
print(nemenyi)

sp.sign_plot(nemenyi.values)
plt.show()























# Plot scatter points with dotted lines
plt.figure(figsize=(18, 12))
colors = plt.cm.tab20(np.linspace(0, 1, 25))
for i, algorithm in enumerate(algos):
    plt.plot(problems21, flat_FRMean_21.loc[algorithm], linestyle='--', marker='o', label=algorithm, color=colors[i])
plt.title('FR values of Algorithms over RWCMOP 1-21')
plt.xlabel('Problems')
plt.ylabel('FR Values')
plt.xticks(ticks=np.arange(0, 21), labels=[f'RWCMOP{i+1}' for i in range(21)], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\fr\\line_21.png")
plt.show()


plt.figure(figsize=(18, 12))
sns.heatmap(flat_FRMean_21, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Performance'}, linewidths=0.5)
plt.title('FR Values of Algorithms over RWCMOP 1-21')
plt.xlabel('Problems')
plt.ylabel('Algorithms')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\fr\\heatmap_21.png")
plt.show()




# Plot scatter points with dotted lines
plt.figure(figsize=(18, 12))
colors = plt.cm.tab20(np.linspace(0, 1, 25))
for i, algorithm in enumerate(algos):
    plt.plot(problems29, flat_FRMean_29.loc[algorithm], linestyle='--', marker='o', label=algorithm, color=colors[i])
plt.title('FR values of Algorithms over RWCMOP 22-29')
plt.xlabel('Problems')
plt.ylabel('FR Values')
plt.xticks(ticks=np.arange(0, 8), labels=[f'RWCMOP{i+22}' for i in range(8)], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\fr\\line_29.png")
plt.show()

plt.figure(figsize=(18, 12))
sns.heatmap(flat_FRMean_29, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Performance'}, linewidths=0.5)
plt.title('FR Values of Algorithms over RWCMOP 22-29')
plt.xlabel('Problems')
plt.ylabel('Algorithms')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\fr\\heatmap_29.png")
plt.show()



# Plot scatter points with dotted lines
plt.figure(figsize=(18, 12))
colors = plt.cm.tab20(np.linspace(0, 1, 25))
for i, algorithm in enumerate(algos):
    plt.plot(problems35, flat_FRMean_35.loc[algorithm], linestyle='--', marker='o', label=algorithm, color=colors[i])
plt.title('FR values of Algorithms over RWCMOP 30-35')
plt.xlabel('Problems')
plt.ylabel('FR Values')
plt.xticks(ticks=np.arange(0, 6), labels=[f'RWCMOP{i+30}' for i in range(6)], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\fr\\line_35.png")
plt.show()

plt.figure(figsize=(18, 12))
sns.heatmap(flat_FRMean_35, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Performance'}, linewidths=0.5)
plt.title('FR Values of Algorithms over RWCMOP 30-35')
plt.xlabel('Problems')
plt.ylabel('Algorithms')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\fr\\heatmap_35.png")
plt.show()



# Plot scatter points with dotted lines
plt.figure(figsize=(18, 12))
colors = plt.cm.tab20(np.linspace(0, 1, 25))
for i, algorithm in enumerate(algos):
    plt.plot(problems50, flat_FRMean_50.loc[algorithm], linestyle='--', marker='o', label=algorithm, color=colors[i])
plt.title('FR values of Algorithms over RWCMOP 36-50')
plt.xlabel('Problems')
plt.ylabel('FR Values')
plt.xticks(ticks=np.arange(0, 15), labels=[f'RWCMOP{i+36}' for i in range(15)], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\fr\\line_50.png")
plt.show()

plt.figure(figsize=(18, 12))
sns.heatmap(flat_FRMean_50, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Performance'}, linewidths=0.5)
plt.title('FR Values of Algorithms over RWCMOP 36-50')
plt.xlabel('Problems')
plt.ylabel('Algorithms')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\fr\\heatmap_50.png")
plt.show()

















# Plot scatter points with dotted lines
plt.figure(figsize=(18, 12))
colors = plt.cm.tab20(np.linspace(0, 1, 25))
for i, algorithm in enumerate(algos):
    plt.plot(problems21, flat_TimeMean_21.loc[algorithm], linestyle='--', marker='o', label=algorithm, color=colors[i])
plt.title('Time values of Algorithms over RWCMOP 1-21')
plt.xlabel('Problems')
plt.ylabel('Time Values')
plt.xticks(ticks=np.arange(0, 21), labels=[f'RWCMOP{i+1}' for i in range(21)], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\time\\line_21.png")
plt.show()


plt.figure(figsize=(18, 12))
sns.heatmap(flat_TimeMean_21, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Performance'}, linewidths=0.5)
plt.title('Time Values of Algorithms over RWCMOP 1-21')
plt.xlabel('Problems')
plt.ylabel('Algorithms')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\time\\heatmap_21.png")
plt.show()




# Plot scatter points with dotted lines
plt.figure(figsize=(18, 12))
colors = plt.cm.tab20(np.linspace(0, 1, 25))
for i, algorithm in enumerate(algos):
    plt.plot(problems29, flat_TimeMean_29.loc[algorithm], linestyle='--', marker='o', label=algorithm, color=colors[i])
plt.title('Time values of Algorithms over RWCMOP 22-29')
plt.xlabel('Problems')
plt.ylabel('Time Values')
plt.xticks(ticks=np.arange(0, 8), labels=[f'RWCMOP{i+22}' for i in range(8)], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\time\\line_29.png")
plt.show()

plt.figure(figsize=(18, 12))
sns.heatmap(flat_TimeMean_29, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Performance'}, linewidths=0.5)
plt.title('Time Values of Algorithms over RWCMOP 22-29')
plt.xlabel('Problems')
plt.ylabel('Algorithms')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\time\\heatmap_29.png")
plt.show()



# Plot scatter points with dotted lines
plt.figure(figsize=(18, 12))
colors = plt.cm.tab20(np.linspace(0, 1, 25))
for i, algorithm in enumerate(algos):
    plt.plot(problems35, flat_TimeMean_35.loc[algorithm], linestyle='--', marker='o', label=algorithm, color=colors[i])
plt.title('Time values of Algorithms over RWCMOP 30-35')
plt.xlabel('Problems')
plt.ylabel('Time Values')
plt.xticks(ticks=np.arange(0, 6), labels=[f'RWCMOP{i+30}' for i in range(6)], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\time\\line_35.png")
plt.show()

plt.figure(figsize=(18, 12))
sns.heatmap(flat_TimeMean_35, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Performance'}, linewidths=0.5)
plt.title('Time Values of Algorithms over RWCMOP 30-35')
plt.xlabel('Problems')
plt.ylabel('Algorithms')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\time\\heatmap_35.png")
plt.show()



# Plot scatter points with dotted lines
plt.figure(figsize=(18, 12))
colors = plt.cm.tab20(np.linspace(0, 1, 25))
for i, algorithm in enumerate(algos):
    plt.plot(problems50, flat_TimeMean_50.loc[algorithm], linestyle='--', marker='o', label=algorithm, color=colors[i])
plt.title('Time values of Algorithms over RWCMOP 36-50')
plt.xlabel('Problems')
plt.ylabel('Time Values')
plt.xticks(ticks=np.arange(0, 15), labels=[f'RWCMOP{i+36}' for i in range(15)], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\time\\line_50.png")
plt.show()

plt.figure(figsize=(18, 12))
sns.heatmap(flat_TimeMean_50, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Performance'}, linewidths=0.5)
plt.title('Time Values of Algorithms over RWCMOP 36-50')
plt.xlabel('Problems')
plt.ylabel('Algorithms')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("D:\\My Folder\\Aus\\Next\\Final\\graph\\time\\heatmap_50.png")
plt.show()


    
df = pd.DataFrame(tot_scores, index=[0])

result_df = pd.DataFrame({
    'Column Name': df.columns,
    'Value': df.iloc[0]
})

result_df.to_csv("write path", index=False)                  
    