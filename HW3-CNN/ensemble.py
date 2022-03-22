import pandas as pd
from collections import Counter

path_1 = '/home/oscarshih/Downloads/submission.csv'
path_2 = '/home/oscarshih/Downloads/submission(1).csv'
path_3 = '/home/oscarshih/Downloads/submission(2).csv'
path_4 = '/home/oscarshih/Downloads/submission(3).csv'
path_5 = '/home/oscarshih/Downloads/submission(4).csv'

arr_1 = []
arr_2 = []
arr_3 = []
arr_4 = []
arr_5 = []

with open(path_1) as f1:
    for lines in f1:
        if (lines != 'Id,Category\n'):
            arr_1.append(lines.split(',')[-1].split('\n')[0])

with open(path_2) as f2:
    for lines in f2:
        if (lines != 'Id,Category\n'):
            arr_2.append(lines.split(',')[-1].split('\n')[0])

with open(path_3) as f3:
    for lines in f3:
        if (lines != 'Id,Category\n'):
            arr_3.append(lines.split(',')[-1].split('\n')[0])

with open(path_5) as f4:
    for lines in f4:
        if (lines != 'Id,Category\n'):
            arr_4.append(lines.split(',')[-1].split('\n')[0])

with open(path_5) as f5:
    for lines in f5:
        if (lines != 'Id,Category\n'):
            arr_5.append(lines.split(',')[-1].split('\n')[0])

pred = []
for i in range(len(arr_1)):
    temp = []
    temp.append(arr_1[i])
    temp.append(arr_2[i])
    temp.append(arr_3[i])
    temp.append(arr_4[i])
    temp.append(arr_5[i])
    classification = Counter(temp)
    pred.append(classification.most_common(1)[0][0])


def pad4(i):
    return "0"*(4-len(str(i)))+str(i)

df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1, len(pred)+1)]
df["Category"] = pred
df.to_csv("ensemble.csv", index = False)
