path_1 = '/home/oscarshih/Desktop/ML/Hw1-Regression/pred_0.75.csv'
path_2 = '/home/oscarshih/Desktop/ML/Hw1-Regression/pred_0.768.csv'
path_3 = '/home/oscarshih/Desktop/ML/Hw1-Regression/pred_1.027.csv'
path_4 = '/home/oscarshih/Downloads/pred_1.csv'
path_5 = '/home/oscarshih/Downloads/pred.csv'

path_ensemble = '/home/oscarshih/Desktop/ML/Hw1-Regression/pred_ensemble.csv'

pred_1 =[]
with open(path_1) as f1:
    for lines in f1:
        if (lines != 'id,tested_positive\n'):
            pred_1.append(float(lines.split(',')[-1]))

pred_2 =[]
with open(path_2) as f2:
    for lines in f2:
        if (lines != 'id,tested_positive\n'):
            pred_2.append(float(lines.split(',')[-1]))

pred_3 =[]
with open(path_3) as f3:
    for lines in f3:
        if (lines != 'id,tested_positive\n'):
            pred_3.append(float(lines.split(',')[-1]))

pred_4 =[]
with open(path_4) as f4:
    for lines in f4:
        if (lines != 'id,tested_positive\n'):
            pred_4.append(float(lines.split(',')[-1]))

pred_5 =[]
with open(path_5) as f5:
    for lines in f5:
        if (lines != 'id,tested_positive\n'):
            pred_5.append(float(lines.split(',')[-1]))

pred = []
for i in range(len(pred_5)):
    avg = (pred_1[i] + pred_2[i] + pred_3[i] + pred_4[i] + pred_5[i]) / 5
    pred.append(format(avg, '.6f'))

with open(path_ensemble, 'w') as output:
    output.write('id,tested_positive\n')
    for i in range(len(pred)):
        row = str(i) + ',' + str(pred[i])
        output.write(row)
        output.write('\n')
