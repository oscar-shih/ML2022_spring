import os

path = './result.csv'
result = []
new = []

with open(path, "r") as input:
    for lines in input:
        new.append(lines.split("\n")[0].split(",")[1])

for i in range(len(new)):
    if new[i] == "":
        continue
    if new[i][-1] == "》" and new[i][0] != "《":
        print(new[i])
        new[i] = "《" + new[i]
        print(new[i])
    if new[i][-1] == "」" and new[i][0] != "「":
        print(new[i])
        new[i] = "「" + new[i]
        print(new[i])
    if new[i][0] == "《" and new[i][-1] != "》":
        print(new[i])
        new[i] = new[i] + "》"
        print(new[i])
    if new[i][0] == "「" and new[i][-1] != "」":
        print(new[i])
        new[i] = new[i] + "」"
        print(new[i])
    
with open("result_post.csv", "w") as output:
    output.write("ID," + new[0] + "\n")
    for i in range(len(new)):
        if i == 0:
            continue
        output.write(str(i-1) + "," + new[i] + "\n")