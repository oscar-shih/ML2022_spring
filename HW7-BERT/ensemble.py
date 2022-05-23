path_1 = "./output_0.82.csv"
path_2 = "./result_post.csv"

f1_ans, f2_ans = [], []

with open(path_1, "r") as f1:
    for lines in f1:
        if lines != "ID,Answer\n":
            f1_ans.append(lines.split("\n")[0].split(",")[1])

with open(path_2, "r") as f2:
    for lines in f2:
        if lines != "ID,Answer\n":
            f2_ans.append(lines.split("\n")[0].split(",")[1])

for i in range(len(f1_ans)):
    if f2_ans[i] == "" and f1_ans[i] != "":
        print(i)
        print(f1_ans[i])
        f2_ans[i] = f1_ans[i]
    
with open("./output.csv", "w") as out:
    out.write("ID,Answer\n")
    for i in range(len(f2_ans)):
        out.write(str(i)+","+f2_ans[i]+"\n")