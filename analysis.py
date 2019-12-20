

def read_txt(list_path):
    with open(list_path) as f:
        content = []
        for line in f:
            line = line.strip()
            content.append(line)
    return content


cont = read_txt("log/margin1_epoch20_seed10_test.txt")

dic = {}
for line in cont:
    print(line)
    spl = line.split()
    seed = int(spl[0])
    mw = float(spl[1])
    acc = spl[-2]

    dic[seed] 
    
