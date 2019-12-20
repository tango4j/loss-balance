import pandas as pd
import numpy as np
import ipdb

def get_all_index(data, x, itv):
    out_list = []
    for i in range(0, int(data.shape[0]/itv)):
        # print(i)
        out_list.append(data.iloc[x+ itv*i, -2])
    return out_list 

def get_all_in_seed(data, seed, itv):
    # print(seed * itv, (seed+1)* itv)
    try:
        out_list = list(data.iloc[seed*itv: (seed+1)*itv, -2] )
    except:
        ipdb.set_trace()
    return out_list 

def read_txt(list_path):
    with open(list_path) as f:
        content = []
        for line in f:
            line = line.strip()
            content.append(line)
    return content

def write_txt(path, the_list): 
    outF = open(path, "w")
    for line in the_list:
        outF.write(line)
        outF.write("\n")
    outF.close()

total_list = []
# for seed_idx in range(0,50, 5):
    # cont = read_txt("log3/margin1.0_seedoffset_{}_epoch_20_intvl0.025.txt".format(seed_idx))
    # total_list.extend(cont[:205])
# write_txt("log3/total_data.txt", total_list)

itv = 41
seed_idx = 10


data_raw = pd.read_csv("log3/total_data.txt", header=None)
data = data_raw

text_out = []
for i in range(0, itv, 1):
    foo = get_all_index(data, i, itv)
    argmin_idx = np.argmin(foo)
    mean = np.mean(foo)
    # print(round(i*0.025, 3), mean)
    text_out.append(str(round(i*0.025, 3)) + ", " + str(round(mean, 8) )  )
write_txt("log3/total50_mean.csv", text_out)

gt_best_mw = []

for seed in range(0, 50):
    foo = get_all_in_seed(data, seed, itv)
    argmin_idx = np.argmax(foo)
    gbm = round(argmin_idx * 0.025, 3)
    gt_best_mw.append("{}, {}, {}".format(seed, gbm, argmin_idx) )
    print(seed, gbm, argmin_idx)

write_txt("log3/gt_best_total50.csv", gt_best_mw)



