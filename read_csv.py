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
fn_list = ["log_grid_search/ATLW-na_margin1.0_seedRange1-10__intvl0.05_EmbeddingNet_MNIST.txt",
           "log_grid_search/ATLW-na_margin1.0_seedRange1-10__intvl0.05_EmbeddingNet_FashionMNIST.txt",
           "log_grid_search/ATLW-na_margin1.0_seedoffset_0__intvl0.05_EmbeddingNetVGG11_CIFAR10.txt",
           "log_grid_search/ATLW-na_margin1.0_seedoffset_0__intvl0.05_EmbeddingNetVGG11_CIFAR100.txt"]


data_raw = pd.read_csv(fn, header=None)

legend = list(data_raw.loc[0].to_numpy())
data = data_raw[1:]
n_epochs = int(data.loc[1, :].to_list()[2].split('/')[1])

text_out = []
start_line = 1
itv = data.shape[0]-start_line
seed_idx = 10
ldict = {name.strip():i for i,name in enumerate(legend)}

# ipdb.set_trace()
### mixing weights 
mw_vars = np.sort([ float(x) for x in set(data.loc[:,1].to_list()) ])
md = {mw:i for i,mw in enumerate(mw_vars)}

### List and sort all the seeds 
seed_vars = np.sort([ int(x) for x in set(data.loc[:,0].to_list()) ])
sd = {seed:i for i,seed in enumerate(seed_vars)}
ed = {epoch:i for i,epoch in enumerate(range(1, n_epochs+1))}
seed_start = np.min(seed_vars)
n_vars, n_seeds = mw_vars.shape[0], seed_vars.shape[0]

idata = np.zeros((n_seeds, n_vars, n_epochs))

for sdx, seed in enumerate(range(seed_start, seed_start + n_seeds )):
    print("indexing seed {}".format(seed))
    # idata[seed] = {}
    for mdx, mw in enumerate(mw_vars):
        # idata[seed][mw] = {}

        for edx, epoch in enumerate(range(1, 1+n_epochs)):
            # print(data.loc[sdx*n_seeds*n_epochs + mdx*n_epochs + epoch].to_list()[ldict['Siam_ACC']])
            idata[sdx, mdx, edx] = float(data.loc[sdx*n_vars*n_epochs + mdx*n_epochs + epoch].to_list()[ldict['Siam_ACC']])

### Example: seed 2 mw 0.05 epoch 
print("seed 2 mw 0.05 epoch:", idata[sd[2], md[0.05], ed[1]] )



# for i in range(start_line, itv, 1):
    # foo = get_all_index(data, i, itv)
    # argmin_idx = np.argmin(foo)
    # ipdb.set_trace()
    # mean = np.mean(foo)
    # # print(round(i*0.025, 3), mean)
    # text_out.append(str(round(i*0.025, 3)) + ", " + str(round(mean, 8) )  )
# write_txt("log3/total50_mean.csv", text_out)

# gt_best_mw = []

# for seed in range(0, 50):
    # foo = get_all_in_seed(data, seed, itv)
    # argmin_idx = np.argmax(foo)
    # gbm = round(argmin_idx * 0.025, 3)
    # gt_best_mw.append("{}, {}, {}".format(seed, gbm, argmin_idx) )
    # print(seed, gbm, argmin_idx)

# write_txt("log3/gt_best_total50.csv", gt_best_mw)



