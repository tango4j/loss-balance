import torch
import numpy as np
import ipdb
from utils import *
from losses import ContrastiveLoss_mod as const_loss
import copy
import operator 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from pylab import logspace
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")

def cp(x):
    return copy.deepcopy(x)

def minmax_norm(row):
    return (row - np.min(row))/(np.max(row) - np.min(row))

def get_pdf(losses, bins=None, n_bins=20):
    '''
    Get pdf from histogram

    Input: loss values, 1-d numpy array

    Output pdf: pdf that is obtained by normalizing the histogram
    Output hist_raw: histogram that is not normalized by the total sum
    Output bins: bins used for histogram

    '''
    max_L = np.max(losses)
    max_h = (max_L + max_L/n_bins)
    if type(bins) != type(np.array(10)):
        bins = np.linspace(0, max_h, n_bins+1)
    hist_raw, bins = np.histogram(losses, bins)
    pdf = hist_raw/np.sum(hist_raw)
    assert len(pdf) == n_bins == len(bins)-1, "The length of given pdf, bins and n_bins is not consistent."
    return pdf, hist_raw, bins

def get_KL_div(p, q):
    '''
    Perform KL divergence for p and q.

    Input p: 1-d numpy array
    Input q: 1-d numpy array

    Output: a scalar value
    '''
    assert len(p) == len(q)
    eps = 1e-10 * np.ones_like(p)  # This is for avoiding div by zero.
    return np.sum(np.dot(p, np.log(( p + eps) / (q+ eps) ) ))

def run_pretrain_task(index_tup, loss_tup, batch_hist_list):
    '''
    A function that performs tasks at epoch -1

    Currently calculates KL div for every iteration

    Input index_tup:
    Input loss_tup:
    Input batch_hist_list:

    Output batch_pdf_cst:
    Output batch_pdf_ce:
    Output batch_hist_list:
    Output var_init_tup:

    '''
    epoch, batch_idx = index_tup
    iter_loss_cst, iter_loss_ce = loss_tup
    batch_pdf_cst, batch_pdf_ce, mw_batch_list, prev_weight = batch_hist_list

    pdf_cst, hist_cst, org_bins_cst = get_pdf(iter_loss_cst, bins=None)
    pdf_ce, hist_ce, org_bins_ce = get_pdf(iter_loss_ce, bins=None)
    
    if batch_idx >= 0:
        KL_val, max_KL_mw, prev_weight = 0, 0.5, 0.5
        org_mixed_pdf, mixed_hist_raw, org_mixed_bins = get_weighted_pdfs(iter_loss_cst, iter_loss_ce, mixed_bins=None)
        
        pdf_cst, hist_cst, new_bins_cst = get_pdf(iter_loss_cst, bins=org_mixed_bins)
        pdf_ce, hist_ce, new_bins_ce = get_pdf(iter_loss_ce, bins=org_mixed_bins)
        
        batch_pdf_cst[epoch][batch_idx] = ( pdf_cst, hist_cst, org_bins_cst, org_mixed_pdf, org_mixed_bins, (cp(epoch), cp(batch_idx)) )
        batch_pdf_ce[epoch][batch_idx] = ( pdf_ce, hist_ce, org_bins_ce, org_mixed_pdf, org_mixed_bins, (cp(epoch), cp(batch_idx)) )

    else:
        if batch_pdf_cst[epoch][batch_idx-1] != []:
            (_, org_hist_cst, org_bins_cst, org_mixed_pdf, org_mixed_bins, (ref_epoch, ref_batch_idx)) = batch_pdf_cst[epoch][batch_idx-1]
            (_, org_hist_ce, org_bins_ce, org_mixed_pdf, org_mixed_bins, (ref_epoch, ref_batch_idx)) = batch_pdf_ce[epoch][batch_idx-1]

        max_KL_mw, KL_val = get_max_KL_mw(iter_loss_cst, iter_loss_ce, org_mixed_bins, org_mixed_pdf)
        org_mixed_pdf, mixed_hist_raw, org_mixed_bins = get_weighted_pdfs(iter_loss_cst, iter_loss_ce, mixed_bins=None)
   
    batch_pdf_cst[epoch][batch_idx] = ( pdf_cst, hist_cst, org_bins_cst, org_mixed_pdf, org_mixed_bins, (cp(epoch), cp(batch_idx)) )
    batch_pdf_ce[epoch][batch_idx] = ( pdf_ce, hist_ce, org_bins_ce, org_mixed_pdf, org_mixed_bins, (cp(epoch), cp(batch_idx)) )
    
    batch_hist_list = [batch_pdf_cst, batch_pdf_ce, mw_batch_list, prev_weight]
    return batch_pdf_cst, batch_pdf_ce, batch_hist_list, (max_KL_mw, KL_val, prev_weight)


def run_epoch_0_task(index_tup, loss_tup, trInst):
    '''
    A function that performs tasks at the first epoch 0

    Currently calculates KL div for every iteration

    Input index_tup:
    Input loss_tup:
    Input batch_hist_list:

    Output batch_pdf_cst:
    Output batch_pdf_ce:
    Output batch_hist_list:
    Output var_init_tup:

    '''
    epoch, batch_idx = index_tup
    iter_loss_cst, iter_loss_ce = loss_tup
    # batch_pdf_cst, batch_pdf_ce, mw_batch_list, prev_weight = batch_hist_list

    pdf_cst, hist_cst, org_bins_cst = get_pdf(iter_loss_cst, bins=None)
    pdf_ce, hist_ce, org_bins_ce = get_pdf(iter_loss_ce, bins=None)
    
    lh_loss_cst, lh_loss_ce = get_lookahead_pdfs(index_tup, loss_tup, trInst)
    org_mixed_pdf, lh_hist_raw, org_mixed_bins = get_weighted_pdfs(lh_loss_cst, lh_loss_ce, mixed_bins=None)
    max_KL_mw, KL_val = get_max_KL_mw_lh(iter_loss_cst, iter_loss_ce, org_mixed_bins, org_mixed_pdf)

    trInst.batch_pdf_cst[epoch][batch_idx] = ( pdf_cst, hist_cst, org_bins_cst, org_mixed_pdf, org_mixed_bins, (cp(epoch), cp(batch_idx)) )
    trInst.batch_pdf_ce[epoch][batch_idx] = ( pdf_ce, hist_ce, org_bins_ce, org_mixed_pdf, org_mixed_bins, (cp(epoch), cp(batch_idx)) )
    
    # save_hist(iter_loss_cst, iter_loss_ce, epoch, batch_idx)
    
    # batch_hist_list = [batch_pdf_cst, batch_pdf_ce, mw_batch_list, prev_weight]
    var_init_tup =  (max_KL_mw, KL_val)
    return trInst, var_init_tup



def run_epoch_1_task(index_tup, loss_tup, trInst):
    # print("run_epoch_1_task:", total_samples)
    '''
    A function that performs tasks after epoch 1

    Input index_tup:
    Input loss_tup:
    Input batch_hist_list:
    Input total_samples:

    Output batch_pdf_cst:
    Output batch_pdf_ce:
    Output batch_hist_list:
    Output var_rest_tup:

    '''
     
    epoch, batch_idx = index_tup
    iter_loss_cst, iter_loss_ce = loss_tup
    # batch_pdf_cst, batch_pdf_ce, mw_batch_list, prev_weight = batch_hist_list

    prev_mixed_pdf_list = []
    # epoch_ceil = epoch
    epoch_ceil = 1


    for epoch_idx in range(0, epoch_ceil):
        (_, org_hist_cst, org_bins_cst, org_mixed_pdf, org_mixed_bins, (ref_epoch, ref_batch_idx)) = trInst.batch_pdf_cst[epoch_idx][batch_idx]
        (_, org_hist_ce,  org_bins_ce,  org_mixed_pdf, org_mixed_bins, (ref_epoch, ref_batch_idx)) = trInst.batch_pdf_ce[epoch_idx][batch_idx]
        prev_mixed_pdf_list.append(org_mixed_pdf)
    
    lh_loss_cst, lh_loss_ce = get_lookahead_pdfs(index_tup, loss_tup, trInst)
    lh_mixed_pdf, lh_hist_raw, lh_mixed_bins = get_weighted_pdfs(lh_loss_cst, lh_loss_ce, mixed_bins=org_mixed_bins)
    ### Get max_KL_mw from two loss vectors and ref_mixed_bins
    # max_KL_mw, KL_val = get_max_KL_mw(iter_loss_cst, iter_loss_ce, org_mixed_bins, org_mixed_pdf)
    # max_KL_mw, KL_val = get_max_KL_mw_lh(iter_loss_cst, iter_loss_ce, org_mixed_bins, org_mixed_pdf)
    # max_KL_mw, KL_val = get_max_KL_mw_from_list(iter_loss_cst, iter_loss_ce, org_mixed_bins, prev_mixed_pdf_list)
    max_KL_mw, KL_val = get_max_KL_mw_from_list_and_lh(iter_loss_cst, iter_loss_ce, org_mixed_bins, prev_mixed_pdf_list, lh_mixed_pdf)
    
    ### Get pdfs
    pdf_cst, hist_cst, new_bins_cst = get_pdf(iter_loss_cst, bins=org_bins_cst)
    pdf_ce, hist_ce, new_bins_ce = get_pdf(iter_loss_ce, bins=org_bins_ce)
    
    ### Get weighted pdfs for reference pdfs
    org_mixed_pdf, mixed_hist_raw, org_mixed_bins = get_weighted_pdfs(iter_loss_cst, iter_loss_ce, mixed_bins=org_mixed_bins)
    trInst.batch_pdf_cst[epoch][batch_idx] = (pdf_cst, hist_cst, org_bins_cst, org_mixed_pdf, org_mixed_bins, (epoch, batch_idx))
    trInst.batch_pdf_ce[epoch][batch_idx] = (pdf_ce, hist_ce, org_bins_ce, org_mixed_pdf, org_mixed_bins, (epoch, batch_idx)) 

    ### For the circulation of mw values
    # mv_mw_sum, kl_mw_sum, total_samples, prev_weight = mw_batch_list
    # batch_hist_list = [batch_pdf_cst, batch_pdf_ce, mw_batch_list, prev_weight]
    
    ### Compare reference pdf and current distribution
    # print("Ref {}-{}:".format(ref_epoch, ref_batch_idx), org_hist_cst,"\nCur {}-{}:".format(epoch, batch_idx), hist_cst)
    # print("Ref {}-{}:".format(ref_epoch, ref_batch_idx), org_hist_ce, "\nCur {}-{}:".format(epoch, batch_idx), hist_ce)
    
    save_hist(iter_loss_cst, iter_loss_ce, epoch, batch_idx)

    var_rest_tup = (max_KL_mw, KL_val)
    return trInst, var_rest_tup



def save_hist(iter_loss_cst, iter_loss_ce, epoch, batch_idx):
    path="/sam/home/inctrl/Dropbox/Special Pics/research_pics/loss_bal/test_e{}_i{}.png".format(epoch, batch_idx)
    # axis = logspace(-1e10,0, 10)
    axis = 100
    plt.hist2d(iter_loss_ce, iter_loss_cst, bins=(axis, axis), norm=mpl.colors.LogNorm() , cmap='BuGn', range=[ [0, 8.0 ], [0.0, 1.8]] )
    # plt.hist2d(iter_loss_ce, iter_loss_cst, bins=(axis, axis),  cmap='Blues')
    plt.xlabel("Cross Entropy")
    plt.ylabel("Constrasive Loss")
    plt.savefig(path)
    plt.cla()
    plt.clf()
    plt.close()

def tonp(x):
    return x.detach().cpu().numpy()

def get_feedback(target, distance, thres = 1.0):
    outputs_np = distance
    target_np = target[0]
    pred = outputs_np.ravel() < thres
    sample_feedback = list(pred == target_np)
    return sample_feedback

def get_lookahead_pdfs(index_tup, loss_tup, trInst):
    with torch.no_grad():
        (loss_fn, loss_fn_ce) = trInst.loss_fn_tup
        (outraw1, outraw2) = trInst.outputs
        lh = trInst.margin_LH
        lh = torch.tensor(lh).cuda()
        label1, label2 = trInst.label1, trInst.label2
        target = trInst.target
        target_float = trInst.target.float()
        target_rep= target_float.repeat(2,1).t()
        
        outraw_mean = 0.5 * (outraw1 + outraw2)
        outraw1_lh = target_rep *(lh*outraw_mean + (1-lh)*outraw1) + (1-target_rep)*(outraw1 + lh*(outraw1-outraw_mean)) 
        outraw2_lh = target_rep *(lh*outraw_mean + (1-lh)*outraw2) + (1-target_rep)*(outraw2 + lh*(outraw2-outraw_mean)) 
        # ipdb.set_trace()
        data_lh = trInst.data + (outraw1_lh, outraw2_lh)
        trInst.model.eval()
        output1, output2, score1, score2 = trInst.model(*data_lh)
        
        # loss_preprocessing_args = (output1, output2, score1, score2, trInst.label1, trInst.label2, trInst.target)
        outputs = (outraw1_lh, outraw2_lh)
        outputs_ce1 = (score1,)
        outputs_ce2 = (score2,)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs_cst = outputs
        if target is not None:
            target = (target,)
            loss_inputs_cst += target
        
        loss_inputs_ce1 = outputs_ce1
        loss_inputs_ce2 = outputs_ce2
        if label1 is not None and label2 is not None:
            loss_inputs_ce1 += (label1,) 
            loss_inputs_ce2 += (label2,)
        # loss_inputs_cst, loss_inputs_ce1, loss_inputs_ce2, _ = loss_input_process(*loss_preprocessing_args)
        
        loss_outputs, distance, losses_const = loss_fn(*loss_inputs_cst)
        loss_outputs_ce1, losses_ce1 = loss_fn_ce(*loss_inputs_ce1)
        loss_outputs_ce2, losses_ce2 = loss_fn_ce(*loss_inputs_ce2)
        if torch.sum(losses_const).detach().cpu().numpy() > np.sum(loss_tup[0]):
            print("Old CST loss", np.sum(loss_tup[0]), "New CST loss:", torch.sum(losses_const) )
        if torch.sum(losses_ce1+losses_ce2).detach().cpu().numpy() > np.sum(loss_tup[1]):
            print("Old CE_ loss", np.sum(loss_tup[1]), "New CE_ loss:", torch.sum(losses_ce1+losses_ce2) )
    # ipdb.set_trace()    
    return tonp(losses_const), tonp(losses_ce1 + losses_ce2)

def get_weighted_pdfs(losses1, losses2, mixed_bins=None, n_bins=500):
    '''
    Get weighted pdfs for the future use.
    This weights the two set loss values for "n_bins" of different weight values.

    Input losses1: 1-d numpy array of loss values
    Input losses2: 1-d numpy array of loss values
    Input mixed_bins: Sometimes you need to specify bins to calculate KL div

    Output mixed_pdf: dictionary that constains mixed pdf with weight "mw"
    Output mixed_hist_raw: Histogram before normalization
    Output mixed_bins: original histogram bins for the future use

    '''
    itv = 1.0/n_bins
    max_ceil = 1.0 + itv
    mixed_pdf, mixed_hist_raw, mixed_bins = {}, {}, {}
    for mw in np.arange(0, max_ceil, itv):
        loss_mix = mw * losses1 + (1-mw) * losses2
        mixed_pdf[mw], mixed_hist_raw[mw], mixed_bins[mw] = get_pdf(loss_mix, bins=mixed_bins)
    return mixed_pdf, mixed_hist_raw, mixed_bins

def get_KL_values_from_pdfs(p_pdf_list, q_pdf_list):
    '''
    Get KL vlaues for each mix weight "mw"
    p_pdf: target pdf
    q_pdf: refernece pdf

    '''
    # Input is two dictionaries containing pdf for each mw
    assert len(p_pdf_list.items()) == len(q_pdf_list.items()), "Two dictionaries have different size."
    KL_values_dict = {}
    for mw, p_pdf in p_pdf_list.items():
        q_pdf = q_pdf_list[mw]
        KL_values_dict[mw] = get_KL_div(p_pdf, q_pdf)
    return KL_values_dict

def get_KL_values_from_list_of_pdfs(p_pdf_list, q_pdf_list_from_epoches):
    '''
    Similar to get_KL_values_from_pdfs() but getting KL div values from multiple of ref q_pdfs
    Input p_pdf: target pdf
    Input q_pdf_list_from_epoches: a list contains multiple of q_pdfs
    Input q_pdf: refernece pdf
    '''
    # Input is two dictionaries containing pdf for each mw.
    assert len(p_pdf_list.items()) == len(q_pdf_list_from_epoches[0].items()), "Two dictionaries have different size."
    KL_values_dict = {}
    for epoch_idx, q_pdf_list in enumerate(q_pdf_list_from_epoches):
        for mw, p_pdf in p_pdf_list.items():
            q_pdf = q_pdf_list[mw]
            if mw not in KL_values_dict:
                KL_values_dict[mw] = get_KL_div(p_pdf, q_pdf)
            else:
                KL_values_dict[mw] += get_KL_div(p_pdf, q_pdf)
    return KL_values_dict

def get_KL_values_from_list_of_pdfs_and_lh(p_pdf_list, q_pdf_list_from_epoches, r_pdf_list):
    '''
    Similar to get_KL_values_from_pdfs() but getting KL div values from multiple of ref q_pdfs
    Input p_pdf: target pdf
    Input q_pdf_list_from_epoches: a list contains multiple of q_pdfs
    Input q_pdf: refernece pdf
    '''
    # Input is two dictionaries containing pdf for each mw.
    assert len(p_pdf_list.items()) == len(q_pdf_list_from_epoches[0].items()), "Two dictionaries have different size."
    KL_values_dict = {}
    for epoch_idx, q_pdf_list in enumerate(q_pdf_list_from_epoches):
        for mw, p_pdf in p_pdf_list.items():
            q_pdf = q_pdf_list[mw]
            r_pdf_lh = r_pdf_list[mw]
            if mw not in KL_values_dict:
                KL_values_dict[mw] = get_KL_div(p_pdf, q_pdf) - get_KL_div(r_pdf_lh, p_pdf)
            else:
                KL_values_dict[mw] += get_KL_div(p_pdf, q_pdf) - get_KL_div(r_pdf_lh, p_pdf)
    return KL_values_dict


def get_argmax_KL_dict(KL_values_dict):
    '''
    Argmax function for the dictionary input KL_values_dict
    '''
    assert len(KL_values_dict.items()) != 0, "KL_values_dict is empty."
    argmax_mw = max(KL_values_dict.items(), key=operator.itemgetter(1))[0]
    return argmax_mw, KL_values_dict[argmax_mw]

def get_argmin_KL_dict(KL_values_dict):
    '''
    Argmin function for the dictionary input KL_values_dict
    '''
    assert len(KL_values_dict.items()) != 0, "KL_values_dict is empty."
    argmin_mw = min(KL_values_dict.items(), key=operator.itemgetter(1))[0]
    return argmin_mw, KL_values_dict[argmin_mw]

def get_max_KL_mw(loss1, loss2, ref_mixed_bins, ref_mixed_pdf):
    '''
    High level function that returns "max_KL_mw" and "KL_val"
    '''
    new_mixed_pdf, _, _ = get_weighted_pdfs(loss1, loss2, ref_mixed_bins)
    KL_values_dict = get_KL_values_from_pdfs(new_mixed_pdf, ref_mixed_pdf)
    max_KL_mw, KL_val = get_argmax_KL_dict(KL_values_dict)
    return max_KL_mw, KL_val

def get_max_KL_mw_lh(loss1, loss2, ref_mixed_bins, ref_mixed_pdf):
    '''
    High level function that returns "max_KL_mw" and "KL_val"
    '''
    new_mixed_pdf, _, _ = get_weighted_pdfs(loss1, loss2, ref_mixed_bins)
    KL_values_dict = get_KL_values_from_pdfs(new_mixed_pdf, ref_mixed_pdf)
    min_KL_mw, KL_val = get_argmin_KL_dict(KL_values_dict)
    return min_KL_mw, KL_val

def get_max_KL_mw_from_list(loss1, loss2, ref_mixed_bins, ref_mixed_pdf_from_epoches):
    '''
    High level function that returns "max_KL_mw" and "KL_val"
    '''
    new_mixed_pdf, _, _ = get_weighted_pdfs(loss1, loss2, ref_mixed_bins)
    KL_values_dict = get_KL_values_from_list_of_pdfs(new_mixed_pdf, ref_mixed_pdf_from_epoches)
    max_KL_mw, KL_val = get_argmax_KL_dict(KL_values_dict)
    return max_KL_mw, KL_val

def get_max_KL_mw_from_list_and_lh(loss1, loss2, ref_mixed_bins, ref_mixed_pdf_from_epoches, ref_mixed_pdf_lh):
    '''
    High level function that returns "max_KL_mw" and "KL_val"
    '''
    new_mixed_pdf, _, _ = get_weighted_pdfs(loss1, loss2, ref_mixed_bins)
    KL_values_dict = get_KL_values_from_list_of_pdfs_and_lh(new_mixed_pdf, ref_mixed_pdf_from_epoches, ref_mixed_pdf_lh)
    max_KL_mw, KL_val = get_argmax_KL_dict(KL_values_dict)
    return max_KL_mw, KL_val


def define_vars_for_MW_est(length_of_data_loader, max_epoch=20, initial_weight=0.5):
    batch_pdf_cst = { i:{ x:[] for x in range(length_of_data_loader)   } for i in range(max_epoch) }
    batch_pdf_ce  = { i:{ x:[] for x in range(length_of_data_loader)   } for i in range(max_epoch) }
    mw_batch_list = [None]*4
    prev_weight = initial_weight
    batch_hist_list = [batch_pdf_cst, batch_pdf_ce, mw_batch_list, prev_weight]
    return batch_hist_list

class TrInst:
    def __init__(self,seed, loss_fn_tup, length_of_data_loader, initial_weight):
        self.seed = seed
       
        self.model = None
        self.data = None
        self.outputs = None
        self.target = None

        self.gpu = gpu 
        self.max_epoch = 20
        self.total_samples = 0
        
        self.mv_mw_sum = 0
        self.kl_mw_sum = 0
        
        self.cum_MV_weight = initial_weight
        self.cum_KL_weight = initial_weight
        
        self.batch_pdf_cst = { i:{ x:[] for x in range(length_of_data_loader)   } 
                               for i in range(self.max_epoch) }
        self.batch_pdf_ce  = { i:{ x:[] for x in range(length_of_data_loader)   } 
                               for i in range(self.max_epoch) }
        self.mw_batch_list = [None]*4

        self.initial_weight = initial_weight
        self.prev_weight = self.initial_weight

        self.metric_instances = None
        self.loss_fn_tup = loss_fn_tup
    
        self.margin_LH = 0.01

'''
Minimum Variance Method

'''
def min_var(X):
    '''
    Input: Should be M stacked rows M x N
    Output: length M list with M weights.
    '''
    om = np.ones((X.shape[0],1))
    cov_mat = np.cov(X) 
    inv_com = np.linalg.inv(cov_mat) 
    weight_mat = np.matmul(inv_com,om)/np.matmul(np.matmul(om.T,inv_com), om ) 
    weight_mat = weight_mat.T
    # weight_mat = np.expand_dims(ptf_m, axis = 0)
    return weight_mat[0]

def get_min_var_result(loss1, loss2):
    '''
    Perform normalization for minimum variance criterion
    Input loss1 and loss2:

    '''
    LossCst, LossCE = loss1/np.max(loss1), (loss2)/np.max(loss2)
    X = np.vstack((LossCst, LossCE))
    wm_norm = min_var(X)
    wm = wm_norm
    wm = [ wm[0]/np.max(loss1), wm[1]/np.max(loss2)]
    wm = [ wm[0]/sum(wm), wm[1]/sum(wm)]
    return wm 

def print_variables(trInst, max_KL_mw, min_var_mw, mix_weight, epoch, batch_idx, mode):
    # print("saving total_samples:", trInst.total_samples) 
    # print("mix_weight before torch.tensor: ", mix_weight)
    # print("{}-{} [seed {}] applied weight (mix_weight): {}".format(epoch, batch_idx, trInst.trInst.seed, mix_weight))
    # print("{}-{} [seed {}] Weight actual: w1:{:.4f} ".format(epoch, batch_idx, trInst.seed, wm[0]), "Cumulative: w1:{:.4f} ".format(trInst.cum_MV_weight))
    if mode == 'train':
        print("{}-{} [seed {}] Applied MW: {:.4f} Curr. KLMW:{:.4f} Curr. MVMW: {:.4f}".format(epoch, batch_idx, trInst.seed, mix_weight, max_KL_mw, min_var_mw[0]), "Cum. KLMW: {:.4f} MVMW: {:.4f} ".format(trInst.cum_KL_weight, trInst.cum_MV_weight))
    elif mode == 'test':
        print("VAL {}-{} [seed {}] Applied MW: >>[{:.4f}]<< Current KLMW:{:.4f} ".format(epoch, batch_idx, trInst.seed, mix_weight, max_KL_mw), "Cum. KLMW: {:.4f} ".format(trInst.cum_KL_weight))

    # print("{}-{} [seed {}] Applied MW: {:.4f} Current MVMW:{:.4f} ".format(epoch, batch_idx, trInst.seed, mix_weight, wm[0]), "Cum. MVMW: {:.4f} ".format(trInst.cum_MV_weight))
    # print("{}-{} [seed {}] max_KL_mw: {} KL_val: {} KL_mean:{}  cum_KL_weight: {}".format(epoch, batch_idx, trInst.seed, round(max_KL_mw, 4), round(KL_val,4), round(np.mean(KL_cum_list), 4), trInst.cum_KL_weight))


def loss_input_process(*args):
    output1, output2, score1, score2, label1, label2, target = args
    outputs = (output1, output2)
    outputs_ce1 = (score1,)
    outputs_ce2 = (score2,)

    if type(outputs) not in (tuple, list):
        outputs = (outputs,)

    loss_inputs_cst = outputs
    if target is not None:
        target = (target,)
        loss_inputs_cst += target
    
    loss_inputs_ce1 = outputs_ce1
    loss_inputs_ce2 = outputs_ce2
    if label1 is not None and label2 is not None:
        loss_inputs_ce1 += (label1,) 
        loss_inputs_ce2 += (label2,)
    outputs_tuple = (outputs, outputs_ce1, outputs_ce2)
    return loss_inputs_cst, loss_inputs_ce1, loss_inputs_ce2, outputs_tuple

def fit_siam(train_loader, val_loader, model, loss_fn_tup, optimizer, scheduler, n_epochs, cuda, log_interval, mix_weight, ATLW, metric_classes=[], seed=0, start_epoch=0):
    # model_org, optimizer_org, scheduler_org = model_org_pack
    for epoch in range(0, start_epoch):
        scheduler.step()
        # scheduler_org.step()
   
    batch_hist_list = define_vars_for_MW_est(len(train_loader), max_epoch=20, initial_weight=0.5)
    start_epoch = 0
    
    if ATLW: 
        mix_weight = 1.0
        mix_weight = 0.5

    trInst = TrInst(seed=seed, 
                    loss_fn_tup=loss_fn_tup,
                    length_of_data_loader=len(train_loader),
                    initial_weight=mix_weight)
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()
        
        # Train stage
        np.random.seed(seed)
        print("\nTraining... ")
        train_loss, vcel1, vcel2, metrics, mix_weight, batch_hist_list, mix_weight_list = train_siam_epoch(train_loader, epoch, model, loss_fn_tup, optimizer, cuda, log_interval, metric_classes, trInst, mix_weight, ATLW)
        message = '[seed: {} mixw: {:.4f}] Epoch: {}/{}. Train set: Const loss: {:.4f} CE-loss1 {:.4f} CE-loss2 {:.4f}'.format(trInst.seed, mix_weight, epoch + 1, n_epochs, train_loss, vcel1, vcel2)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print("\nTesting... ")
        val_loss, vcel1, vcel2, metrics = test_siam_epoch(val_loader, epoch, model, loss_fn_tup, cuda, metric_classes, trInst, ATLW)
        val_loss /= len(val_loader)

        message += '\n[seed: {} mixw: {:.4f}] Epoch: {}/{}. Validation set: Const loss: {:.4f} CE-loss1 {:.4f} CE-loss2 {:.4f}'.format(trInst.seed, mix_weight, epoch + 1, n_epochs, val_loss, vcel1, vcel2)
        write_var = "{}, {:.4f}, {}, {}, {:.5f}, {:.4f}, {:.4f}".format(trInst.seed, mix_weight, epoch + 1, n_epochs, val_loss, vcel1, vcel2)

        for metric in metrics:
            message += ' {}: {}'.format(metric.name(), metric.value())
            write_var += ', {}'.format(metric.value())

        print(message)
    return write_var, mix_weight, mix_weight_list

def train_siam_epoch(train_loader, epoch, model, loss_fn_tup, optimizer, cuda, log_interval, metric_classes, trInst, mix_weight, ATLW=0):
    metric_instances=[]
    for metric_class in metric_classes:
        metric_instance = metric_class()
        metric_instances.append(metric_instance)


    # model_org, optimizer_org, scheduler_org = model_org_pack
    # model_org.train()
    model.train()
    losses = []
    total_loss, ce_loss1, ce_loss2= 0, 0, 0
    
    loss_list=[ [], [] ]
    
    KL_cum_list = []
    org_mixed_bins, org_mixed_bins = [], []
    for batch_idx, (data, target, label1, label2) in enumerate(train_loader):
        
        iter_loss_list = [ [], [] ]
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
                label1 = label1.cuda()
                label2 = label2.cuda()
                mix_weight = torch.tensor(mix_weight).cuda()
                mix_weight.requires_grad = False


        # optimizer_org.zero_grad()
        optimizer.zero_grad()
        data_siam = data + (None, None)
        output1, output2, score1, score2 = model(*data_siam)
        # output1, output2 = model_org(*data)
        # for param_group in optimizer.param_groups:
            # print("============ LEARNING RATE:", param_group["step_size"])
        
        outputs = (output1, output2)
        outputs_ce1 = (score1,)
        outputs_ce2 = (score2,)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs_cst = outputs
        if target is not None:
            target = (target,)
            loss_inputs_cst += target
        
        loss_inputs_ce1 = outputs_ce1
        loss_inputs_ce2 = outputs_ce2
        if label1 is not None and label2 is not None:
            loss_inputs_ce1 += (label1,) 
            loss_inputs_ce2 += (label2,)
        # outputs_tuple = (outputs, outputs_ce1, outputs_ce2)
        # (outputs, outputs_ce1, outputs_ce2) = outputs_tuple

        ### Put data, target, output into trInst
        assert label1.shape[0] == score1.shape[0], "Label and score dimension should match."
        trInst.model, trInst.data, trInst.target, trInst.outputs, trInst.metric_instances, trInst.label1, trInst.label2= \
        (model, data, target[0], (output1, output2), metric_instances, label1, label2)
        
        loss_fn, loss_fn_ce = loss_fn_tup
        loss_outputs, distance, losses_const = loss_fn(*loss_inputs_cst)
        
        loss_outputs_ce1, losses_ce1 = loss_fn_ce(*loss_inputs_ce1)
        loss_outputs_ce2, losses_ce2 = loss_fn_ce(*loss_inputs_ce2)

        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses = [loss_outputs.item(), loss_outputs_ce1.item(), loss_outputs_ce2.item()]
        total_loss += loss_outputs.item() 
        ce_loss1 += loss_outputs_ce1.item() 
        ce_loss2 += loss_outputs_ce2.item()

        lcst, lce1, lce2 = (losses_const.detach().cpu().numpy(), losses_ce1.detach().cpu().numpy(), losses_ce2.detach().cpu().numpy()) 
        
        # ATLW=False
        if ATLW:
            iter_loss_cst = lcst
            iter_loss_ce =  lce1 + lce2

            min_var_mw =  get_min_var_result(iter_loss_cst, iter_loss_ce)
            if epoch == 0:
                trInst, var_init_tup = run_epoch_0_task((epoch, batch_idx), (iter_loss_cst, iter_loss_ce), trInst) 
                (max_KL_mw, KL_val) = var_init_tup
            
            else:
                trInst, var_rest_tup = run_epoch_1_task((epoch, batch_idx), (iter_loss_cst, iter_loss_ce), trInst)
                (max_KL_mw, KL_val) = var_rest_tup

            decay = 0 # decay=epoch for decaying values 

            if epoch >= 0 :
                trInst.mv_mw_sum +=  min_var_mw[0] * losses_const.shape[0] * np.exp(-1*decay)
                trInst.kl_mw_sum += max_KL_mw * losses_const.shape[0] * np.exp(-1*decay)
                trInst.total_samples += losses_const.shape[0] * np.exp(-1*decay)

                trInst.cum_MV_weight = round(float(trInst.mv_mw_sum/trInst.total_samples), 4)
                trInst.cum_KL_weight = round(float(trInst.kl_mw_sum/trInst.total_samples), 4)
            
                if epoch == 0:   
                    mix_weight = cp(trInst.initial_weight)
                else:
                    mix_weight = cp(trInst.prev_weight)

            print_variables(trInst, max_KL_mw, min_var_mw, mix_weight, epoch, batch_idx, mode='train')
            KL_cum_list.append(KL_val)
       
        # mix_weight = cp(trInst.prev_weight)
        mix_weight = torch.tensor(mix_weight).cuda().detach()
        mix_weight.requires_grad = False
        loss_mt = torch.mul(mix_weight, loss_outputs) + torch.mul(1-mix_weight, 1.0*(loss_outputs_ce1 +  loss_outputs_ce2))
        # loss_mt = loss_outputs

        loss_mt.backward()
        optimizer.step()

        target_source = [target, (label1,)]
        output_sources = [outputs, outputs_ce1]
        for k, metric_instance in enumerate(metric_instances):
            met_target, met_outputs = target_source[k], output_sources[k]
            metric_instance.eval_score(met_outputs, met_target, distance)
        
        if batch_idx % log_interval == 0:
            message = '[SIAM mixw: {:.1f}] Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(mix_weight.item(),
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metric_instances:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            # print(message)
            losses = []
   
    print("\n===Epoch Level mix weight : w1:{:.4f} w2:{:.4f}".format(mix_weight, (1-mix_weight)))

    if 0 <= epoch <= 500:
        delta_w = torch.abs(trInst.cum_KL_weight - mix_weight)
        sign_delta_w = torch.sign(trInst.cum_KL_weight - mix_weight)
        trInst.prev_weight = mix_weight + sign_delta_w * delta_w 
        print("delta_w:{:.4f} sign: {:.4f}".format(delta_w, sign_delta_w))
        print("====== mix_weight: {:.4f} trInst.cum_KL_weight: {:.4f} result prev_weight: {:.4f}".format(mix_weight, trInst.cum_KL_weight, trInst.prev_weight))
    
    trInst.total_samples = 0
    trInst.kl_mw_sum = 0
    trInst.mv_mw_sum = 0

    mix_weight_list = (str(trInst.seed), str(round(mix_weight.item(),4)), str(trInst.cum_KL_weight))
    total_loss /= (batch_idx + 1)
    ce_loss1 /= (batch_idx + 1)
    ce_loss2 /= (batch_idx + 1)
    return total_loss, ce_loss1, ce_loss2, metric_instances, mix_weight, trInst, mix_weight_list

def test_epoch(val_loader, model, loss_fn, cuda, metric_classes):
    gpu=0
    with torch.no_grad():
        # for metric in metrics:
            # metric.reset()
        metric_instances=[]
        for metric_class in metric_classes:
            metric_instance = metric_class()
            metric_instances.append(metric_instance)
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
        # for batch_idx, (data, target, _,_ ) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            # loss_outputs, distance, losses_const = loss_fn(*loss_inputs)
            if loss_fn.__class__.__name__ == 'CrossEntropy':
                loss_mean, loss_outputs = loss_fn(*loss_inputs)
                loss, distance = loss_mean, None
            else:
                loss_outputs, distance, losses_const = loss_fn(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            
            for k, metric_instance in enumerate(metric_instances):
                met_target, met_outputs = target, outputs
                metric_instance.eval_score(met_outputs, met_target, distance)

            val_loss += loss.item()

    return val_loss, metric_instances


def test_siam_epoch(val_loader, epoch, model, loss_fn_tup, cuda, metric_classes, trInst, ATLW):
    with torch.no_grad():
        metric_instances=[]
        for metric_class in metric_classes:
            metric_instance = metric_class()
            metric_instances.append(metric_instance)

        # model_org, optimizer_org, scheduler_org = model_org_pack
        
        model.eval()
        
        losses = []
        val_loss, total_loss, ce_loss1, ce_loss2= 0, 0, 0, 0
        
        loss_list=[ [], [] ]
        
        KL_cum_list = []
        for batch_idx, (data, target, label1, label2) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
                    label1 = label1.cuda()
                    label2 = label2.cuda()

            data_siam = data + (None, None)
            output1, output2, score1, score2 = model(*data_siam)
            # loss_preprocessing_args = (output1, output2, score1, score2, label1, label2, target)
            # loss_inputs_cst, loss_inputs_ce1, loss_inputs_ce2, outputs_tuple = loss_input_process(*loss_preprocessing_args)
            outputs = (output1, output2)
            outputs_ce1 = (score1,)
            outputs_ce2 = (score2,)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs_cst = outputs
            if target is not None:
                target = (target,)
                loss_inputs_cst += target
            
            loss_inputs_ce1 = outputs_ce1
            loss_inputs_ce2 = outputs_ce2
            if label1 is not None and label2 is not None:
                loss_inputs_ce1 += (label1,) 
                loss_inputs_ce2 += (label2,)
            
            ### Put data, target, output into trInst
            assert label1.shape[0] == score1.shape[0], "Label and score dimension should match."
            trInst.model, trInst.data, trInst.target, trInst.outputs, trInst.metric_instances, trInst.label1, trInst.label2= \
            (model, data, target, (output1, output2), metric_instances, label1, label2)

            loss_fn, loss_fn_ce = loss_fn_tup
            loss_outputs, distance, losses_const = loss_fn(*loss_inputs_cst)
            
            loss_outputs_ce1, losses_ce1 = loss_fn_ce(*loss_inputs_ce1)
            loss_outputs_ce2, losses_ce2 = loss_fn_ce(*loss_inputs_ce2)

            losses = [loss_outputs.item(), loss_outputs_ce1.item(), loss_outputs_ce2.item()]

            val_loss += loss_outputs.item() 
            ce_loss1 = loss_outputs_ce1.item() 
            ce_loss2 = loss_outputs_ce2.item()

            ### Loss processing 
            lcst, lce1, lce2 = (losses_const.detach().cpu().numpy(), losses_ce1.detach().cpu().numpy(), losses_ce2.detach().cpu().numpy()) 
            
            ATLW = False
            if ATLW:
                iter_loss_cst = lcst
                iter_loss_ce =  lce1 + lce2

                min_var_mw =  get_min_var_result(iter_loss_cst, iter_loss_ce)
                if epoch == 0:
                    trInst, var_init_tup = run_epoch_0_task((epoch, batch_idx), (iter_loss_cst, iter_loss_ce), trInst) 
                    (max_KL_mw, KL_val) = var_init_tup
                
                else:
                    trInst, var_rest_tup = run_epoch_1_task((epoch, batch_idx), (iter_loss_cst, iter_loss_ce), trInst)
                    (max_KL_mw, KL_val) = var_rest_tup
                decay = 0 # decay=epoch for decaying values 

                if epoch >= 0 :
                    trInst.mv_mw_sum +=  min_var_mw[0] * losses_const.shape[0] * np.exp(-1*decay)
                    trInst.kl_mw_sum += max_KL_mw * losses_const.shape[0] * np.exp(-1*decay)
                    trInst.total_samples += losses_const.shape[0] * np.exp(-1*decay)

                    trInst.cum_MV_weight = round(float(trInst.mv_mw_sum/trInst.total_samples), 4)
                    trInst.cum_KL_weight = round(float(trInst.kl_mw_sum/trInst.total_samples), 4)
                
                    if epoch == 0:   
                        mix_weight = cp(trInst.prev_weight)
                        # trInst.prev_weight = trInst.cum_KL_weight
                    else:
                        mix_weight = cp(trInst.prev_weight)
                print_variables(trInst, max_KL_mw, min_var_mw, mix_weight, epoch, batch_idx, mode='test')
                
                mix_weight = torch.tensor(mix_weight).cuda()
                mix_weight.requires_grad = False
                KL_cum_list.append(KL_val)

            target_source = [target, (label1,)]
            output_sources = [outputs, outputs_ce1]
            for k, metric_instance in enumerate(metric_instances):
                target, outputs = target_source[k], output_sources[k]
                metric_instance.eval_score(outputs, target, distance)
    
    # trInst.prev_weight = cp(trInst.cum_KL_weight)
    # ##:# trInst.prev_weight = cp(trInst.cum_MV_weight)
    # trInst.total_samples = 0
    # trInst.kl_mw_sum = 0
    # trInst.mv_mw_sum = 0

    return val_loss, ce_loss1, ce_loss2, metric_instances

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[], start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += ' {}: {}'.format(metric.name(), metric.value())
        
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += ' {}: {:.4f}'.format(metric.name(), metric.value())

        print(message)


def fit_org(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metric_classes=[], start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metric_classes)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += ' {}: {:.4f}'.format(metric.name(), metric.value())
        
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metric_classes)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += ' {}: {:.4f}'.format(metric.name(), metric.value())

        print(message)

def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metric_classes):
    # for metric in metrics:
        # metric.reset()

    gpu = 0
    metric_instances=[]
    for metric_class in metric_classes:
        metric_instance = metric_class()
        metric_instances.append(metric_instance)

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
    # for batch_idx, (data, target, _, _) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        if loss_fn.__class__.__name__ == 'CrossEntropy':
            loss_mean, loss_outputs = loss_fn(*loss_inputs)
            loss, distance = loss_mean, None
        else:
            loss_outputs, distance, losses_const = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

        losses.append(loss.item())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()


        for k, metric_instance in enumerate(metric_instances):
            met_target, met_outputs = target, outputs
            metric_instance.eval_score(met_outputs, met_target, distance)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metric_instances:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metric_instances

