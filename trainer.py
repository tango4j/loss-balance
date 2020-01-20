import warnings
import matplotlib as mpl
from pylab import logspace
import torch
import numpy as np
import ipdb
from utils import *
from losses import ContrastiveLoss_mod as const_loss
import copy
import operator
import matplotlib
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def cp(x):
    return copy.deepcopy(x)


def minmax_norm(row):
    return (row - np.min(row))/(np.max(row) - np.min(row))

def save_hist(ter_loss_cst, iter_loss_ce, epoch, batch_idx):
    path = "/sam/home/inctrl/Dropbox/Special Pics/research_pics/loss_bal/test_e{}_i{}.png".format(
        epoch, batch_idx)
    # axis = logspace(-1e10,0, 10)
    axis = 100
    plt.hist2d(iter_loss_ce, iter_loss_cst, bins=(axis, axis),
               norm=mpl.colors.LogNorm(), cmap='BuGn', range=[[0, 8.0], [0.0, 1.8]])
    # plt.hist2d(iter_loss_ce, iter_loss_cst, bins=(axis, axis),  cmap='Blues')
    plt.xlabel("Cross Entropy")
    plt.ylabel("Constrasive Loss")
    plt.savefig(path)
    plt.cla()
    plt.clf()
    plt.close()

def writeAndSaveEpoch(write_var, vd, ATLW, mix_weight_list, margin, seed_range, n_epochs, interval, log_tag, write_list):
    ### Write the names of the columns 
    if len(write_list) == 0:
        col_names = "seed, mix_weight, epoch/n_epochs, val_loss, vcel1, vcel2, Siam_ACC, CE_ACC"
        write_list.append(col_names) 

    write_list.append(write_var)
    # mw_list.append(', '.join(mix_weight_list))
    
    exp_tag = getSaveTag(vd, ATLW, margin, seed_range, n_epochs, interval)
    write_txt("log_{}/{}.txt".format(log_tag, exp_tag), write_list)

    print("Logged Variables: ", write_var)
    return write_list


def tonp(x):
    return x.detach().cpu().numpy()
    

class KLInst:
    def __init__(self, ATLW, seed, loss_fn_tup, length_of_data_loader, n_epochs, initial_weight, embd_size, vd):
        self.ATLW = ATLW
        self.seed = seed

        self.model = None
        self.data = None
        self.outputs = None
        self.target = None
        self.metric_instances = None
        self.label1 = None
        self.label2 = None

        self.org_output = None
        self.org_label = None

        self.gpu = gpu
        self.max_epoch = n_epochs
        self.total_samples = 0

        self.mv_mw_sum = 0
        self.kl_mw_sum = 0
        self.ANAL_kl_mw_sum = 0

        self.cum_MV_weight = initial_weight
        self.cum_KL_weight = initial_weight
        self.ANAL_cum_KL_weight = initial_weight

        self.batch_pdf_cst = {i: {x: [] for x in range(length_of_data_loader)}
                              for i in range(self.max_epoch)}
        self.batch_pdf_ce = {i: {x: [] for x in range(length_of_data_loader)}
                             for i in range(self.max_epoch)}
        self.batch_loss = {i: {x: [] for x in range(length_of_data_loader)}
                              for i in range(self.max_epoch)}
        self.stacked_loss_mat = {i: {x: [] for x in range(length_of_data_loader)}
                              for i in range(self.max_epoch)}
        self.mw_batch_list = [None]*4

        self.initial_weight = initial_weight
        self.prev_weight = self.initial_weight

        self.loss_fn_tup = loss_fn_tup
        self.triplet_indices = None
        # self.margin_LH = 0.01
        self.margin_LH = 0.05

        self.embd_size = embd_size

        self.n_bins = 500
        self.pdf_bins = 20

        self.vd = vd


    def get_pdf(self, losses, bins=None):
        '''
        Get pdf from histogram

        Input: loss values, 1-d numpy array

        Output pdf: pdf that is obtained by normalizing the histogram
        Output hist_raw: histogram that is not normalized by the total sum
        Output bins: bins used for histogram

        '''
        max_L = np.max(losses)
        max_h = (max_L + max_L/self.pdf_bins)
        if type(bins) != type(np.array(10)):
            bins = np.linspace(0, max_h, self.pdf_bins+1)
        hist_raw, bins = np.histogram(losses, bins)
        pdf = hist_raw/np.sum(hist_raw)
        assert len(pdf) == self.pdf_bins == len(bins) - \
            1, "The length of given pdf, bins and self.pdf_bins is not consistent."
        return pdf, hist_raw, bins
    
    def get_KL_div(self, p, q):
        '''
        Perform KL divergence for p and q.

        Input p: 1-d numpy array
        Input q: 1-d numpy array

        Output: a scalar value
        '''
        assert len(p) == len(q)
        eps = 1e-10 * np.ones_like(p)  # This is for avoiding div by zero.
        return np.sum(np.dot(p, np.log((p + eps) / (q + eps))))
        # return np.sum(np.dot(q, np.log((q + eps) / (p + eps))))


    def run_epoch_0_task(self, index_tup, loss_tup, trInst):
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
        lh_loss_cst, lh_loss_ce = self.get_lookahead_pdfs(index_tup, loss_tup, trInst)

        if trInst.ATLW in ['kl', 'klan']:
            # batch_pdf_cst, batch_pdf_ce, mw_batch_list, prev_weight = batch_hist_list

            pdf_cst, hist_cst, org_bins_cst = self.get_pdf(iter_loss_cst, bins=trInst.pdf_bins)
            pdf_ce, hist_ce, org_bins_ce = self.get_pdf(iter_loss_ce, bins=trInst.pdf_bins)
            pdf_ce, hist_ce, org_bins_ce = self.get_pdf(iter_loss_ce, bins=trInst.pdf_bins)

            # assert lh_loss_cst.shape[0] == lh_loss_ce.shape[0], "Lookahead loss dimension mismatch."
            org_mixed_pdf, lh_hist_raw, org_mixed_bins = self.get_weighted_pdfs(
                lh_loss_cst, lh_loss_ce, mixed_bins=None)
            
            ### Get the arg-max mixed_weight: "max_KL_mw" and its value "KL_val"
            max_KL_mw, KL_val = self.get_max_KL_mw_lh(
                iter_loss_cst, iter_loss_ce, org_mixed_bins, org_mixed_pdf)

            trInst.batch_pdf_cst[epoch][batch_idx] = (
                pdf_cst, hist_cst, org_bins_cst, org_mixed_pdf, org_mixed_bins, (cp(epoch), cp(batch_idx)))
            trInst.batch_pdf_ce[epoch][batch_idx] = (
                pdf_ce, hist_ce, org_bins_ce, org_mixed_pdf, org_mixed_bins, (cp(epoch), cp(batch_idx)))
          
            ###############################################################################################
            ### Temporary 

            ANAL_max_KL_mw, ANAL_KL_val = self.get_ANAL_max_KL_mw_lh(loss_tup, (lh_loss_cst, lh_loss_ce), trInst)
            trInst.batch_loss[epoch][batch_idx] = loss_tup
            trInst.stacked_loss_mat[epoch][batch_idx] = loss_tup
            ##################################################3

        # elif trInst.ATLW == 'klan':
            # max_KL_mw, KL_val = self.get_ANAL_max_KL_mw_lh(loss_tup, (lh_loss_cst, lh_loss_ce), trInst)
            # trInst.batch_loss[epoch][batch_idx] = loss_tup
            # trInst.stacked_loss_mat[epoch][batch_idx] = loss_tup
            
            # pass

        var_init_tup = (max_KL_mw, KL_val)
        ANAL_var_init_tup = (ANAL_max_KL_mw, ANAL_KL_val)
        return trInst, var_init_tup, ANAL_var_init_tup


    def run_epoch_1_task(self, index_tup, loss_tup, trInst):
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
        lh_loss_cst, lh_loss_ce = self.get_lookahead_pdfs(index_tup, loss_tup, trInst)
        # batch_pdf_cst, batch_pdf_ce, mw_batch_list, prev_weight = batch_hist_list
        
        # epoch_ceil = epoch
        epoch_ceil = 1


        if trInst.ATLW in ['kl', 'klan']:
            prev_mixed_pdf_list = []

            for epoch_idx in range(0, epoch_ceil):
                (_, org_hist_cst, org_bins_cst, org_mixed_pdf, org_mixed_bins,
                 (ref_epoch, ref_batch_idx)) = trInst.batch_pdf_cst[epoch_idx][batch_idx]
                (_, org_hist_ce,  org_bins_ce,  org_mixed_pdf, org_mixed_bins,
                 (ref_epoch, ref_batch_idx)) = trInst.batch_pdf_ce[epoch_idx][batch_idx]
                prev_mixed_pdf_list.append(org_mixed_pdf)

            lh_mixed_pdf, lh_hist_raw, lh_mixed_bins = self.get_weighted_pdfs(
                lh_loss_cst, lh_loss_ce, mixed_bins=org_mixed_bins, n_bins=trInst.n_bins)
            ### Get max_KL_mw from two loss vectors and ref_mixed_bins
            # max_KL_mw, KL_val = self.get_max_KL_mw(iter_loss_cst, iter_loss_ce, org_mixed_bins, org_mixed_pdf)
            # max_KL_mw, KL_val = self.get_max_KL_mw_lh(iter_loss_cst, iter_loss_ce, org_mixed_bins, org_mixed_pdf)
            # max_KL_mw, KL_val = self.get_max_KL_mw_from_list(iter_loss_cst, iter_loss_ce, org_mixed_bins, prev_mixed_pdf_list)
            max_KL_mw, KL_val = self.get_max_KL_mw_from_list_and_lh(
                iter_loss_cst, iter_loss_ce, org_mixed_bins, prev_mixed_pdf_list, lh_mixed_pdf)

            ### Get pdfs
            pdf_cst, hist_cst, new_bins_cst = self.get_pdf(iter_loss_cst, bins=org_bins_cst)
            pdf_ce, hist_ce, new_bins_ce = self.get_pdf(iter_loss_ce, bins=org_bins_ce)

            ### Get weighted pdfs for reference pdfs
            org_mixed_pdf, mixed_hist_raw, org_mixed_bins = self.get_weighted_pdfs(
                iter_loss_cst, iter_loss_ce, mixed_bins=org_mixed_bins, n_bins=trInst.n_bins)
            trInst.batch_pdf_cst[epoch][batch_idx] = (
                pdf_cst, hist_cst, org_bins_cst, org_mixed_pdf, org_mixed_bins, (epoch, batch_idx))
            trInst.batch_pdf_ce[epoch][batch_idx] = (
                pdf_ce, hist_ce, org_bins_ce, org_mixed_pdf, org_mixed_bins, (epoch, batch_idx))
       
            ### Temporary klan mode
            trInst.batch_loss[epoch][batch_idx] = loss_tup
            
            # load_epoch = epoch-1
            load_epoch = 0
            ANAL_max_KL_mw, ANAL_KL_val = self.get_ANAL_max_KL_mw_from_list_and_lh(loss_tup, 
                                                                   (lh_loss_cst, lh_loss_ce), 
                                                                   trInst, 
                                                                   trInst.stacked_loss_mat[load_epoch][batch_idx])
            
            # stacked_loss1, stacked_loss2 = trInst.stacked_loss_mat[epoch-1][batch_idx] 
            # trInst.stacked_loss_mat[epoch][batch_idx] = (np.vstack((stacked_loss1, loss_tup[0])), 
                                                        # np.vstack((stacked_loss2, loss_tup[1])))

        #########################################################

        # elif trInst.ATLW == 'klan':
            
            # trInst.batch_loss[epoch][batch_idx] = loss_tup
            # max_KL_mw, KL_val = self.get_ANAL_max_KL_mw_from_list_and_lh(loss_tup, 
                                                                   # (lh_loss_cst, lh_loss_ce), 
                                                                   # trInst, 
                                                                   # trInst.stacked_loss_mat[epoch-1][batch_idx])
            
            # stacked_loss1, stacked_loss2 = trInst.stacked_loss_mat[epoch-1][batch_idx] 
            # trInst.stacked_loss_mat[epoch][batch_idx] = (np.vstack((stacked_loss1, loss_tup[0])), 
                                                         # np.vstack((stacked_loss2, loss_tup[1])))

            # ### Dimensionality Check
            # print("trInst.stacked_loss_mat[batch_idx]", trInst.stacked_loss_mat[epoch][batch_idx][0].shape, trInst.stacked_loss_mat[epoch][batch_idx][1].shape)

        ### Save the figure 
        # save_hist(iter_loss_cst, iter_loss_ce, epoch, batch_idx)

        var_rest_tup = (max_KL_mw, KL_val)
        ANAL_var_rest_tup = (ANAL_max_KL_mw, ANAL_KL_val)
        # print("max_KL_mw:{:.4f} ANAL_max_KL_mw: {:.4f}".format(max_KL_mw, ANAL_max_KL_mw))
        return trInst, var_rest_tup, ANAL_var_rest_tup

    def getTripletEmbeddings(self, ed, ti):
        return ed[ti[:,0]], ed[ti[:,1]], ed[ti[:,2]]

    def get_lookahead_pdfs(self, index_tup, loss_tup, trInst):
        '''
        This function creates an outlook embedding to estimate the loss distribution.
        '''
        with torch.no_grad():
            if trInst.model.model_name == "SiameseNet_ClassNet":
                (loss_fn, loss_fn_ce) = trInst.loss_fn_tup
            elif trInst.model.model_name == "Triplet_ClassNet":
                (loss_fn_ctrs, loss_fn_triplet, loss_fn_ce) = trInst.loss_fn_tup
                loss_fn = loss_fn_triplet
            
            (outraw1, outraw2) = trInst.outputs
            lh = trInst.margin_LH
            lh = torch.tensor(lh).cuda()
            label1, label2 = trInst.label1, trInst.label2
            target = trInst.target
            target_float = trInst.target.float()
            target_rep = target_float.repeat(trInst.embd_size, 1).t()

            outraw_mean = 0.5 * (outraw1 + outraw2)
            outraw1_lh = target_rep * (lh*outraw_mean + (1-lh)*outraw1) + \
                (1-target_rep)*(outraw1 + lh*(outraw1-outraw_mean))
            outraw2_lh = target_rep * (lh*outraw_mean + (1-lh)*outraw2) + \
                (1-target_rep)*(outraw2 + lh*(outraw2-outraw_mean))

            # data_lh = trInst.data + (outraw1_lh, outraw2_lh)
            trInst.model.eval()
            if trInst.model.model_name == "SiameseNet_ClassNet":
                data_lh = (None, None)  + (outraw1_lh, outraw2_lh)
                output1, output2, score1, score2 = trInst.model(*data_lh)

            elif trInst.model.model_name == "Triplet_ClassNet":
                ### Since we used "target" for calculating the lookahead embeddings,
                ### we need to calculate "triplet_len" amount of embeddings to calculate 
                ### CE losses. 
                tplt_len = int(outraw2_lh.shape[0]/2)
                ebd_pos, ebd_neg = outraw2_lh[:tplt_len], outraw2_lh[tplt_len:]
                data_lh = (None, None)  + (ebd_pos, ebd_neg)
                _, _, score1, score2 = trInst.model(*data_lh)

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
            
            if trInst.model.model_name == "SiameseNet_ClassNet":
                loss_outputs, distance, losses_vec = loss_fn(*loss_inputs_cst)
            elif trInst.model.model_name == 'Triplet_ClassNet':
                loss_inputs_triplet = (trInst.org_output, trInst.org_label)
                loss_tplt_outputs, triplet_len, losses_tplt_vec, embeddings, triplet_indices = \
                    loss_fn_triplet(loss_inputs_triplet[0], loss_inputs_triplet[1], pre_triplets=trInst.triplet_indices, lh_mode=True)
                loss_outputs, losses_vec = loss_tplt_outputs, losses_tplt_vec

            loss_outputs_ce1, losses_ce1 = loss_fn_ce(*loss_inputs_ce1)
            loss_outputs_ce2, losses_ce2 = loss_fn_ce(*loss_inputs_ce2)
            if torch.sum(losses_vec).detach().cpu().numpy() > np.sum(loss_tup[0]):
                print("Loss increased! Old CST loss", np.sum(
                    loss_tup[0]), "New CST loss:", torch.sum(losses_vec))
            if torch.sum(losses_ce1+losses_ce2).detach().cpu().numpy() > np.sum(loss_tup[1]):
                print("Loss increased! Old CE_ loss", np.sum(
                    loss_tup[1]), "New CE_ loss:", torch.sum(losses_ce1+losses_ce2))

        if losses_vec.shape[0] != losses_ce1.shape[0]:
            ipdb.set_trace()

        return tonp(losses_vec), tonp(losses_ce1 + losses_ce2)

    def get_ANAL_max_KL_mw_from_list_and_lh(self, current_losses, lh_losses, trInst, batch_stacked_loss):
        n_bins = trInst.n_bins
        itv = 1.0/n_bins
        max_ceil = 1.0 + itv
        kl_from_each_mw = {}
        curr_losses1, curr_losses2 = current_losses
        lh_losses1, lh_losses2 = lh_losses
       
        # ipdb.set_trace()
        stacked_loss1, stacked_loss2 = batch_stacked_loss
        for mw in np.arange(0, max_ceil, itv):
            stacked_loss_mix = mw * stacked_loss1 + (1-mw) * stacked_loss2
            curr_loss_mix = mw * curr_losses1 + (1-mw) * curr_losses2
            lh_loss_mix = mw * lh_losses1 + (1-mw) * lh_losses2
            
            cdim = 0 if len(stacked_loss_mix.shape)==1 else 1
            mean_stacked, mean_curr, mean_lh = np.mean(stacked_loss_mix, axis=cdim), np.mean(curr_loss_mix), np.mean(lh_loss_mix)
            std_stacked, std_curr, std_lh = np.std(stacked_loss_mix, axis=cdim), np.std(curr_loss_mix), np.std(lh_loss_mix)
            
            # if len(stacked_loss1.shape) >1:
                # ipdb.set_trace()

            # KL_curr_stacked = np.log(std_curr/std_stacked) + ( (std_stacked**2) + (mean_stacked-mean_curr)**2 )/(2*std_curr**2) - 0.5
            # KL_lh_curr      = np.log(std_curr/std_lh)      + ( (std_lh**2) + (mean_lh-mean_curr)**2 )/(2*std_curr**2) - 0.5
            KL_curr_stacked = np.log(std_stacked/std_curr) + ( (std_curr**2) + (mean_curr-mean_stacked)**2 )/(2*std_stacked**2) - 0.5
            KL_curr_lh      = np.log(std_lh/std_curr)      + ( (std_curr**2) + (mean_curr-mean_lh)**2 )/(2*std_lh**2) - 0.5
            kl_from_each_mw[mw] = KL_curr_lh - np.sum(KL_curr_stacked)
            # kl_from_each_mw[mw] =  - np.sum(KL_curr_stacked)
            # kl_from_each_mw[mw] = KL_curr_lh 

        argmin_mw = min(kl_from_each_mw.items(), key=operator.itemgetter(1))[0]
        return argmin_mw, kl_from_each_mw[argmin_mw]


    def get_ANAL_max_KL_mw_lh(self, current_losses, lh_losses, trInst):
        n_bins = trInst.n_bins
        itv = 1.0/n_bins
        max_ceil = 1.0 + itv
        kl_from_each_mw = {}
        curr_losses1, curr_losses2 = current_losses
        lh_losses1, lh_losses2 = lh_losses

        for mw in np.arange(0, max_ceil, itv):
            curr_loss_mix = mw * curr_losses1 + (1-mw) * curr_losses2
            lh_loss_mix = mw * lh_losses1 + (1-mw) * lh_losses2

            mean_curr, mean_lh = np.mean(curr_loss_mix), np.mean(lh_loss_mix)
            std_curr, std_lh = np.std(curr_loss_mix), np.std(lh_loss_mix)
            
            KL_curr_lh      = np.log(std_lh/std_curr)      + ( (std_curr**2) + (mean_curr-mean_lh)**2 )/(2*std_lh**2) - 0.5
            kl_from_each_mw[mw] = KL_curr_lh

        argmin_mw = min(kl_from_each_mw.items(), key=operator.itemgetter(1))[0]
        return argmin_mw, kl_from_each_mw[argmin_mw]

    def get_weighted_pdfs(self, losses1, losses2, mixed_bins=None, n_bins=500):
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
            mixed_pdf[mw], mixed_hist_raw[mw], mixed_bins[mw] = self.get_pdf(
                loss_mix, bins=mixed_bins)
        return mixed_pdf, mixed_hist_raw, mixed_bins


    def get_KL_values_from_pdfs(self, p_pdf_list, q_pdf_list):
        '''
        Get KL vlaues for each mix weight "mw"
        p_pdf: target pdf
        q_pdf: refernece pdf

        '''
        # Input is two dictionaries containing pdf for each mw
        assert len(p_pdf_list.items()) == len(q_pdf_list.items()
                                              ), "Two dictionaries have different size."
        KL_values_dict = {}
        for mw, p_pdf in p_pdf_list.items():
            q_pdf = q_pdf_list[mw]
            KL_values_dict[mw] = self.get_KL_div(p_pdf, q_pdf)
        return KL_values_dict


    def get_KL_values_from_list_of_pdfs(self, p_pdf_list, q_pdf_list_from_epoches):
        '''
        Similar to get_KL_values_from_pdfs() but getting KL div values from multiple of ref q_pdfs
        Input p_pdf: target pdf
        Input q_pdf_list_from_epoches: a list contains multiple of q_pdfs
        Input q_pdf: refernece pdf
        '''
        # Input is two dictionaries containing pdf for each mw.
        assert len(p_pdf_list.items()) == len(
            q_pdf_list_from_epoches[0].items()), "Two dictionaries have different size."
        KL_values_dict = {}
        for epoch_idx, q_pdf_list in enumerate(q_pdf_list_from_epoches):
            for mw, p_pdf in p_pdf_list.items():
                q_pdf = q_pdf_list[mw]
                if mw not in KL_values_dict:
                    KL_values_dict[mw] = self.get_KL_div(p_pdf, q_pdf)
                else:
                    KL_values_dict[mw] += self.get_KL_div(p_pdf, q_pdf)
        return KL_values_dict


    def get_KL_values_from_list_of_pdfs_and_lh(self, p_pdf_list, q_pdf_list_from_epoches, r_pdf_list):
        '''
        Similar to get_KL_values_from_pdfs() but getting KL div values from multiple of ref q_pdfs
        Input p_pdf: target pdf
        Input q_pdf_list_from_epoches: a list contains multiple of q_pdfs
        Input q_pdf: refernece pdf
        '''
        # Input is two dictionaries containing pdf for each mw.
        assert len(p_pdf_list.items()) == len(
            q_pdf_list_from_epoches[0].items()), "Two dictionaries have different size."
        KL_values_dict = {}
        for epoch_idx, q_pdf_list in enumerate(q_pdf_list_from_epoches):
            for mw, p_pdf in p_pdf_list.items():
                q_pdf = q_pdf_list[mw]
                r_pdf_lh = r_pdf_list[mw]
                if mw not in KL_values_dict:
                    KL_values_dict[mw] = self.get_KL_div(
                        p_pdf, q_pdf) - self.get_KL_div(r_pdf_lh, p_pdf)
                else:
                    KL_values_dict[mw] += self.get_KL_div(p_pdf,
                                                     q_pdf) - self.get_KL_div(r_pdf_lh, p_pdf)
        return KL_values_dict


    def get_argmax_KL_dict(self, KL_values_dict):
        '''
        Argmax function for the dictionary input KL_values_dict
        '''
        assert len(KL_values_dict.items()) != 0, "KL_values_dict is empty."
        argmax_mw = max(KL_values_dict.items(), key=operator.itemgetter(1))[0]
        return argmax_mw, KL_values_dict[argmax_mw]


    def get_argmin_KL_dict(self, KL_values_dict):
        '''
        Argmin function for the dictionary input KL_values_dict
        '''
        assert len(KL_values_dict.items()) != 0, "KL_values_dict is empty."
        argmin_mw = min(KL_values_dict.items(), key=operator.itemgetter(1))[0]
        return argmin_mw, KL_values_dict[argmin_mw]


    def get_max_KL_mw(self, loss1, loss2, ref_mixed_bins, ref_mixed_pdf):
        '''
        High level function that returns "max_KL_mw" and "KL_val"
        '''
        new_mixed_pdf, _, _ = self.get_weighted_pdfs(loss1, loss2, ref_mixed_bins, n_bins)
        KL_values_dict = self.get_KL_values_from_pdfs(new_mixed_pdf, ref_mixed_pdf)
        max_KL_mw, KL_val = self.get_argmax_KL_dict(KL_values_dict)
        return max_KL_mw, KL_val


    def get_max_KL_mw_lh(self, loss1, loss2, ref_mixed_bins, ref_mixed_pdf):
        '''
        High level function that returns "max_KL_mw" and "KL_val"
        '''
        new_mixed_pdf, _, _ = self.get_weighted_pdfs(loss1, loss2, ref_mixed_bins)
        KL_values_dict = self.get_KL_values_from_pdfs(new_mixed_pdf, ref_mixed_pdf)
        min_KL_mw, KL_val = self.get_argmin_KL_dict(KL_values_dict)
        return min_KL_mw, KL_val


    def get_max_KL_mw_from_list(self, loss1, loss2, ref_mixed_bins, ref_mixed_pdf_from_epoches):
        '''
        High level function that returns "max_KL_mw" and "KL_val"
        '''
        new_mixed_pdf, _, _ = self.get_weighted_pdfs(loss1, loss2, ref_mixed_bins)
        KL_values_dict = self.get_KL_values_from_list_of_pdfs(
            new_mixed_pdf, ref_mixed_pdf_from_epoches)
        max_KL_mw, KL_val = self.get_argmax_KL_dict(KL_values_dict)
        return max_KL_mw, KL_val


    def get_max_KL_mw_from_list_and_lh(self, loss1, loss2, ref_mixed_bins, ref_mixed_pdf_from_epoches, ref_mixed_pdf_lh):
        '''
        High level function that returns "max_KL_mw" and "KL_val"
        '''
        new_mixed_pdf, _, _ = self.get_weighted_pdfs(loss1, loss2, ref_mixed_bins)
        KL_values_dict = self.get_KL_values_from_list_of_pdfs_and_lh(
            new_mixed_pdf, ref_mixed_pdf_from_epoches, ref_mixed_pdf_lh)
        max_KL_mw, KL_val = self.get_argmax_KL_dict(KL_values_dict)
        return max_KL_mw, KL_val



class GNInst(KLInst):
    def __init__(self, seed, loss_fn_tup, alpha, T):
        super().__init__(seed, loss_fn_tup, alpha, T)
        self.seed = seed

        self.model = None
        self.data = None
        self.outputs = None
        self.target = None
        self.metric_instances = None
        self.label1 = None
        self.label2 = None
        
        self.n_w = len(loss_fn_tup) # The number of 
        self.T = 5
        self.loss_weights = nn.Parameter((self.T/self.n_w)*torch.ones(2, requires_grad=True, device='cuda'))
        self.alpha = alpha
        self.T = T

    
'''
Minimum Variance Method

'''


def min_var(X):
    '''
    Input: Should be M stacked rows M x N
    Output: length M list with M weights.
    '''
    om = np.ones((X.shape[0], 1))
    cov_mat = np.cov(X)
    inv_com = np.linalg.inv(cov_mat)
    weight_mat = np.matmul(inv_com, om)/np.matmul(np.matmul(om.T, inv_com), om)
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
    wm = [wm[0]/np.max(loss1), wm[1]/np.max(loss2)]
    wm = [wm[0]/sum(wm), wm[1]/sum(wm)]
    return wm


def print_variables(trInst, bLen, ANAL_max_KL_mw, max_KL_mw, min_var_mw, mix_weight, epoch, batch_idx, mode):
    if trInst.vd['loss_type'] == "Ctrst":
        tplt_txt = ""
    elif trInst.vd['loss_type'] == "Trplt":
        tplt_txt = "Triplet #:{}".format(len(trInst.label1))

    if 'kl' in trInst.ATLW:
        if mode == 'train':
            print("TRA {}-{}/{} [seed {}]  {} Applied MW: {:.4f} Curr. a-KL:{:.4f} KL:{:.4f} Curr. MV:{:.4f}".format(epoch+1, batch_idx+1, bLen, trInst.seed, tplt_txt, 
                mix_weight, ANAL_max_KL_mw, max_KL_mw, min_var_mw[0]), "Cum.a-KL: {:.4f} Cum.KL: {:.4f} MV: {:.4f} ".format(trInst.ANAL_cum_KL_weight, trInst.cum_KL_weight, trInst.cum_MV_weight))
        elif mode == 'test':
            print("VAL {}-{}/{} [seed {}]  Applied MW: >>[{:.4f}]<< Current a-KL:{:.4f} KL:{:.4f} ".format(epoch+1, batch_idx+1, bLen,
                trInst.seed, tplt_txt, mix_weight, ANAL_max_KL_mw, max_KL_mw), "Cum.a-KL: {:.4f}  Cum.KL: {:.4f} ".format(trInst.ANAL_cum_KL_weight, trInst.cum_KL_weight))
    
    elif 'na' in trInst.ATLW:
        print("{} {}-{}/{} [seed {}]  {} Applied MW: {:.4f}" .format(mode, epoch+1, batch_idx+1, bLen, trInst.seed, tplt_txt, mix_weight))


    # print("{}-{} [seed {}] Applied MW: {:.4f} Current MVMW:{:.4f} ".format(epoch, batch_idx, trInst.seed, mix_weight, wm[0]), "Cum. MVMW: {:.4f} ".format(trInst.cum_MV_weight))
    # print("{}-{} [seed {}] max_KL_mw: {} KL_val: {} KL_mean:{}  cum_KL_weight: {}".format(epoch, batch_idx, trInst.seed, round(max_KL_mw, 4), round(KL_val,4), round(np.mean(KL_cum_list), 4), trInst.cum_KL_weight))

def getTripletVars(trInst, model, data, cuda, metric_instances, loss_fn_triplet, output1, labels):
   
    ### Original data
    loss_inputs_triplet = (output1, labels)
    
    ### Calculate the triplet losse first
    loss_tplt_outputs, triplet_len, losses_tplt_vec, embeddings, triplet_indices = loss_fn_triplet(*loss_inputs_triplet)
    
    ### Get anchor, postivie, negative embeddings
    ebd_anc, ebd_pos, ebd_neg = trInst.getTripletEmbeddings(embeddings, triplet_indices)
    
    ### Get the softmax scores from ebd_pos and ebd_neg 
    data_tplt = (None, None, ebd_pos, ebd_neg)
    _, _, score_pos, score_neg = model(*data_tplt)
    
    ### The target_pns is for calculating Contrast accuracy
    ebd_anc = torch.cat((ebd_anc, ebd_anc), dim=0)
    ebd_pns = torch.cat((ebd_pos, ebd_neg), dim=0)  
    target_pns = torch.cat((torch.ones(triplet_len, dtype=torch.int), 
                            torch.zeros(triplet_len, dtype=torch.int)) ) 
   
    ### These labels are for training with CE loss
    ### For each triplet, two cross entropy losses are applied
    label_pos, label_neg  = labels[triplet_indices[:,1]], labels[triplet_indices[:,2]]
    if cuda:
        label_pos, label_neg = label_pos.cuda(), label_neg.cuda()
        score_pos, score_neg = score_pos.cuda(), score_neg.cuda()
        ebd_anc, ebd_pns, target_pns = ebd_anc.cuda(), ebd_pns.cuda(), target_pns.cuda()

    loss_inputs_cst = (ebd_anc, ebd_pns, target_pns)
    loss_inputs_ce1, loss_inputs_ce2= (score_pos, label_pos), (score_neg, label_neg)

    ### We do backpropagate with triplet loss only (NOT contrastive loss)
    loss_outputs, losses_vec = loss_tplt_outputs, losses_tplt_vec
    trInst.org_output, trInst.org_label, trInst.triplet_indices= output1, labels, triplet_indices
    trInst.model, trInst.data, trInst.target, trInst.outputs, trInst.metric_instances, trInst.label1, trInst.label2 = \
        (model, data, target_pns, (ebd_anc, ebd_pns), metric_instances, label_pos, label_neg)
    
    triplet_ebdscr = (target_pns, label_pos, ebd_anc, ebd_pns, score_pos)
    loss_out_tup = (loss_inputs_cst, loss_inputs_ce1, loss_inputs_ce2, loss_outputs, losses_vec)
    assert losses_tplt_vec.shape[0] == loss_inputs_ce1[0].shape[0] == loss_inputs_ce2[0].shape[0], "Loss dimension mismatch."
    return trInst, loss_out_tup, triplet_ebdscr


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

def define_vars_for_MW_est(length_of_data_loader, max_epoch=500, initial_weight=0.5):
    batch_pdf_cst = {i: {x: [] for x in range(
        length_of_data_loader)} for i in range(max_epoch)}
    batch_pdf_ce = {i: {x: [] for x in range(
        length_of_data_loader)} for i in range(max_epoch)}

    mw_batch_list = [None]*4
    prev_weight = initial_weight
    batch_hist_list = [batch_pdf_cst, batch_pdf_ce, mw_batch_list, prev_weight]
    return batch_hist_list



def fit_siam(train_loader, val_loader, model, loss_fn_tup, optimizer_func, scheduler, n_epochs, cuda, log_interval, metric_classes, vd, start_epoch=0):
    ### Assign the variables in the variables dict 
    ATLW, mix_weight, embd_size, seed, write_list = vd['ATLW'], vd['mix_weight'], vd['embd_size'], vd['seed'], vd['write_list']
    
    # ipdb.set_trace() 
    print("define_vars_for_MW_est got max_epoch:", n_epochs)
    batch_hist_list = define_vars_for_MW_est(
        len(train_loader), max_epoch=n_epochs, initial_weight=0.5)
    start_epoch = 0

    # Variable setup.
    if ATLW != 'na':
        mix_weight = 0.5

    if ATLW in ['kl', 'klan', 'klgd', 'na']:
        trInst = KLInst(ATLW=ATLW, 
                        seed=seed,
                        loss_fn_tup=loss_fn_tup,
                        length_of_data_loader=len(train_loader),
                        n_epochs=n_epochs,
                        initial_weight=mix_weight, 
                        embd_size=embd_size, 
                        vd=vd)
        train_function = train_siam_epoch
        optimizer = optimizer_func(model.parameters(), lr=1e-3)
        # optimizer_gd = torch.optim.Adam([trInst.loss_weights], lr=1e-3)

    elif ATLW == 'gn':
        T, alpha=1, 1.5
        trInst = GNInst(seed=seed,
                        loss_fn_tup=loss_fn_tup,
                        alpha=1.5, T=T)
        train_function = train_siam_gn_epoch
        if cuda:
            trInst.loss_weights = trInst.loss_weights.cuda()
        # optimizer = torch.optim.Adam(
            # [*model.parameters(), trInst.loss_weights], lr=1e-3)
        optimizer_W = torch.optim.Adam(
            [*model.parameters()], lr=1e-3)
        optimizer_gn = torch.optim.Adam(
            [trInst.loss_weights], lr=1e-2)
        optimizer = (optimizer_W, optimizer_gn)

    # Training loop.
    for epoch in range(start_epoch, n_epochs):
        # scheduler.step()

        # Train stage
        np.random.seed(seed)
        print("\nTraining... ")
        
        train_loss, vcel1, vcel2, metrics, mix_weight, batch_hist_list, mix_weight_list = train_function(
            train_loader, epoch, model, loss_fn_tup, optimizer, cuda, log_interval, metric_classes, trInst, mix_weight, ATLW)
        message = '[seed: {} mixw: {:.4f}] Epoch: {}/{}. Train set: Const loss: {:.4f} CE-loss1 {:.4f} CE-loss2 {:.4f}'.format(
            trInst.seed, mix_weight, epoch + 1, n_epochs, train_loss, vcel1, vcel2)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print("\nTesting... ")
        val_loss, vcel1, vcel2, metrics = test_siam_epoch(
            val_loader, epoch, model, loss_fn_tup, cuda, metric_classes, trInst, ATLW)
        try:
            val_loss /= len(val_loader)
        except:
            ipdb.set_trace()
        
        VARS = (trInst.seed, mix_weight, epoch + 1, n_epochs, val_loss, vcel1, vcel2)

        message += '\n[seed: {} mixw: {:.4f}] Epoch: {}/{}. Validation set: Const loss: {:.4f} CE-loss1 {:.4f} CE-loss2 {:.4f}'.format(*VARS)
            # trInst.seed, mix_weight, epoch + 1, n_epochs, val_loss, vcel1, vcel2)
        write_var = "{}, {:.4f}, {}, {:.5f}, {:.4f}, {:.4f}".format(
            trInst.seed, mix_weight, str(epoch + 1)+"/"+str(n_epochs), val_loss, vcel1, vcel2)

        for metric in metrics:
            message += ' {}: {}'.format(metric.name(), metric.value())
            write_var += ', {}'.format(metric.value())

        print(message)
        
        write_list = writeAndSaveEpoch(write_var, vd, ATLW, mix_weight_list, vd['margin'], vd['seed_range'], n_epochs, vd['interval'], vd['log_tag'], write_list)
    return write_var, write_list, mix_weight, mix_weight_list


def train_siam_gn_epoch(train_loader, epoch, model, loss_fn_tup, optimizer, cuda, log_interval, metric_classes, trInst, mix_weight, ATLW=0):

    if type(optimizer) == tuple:
        optimizer_W, optimizer_gn = optimizer
    
    metric_instances = []
    for metric_class in metric_classes:
        metric_instance = metric_class()
        metric_instances.append(metric_instance)

    model.train()
    losses = []
    total_loss, ce_loss1, ce_loss2 = 0, 0, 0

    loss_list = [[], []]

    KL_cum_list = []
    org_mixed_bins, org_mixed_bins = [], []
    for batch_idx, (data, target, labels) in enumerate(train_loader):
        
        if model.model_name == 'SiameseNet_ClassNet':
            label1, label2 = labels
        elif model.model_name == 'Triplet_ClassNet':
            label1, label2, label3 = labels

        iter_loss_list = [[], []]
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
                # trInst.loss_weights = trInst.loss_weights.cuda()

        data_siam = data + (None, None)
        output1, output2, score1, score2 = model(*data_siam)

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

        # Put data, target, output into trInst
        assert label1.shape[0] == score1.shape[0], "Label and score dimension should match."
        trInst.model, trInst.data, trInst.target, trInst.outputs, trInst.metric_instances, trInst.label1, trInst.label2 = \
            (model, data, target[0], (output1, output2),
             metric_instances, label1, label2)

        loss_fn, loss_fn_ce = loss_fn_tup
        loss_outputs, distance, losses_vec = loss_fn(*loss_inputs_cst)

        loss_outputs_ce1, losses_ce1 = loss_fn_ce(*loss_inputs_ce1)
        loss_outputs_ce2, losses_ce2 = loss_fn_ce(*loss_inputs_ce2)

        loss = loss_outputs[0] if type(loss_outputs) in (
            tuple, list) else loss_outputs
        losses = [loss_outputs.item(), loss_outputs_ce1.item(),
                  loss_outputs_ce2.item()]
        total_loss += loss_outputs.item()
        ce_loss1 += loss_outputs_ce1.item()
        ce_loss2 += loss_outputs_ce2.item()
        
        if ATLW == 'gn':
            iter_loss_cst, iter_loss_ce = loss_outputs, loss_outputs_ce1+loss_outputs_ce2

            task_losses = [iter_loss_cst, iter_loss_ce]
            task_losses = torch.stack(task_losses)

            # get the sum of weighted losses
            weighted_losses = trInst.loss_weights * task_losses
            
            total_weighted_loss = weighted_losses.sum()

            optimizer_W.zero_grad()
            optimizer_gn.zero_grad()

            # compute and retain gradients
            total_weighted_loss.backward(retain_graph=True)

            # GRADNORM - learn the loss_weights for each tasks gradients

            # zero the w_i(t) gradients since we want to update the weights using gradnorm loss
            # if trInst.loss_weights.grad != None:
                # trInst.loss_weights.grad = 0.0 * trInst.loss_weights.data
            trInst.loss_weights.grad = 0.0 * trInst.loss_weights.grad

            W = list(trInst.model.parameters())
            G_w_t_L2norm_list = []

            ### Compute Gw(t) and r_i(t) 
            for w_i, L_i in zip(trInst.loss_weights, task_losses):
                # gradient of L_i(t) w.r.t. W
                gLgW = torch.autograd.grad(L_i, W, retain_graph=True)

                # G^{(i)}_W(t)
                G_w_t_L2norm_list.append(torch.norm(w_i * gLgW[0]))

            G_w_t_L2norm_list = torch.stack(G_w_t_L2norm_list)

            # set L(0)
            # if using log(C) init, remove these two lines
            if batch_idx == 0 and epoch == 0:
                trInst.initial_losses = task_losses.detach()


            bar_G_w_t = G_w_t_L2norm_list.mean()
            with torch.no_grad():

                # loss ratios \curl{L}(t)
                curl_L_i_t = task_losses / trInst.initial_losses

                # inverse training rate r(t)
                r_t = curl_L_i_t / curl_L_i_t.mean()

                # Constant term 
                constant_term = bar_G_w_t * (r_t ** trInst.alpha)

            # write out the gradnorm loss L_grad and set the weight gradients
            L_grad = (G_w_t_L2norm_list - constant_term).abs().sum()
            trInst.loss_weights.grad = torch.autograd.grad(L_grad, trInst.loss_weights)[0]

            # apply gradient descent
            # print(f"BEFORE STEP trInst.loss_weights.data {trInst.loss_weights.data, trInst.loss_weights.grad} ")
            optimizer_W.step()
            optimizer_gn.step()
            # print(f"AFTER  STEP trInst.loss_weights.data {trInst.loss_weights.data, trInst.loss_weights.grad} ")

            # renormalize the gradient weights
            with torch.no_grad():

                normalize_coeff = len(trInst.loss_weights) / trInst.loss_weights.sum()
                trInst.loss_weights.data = trInst.loss_weights.data * normalize_coeff

        mix_weight = trInst.loss_weights.detach().cpu().numpy()[0]
        target_source = [target, (label1,)]
        output_sources = [outputs, outputs_ce1]

        for k, metric_instance in enumerate(metric_instances):
            met_target, met_outputs = target_source[k], output_sources[k]
            metric_instance.eval_score(met_outputs, met_target, distance)

        if batch_idx % log_interval == 0:
            message = '[SIAM mixw: {:.1f}] Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(mix_weight.item(),
                                                                                          batch_idx *
                                                                                          len(data[0]), len(
                                                                                              train_loader.dataset),
                                                                                          100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metric_instances:
                message += ' {}: {}'.format(metric.name(), metric.value())

            # print(message)
            losses = []
    
    print("\n===Epoch Level mix weight : w1:{:.4f} w2:{:.4f}".format(
        mix_weight, (trInst.T-mix_weight)))


    mix_weight_list = (str(trInst.seed), str(
        round(mix_weight.item(), 4)), str(trInst.loss_weights))
    total_loss /= (batch_idx + 1)
    ce_loss1 /= (batch_idx + 1)
    ce_loss2 /= (batch_idx + 1)
    return total_loss, ce_loss1, ce_loss2, metric_instances, mix_weight, trInst, mix_weight_list

def train_siam_epoch(train_loader, epoch, model, loss_fn_tup, optimizer, cuda, log_interval, metric_classes, trInst, mix_weight, ATLW=0):
    metric_instances = []
    
    for metric_class in metric_classes:
        metric_instance = metric_class()
        metric_instances.append(metric_instance)

    model.train()
    losses = []
    total_loss, ce_loss1, ce_loss2 = 0, 0, 0

    loss_list = [[], []]

    KL_cum_list = []
    org_mixed_bins, org_mixed_bins = [], []
    
    # for batch_idx, (data, target, labels) in enumerate(train_loader):
    for batch_idx, data_target_labels in enumerate(train_loader):
        # if batch_idx == 1:
            # break
        if model.model_name == 'SiameseNet_ClassNet':
            data, target, labels = data_target_labels
            label1, label2 = labels
            target = target if len(target) > 0 else None
        if model.model_name == 'Triplet_ClassNet':
            data, labels = data_target_labels
            target = labels[0]
            label1, label2 = labels, labels


        iter_loss_list = [[], []]

        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                if model.model_name == 'SiameseNet_ClassNet':
                    target, label1, label2 = target.cuda(), label1.cuda(), label2.cuda()
                
                if model.model_name == 'Triplet_ClassNet':
                    labels = labels.cuda()

                mix_weight = torch.tensor(mix_weight).cuda()
                mix_weight.requires_grad = False

        optimizer.zero_grad()

        if model.model_name == 'SiameseNet_ClassNet':
            data_siam = data + (None, None)
            output1, output2, score1, score2 = model(*data_siam)
        if model.model_name == 'Triplet_ClassNet':
            data_tplt = (data[0], None) + (None, None)
            output1, output2, score1, score2 = model(*data_tplt)
            # output1, output2, None, score1, score2, None = model(*data_tplt)

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

        # Put data, target, output into trInst
        assert label1.shape[0] == score1.shape[0], "Label and score dimension should match."

        if model.model_name == 'SiameseNet_ClassNet':
            loss_fn, loss_fn_ce = loss_fn_tup
            loss_outputs, distance, losses_vec = loss_fn(*loss_inputs_cst)
            loss_outputs_ce1, losses_ce1 = loss_fn_ce(*loss_inputs_ce1)
            loss_outputs_ce2, losses_ce2 = loss_fn_ce(*loss_inputs_ce2)
            trInst.model, trInst.data, trInst.target, trInst.outputs, trInst.metric_instances, trInst.label1, trInst.label2 = \
                (model, data, target[0], outputs,
                 metric_instances, label1, label2)

        if model.model_name == 'Triplet_ClassNet':
            loss_fn, loss_fn_triplet, loss_fn_ce = loss_fn_tup

            trInst, loss_out_tup, triplet_ebdscr = getTripletVars(trInst, model, data, cuda, metric_instances, loss_fn_triplet, output1, labels)
            loss_inputs_cst, loss_inputs_ce1, loss_inputs_ce2, loss_outputs, losses_vec = loss_out_tup
            
            ### Calculate the losses 
            loss_ctrs_outputs, distance, losses_ctrs_vec = loss_fn(*loss_inputs_cst) # For monitoring purpose, no backprop
            loss_outputs_ce1, losses_ce1 = loss_fn_ce(*loss_inputs_ce1)
            loss_outputs_ce2, losses_ce2 = loss_fn_ce(*loss_inputs_ce2)


        loss = loss_outputs[0] if type(loss_outputs) in (
            tuple, list) else loss_outputs
        losses = [loss_outputs.item(), 
                  loss_outputs_ce1.item(),
                  loss_outputs_ce2.item()]

        total_loss += loss_outputs.item()
        ce_loss1 += loss_outputs_ce1.item()
        ce_loss2 += loss_outputs_ce2.item()

        lcst, lce1, lce2 = (losses_vec.detach().cpu().numpy(), 
                            losses_ce1.detach().cpu().numpy(), 
                            losses_ce2.detach().cpu().numpy())
            
        iter_loss_cst, iter_loss_ce = lcst, lce1 + lce2
        
        if ATLW in ['kl', 'klan']:

            min_var_mw = get_min_var_result(iter_loss_cst, iter_loss_ce)
            assert iter_loss_cst.shape[0] == iter_loss_ce.shape[0]

            if epoch == 0:
                trInst, var_init_tup, ANAL_var_init_tup = trInst.run_epoch_0_task(
                    (epoch, batch_idx), (iter_loss_cst, iter_loss_ce), trInst)
                (max_KL_mw, KL_val) = var_init_tup
                (ANAL_max_KL_mw, ANAL_KL_val) = ANAL_var_init_tup

            else:
                trInst, var_rest_tup, ANAL_var_rest_tup = trInst.run_epoch_1_task(
                    (epoch, batch_idx), (iter_loss_cst, iter_loss_ce), trInst)
                (max_KL_mw, KL_val) = var_rest_tup
                (ANAL_max_KL_mw, ANAL_KL_val) = ANAL_var_rest_tup

            decay = 0  # decay=epoch for decaying values
            
            trInst.mv_mw_sum += min_var_mw[0] * \
                data[0].shape[0] * np.exp(-1*decay)
            trInst.kl_mw_sum += max_KL_mw * \
                data[0].shape[0] * np.exp(-1*decay)
            trInst.ANAL_kl_mw_sum += ANAL_max_KL_mw * \
                data[0].shape[0] * np.exp(-1*decay)
            trInst.total_samples += data[0].shape[0] * \
                np.exp(-1*decay)

            trInst.cum_MV_weight = round(
                float(trInst.mv_mw_sum/trInst.total_samples), 4)
            trInst.cum_KL_weight = round(
                float(trInst.kl_mw_sum/trInst.total_samples), 4)
            trInst.ANAL_cum_KL_weight = round(
                float(trInst.ANAL_kl_mw_sum/trInst.total_samples), 4)

            if epoch == 0:
                mix_weight = cp(trInst.initial_weight)
            else:
                mix_weight = cp(trInst.prev_weight)

            print_variables(trInst, len(train_loader), ANAL_max_KL_mw, max_KL_mw, min_var_mw,
                            mix_weight, epoch, batch_idx, mode='train')
            KL_cum_list.append(KL_val)

        ### END of if ATLW == 'kl':
        if ATLW == 'na':
            ANAL_max_KL_mw, max_KL_mw, min_var_mw = mix_weight, mix_weight, mix_weight
            print_variables(trInst, len(train_loader), ANAL_max_KL_mw, max_KL_mw, min_var_mw,
                            mix_weight, epoch, batch_idx, mode='train')

        if ATLW == 'kl':
            mix_weight = trInst.cum_KL_weight
        if ATLW == 'klan':
            mix_weight = trInst.ANAL_cum_KL_weight 

        mix_weight = torch.tensor(mix_weight).cuda().detach()
        mix_weight.requires_grad = False
        loss_mt = torch.mul(mix_weight, loss_outputs) + torch.mul(1 - mix_weight, 1.0*(loss_outputs_ce1 + loss_outputs_ce2))
        loss_mt.backward()
        optimizer.step()


        if model.model_name == 'Triplet_ClassNet':
            target_pns, label_pos, ebd_anc, ebd_pns, score_pos = triplet_ebdscr
            target, label1, outputs, outputs_ce1= (target_pns,), label_pos, (ebd_anc, ebd_pns), (score_pos,)

        ### START of display section
        target_source = [target, (label1,)]
        output_sources = [outputs, outputs_ce1]

        for k, metric_instance in enumerate(metric_instances):
            met_target, met_outputs = target_source[k], output_sources[k]
            metric_instance.eval_score(met_outputs, met_target, distance)

        if batch_idx % log_interval == 0:
            message = '[SIAM mixw: {:.1f}] Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(mix_weight.item(),
                                                                                          batch_idx *
                                                                                          len(data[0]), 
                                                                                          len(train_loader.dataset),
                                                                                          100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metric_instances:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            # print(message)
            losses = []

    print("\n===Epoch Level mix weight : w1:{:.4f} w2:{:.4f}".format(
        mix_weight, (1-mix_weight)))

    if 0 <= epoch <= 500:
        delta_w = torch.abs(trInst.cum_KL_weight - mix_weight)
        sign_delta_w = torch.sign(trInst.cum_KL_weight - mix_weight)
        trInst.prev_weight = mix_weight

        ### If you want to apply the weight of the previous epoch
        # if ATLW == 'kl':
            # trInst.prev_weight = trInst.cum_KL_weight
        # if ATLW == 'klan':
            # trInst.prev_weight = trInst.ANAL_cum_KL_weight 
        # print("delta_w:{:.4f} sign: {:.4f}".format(delta_w, sign_delta_w))
        print("====== mix_weight: {:.4f} trInst.cum_KL_weight: {:.4f} trInst.ANAL_cum_KL_weight: {:.4f} result prev_weight: {:.4f}".format(
            mix_weight, trInst.cum_KL_weight, trInst.ANAL_cum_KL_weight, trInst.prev_weight))
    
    ### Initialize all the score and sum
    trInst.total_samples = 0
    trInst.kl_mw_sum = 0
    trInst.ANAL_kl_mw_sum = 0
    trInst.mv_mw_sum = 0

    mix_weight_list = (str(trInst.seed), str(
        round(mix_weight.item(), 4)), str(trInst.cum_KL_weight))
    total_loss /= (batch_idx + 1)
    ce_loss1 /= (batch_idx + 1)
    ce_loss2 /= (batch_idx + 1)
    return total_loss, ce_loss1, ce_loss2, metric_instances, mix_weight, trInst, mix_weight_list


def test_epoch(val_loader, model, loss_fn, cuda, metric_classes):
    gpu = 0
    with torch.no_grad():
        # for metric in metrics:
            # metric.reset()
        metric_instances = []
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
            

            if loss_fn.__class__.__name__ == 'CrossEntropy':
                loss_mean, loss_outputs = loss_fn(*loss_inputs)
                loss, distance = loss_mean, None
            else:
                loss_outputs, distance, losses_vec = loss_fn(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (
                    tuple, list) else loss_outputs

            for k, metric_instance in enumerate(metric_instances):
                met_target, met_outputs = target, outputs
                metric_instance.eval_score(met_outputs, met_target, distance)

            val_loss += loss.item()

    return val_loss, metric_instances


def test_siam_epoch(val_loader, epoch, model, loss_fn_tup, cuda, metric_classes, trInst, ATLW):
    with torch.no_grad():
        metric_instances = []
        for metric_class in metric_classes:
            metric_instance = metric_class()
            metric_instances.append(metric_instance)

        # model_org, optimizer_org, scheduler_org = model_org_pack

        model.eval()

        losses = []
        val_loss, total_loss, ce_loss1, ce_loss2 = 0, 0, 0, 0

        loss_list = [[], []]

        KL_cum_list = []
        for batch_idx, data_target_labels in enumerate(val_loader):
            
            if model.model_name == 'SiameseNet_ClassNet':
                data, target, labels = data_target_labels
                label1, label2 = labels
                target = target if len(target) > 0 else None

            if model.model_name == 'Triplet_ClassNet':
                data, labels = data_target_labels
                target = (labels[0])
                label1, label2 = labels, labels

            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    if model.model_name == 'SiameseNet_ClassNet':
                        target, label1, label2 = target.cuda(), label1.cuda(), label2.cuda()
                    
                    if model.model_name == 'Triplet_ClassNet':
                        labels = labels.cuda()

            if model.model_name == 'SiameseNet_ClassNet':
                data_siam = data + (None, None)
                output1, output2, score1, score2 = model(*data_siam)
            if model.model_name == 'Triplet_ClassNet':
                data_tplt = (data[0], None) + (None, None)
                output1, output2, score1, score2 = model(*data_tplt)

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

            # Put data, target, output into trInst
            assert label1.shape[0] == score1.shape[0], "Label and score dimension should match."
            trInst.model, trInst.data, trInst.target, trInst.outputs, trInst.metric_instances, trInst.label1, trInst.label2 = \
                (model, data, target, (output1, output2),
                 metric_instances, label1, label2)

            # loss_fn, loss_fn_ce = loss_fn_tup
            # loss_outputs, distance, losses_vec = loss_fn(*loss_inputs_cst)

            if model.model_name == 'SiameseNet_ClassNet':
                loss_fn, loss_fn_ce = loss_fn_tup
                loss_outputs, distance, losses_vec = loss_fn(*loss_inputs_cst)
                loss_outputs_ce1, losses_ce1 = loss_fn_ce(*loss_inputs_ce1)
                loss_outputs_ce2, losses_ce2 = loss_fn_ce(*loss_inputs_ce2)
                trInst.model, trInst.data, trInst.target, trInst.outputs, trInst.metric_instances, trInst.label1, trInst.label2 = \
                    (model, data, target[0], outputs,
                     metric_instances, label1, label2)

            if model.model_name == 'Triplet_ClassNet':
                loss_fn, loss_fn_triplet, loss_fn_ce = loss_fn_tup

                trInst, loss_out_tup, triplet_ebdscr = getTripletVars(trInst, model, data, cuda, metric_instances, loss_fn_triplet, output1, labels)
                loss_inputs_cst, loss_inputs_ce1, loss_inputs_ce2, loss_outputs, losses_vec = loss_out_tup
                
                ### Calculate the losses 
                loss_ctrs_outputs, distance, losses_ctrs_vec = loss_fn(*loss_inputs_cst) # For monitoring purpose, no backprop
                loss_outputs_ce1, losses_ce1 = loss_fn_ce(*loss_inputs_ce1)
                loss_outputs_ce2, losses_ce2 = loss_fn_ce(*loss_inputs_ce2)

   ######################################

            loss_outputs_ce1, losses_ce1 = loss_fn_ce(*loss_inputs_ce1)
            loss_outputs_ce2, losses_ce2 = loss_fn_ce(*loss_inputs_ce2)

            losses = [loss_outputs.item(), loss_outputs_ce1.item(),
                      loss_outputs_ce2.item()]

            val_loss += loss_outputs.item()
            ce_loss1 = loss_outputs_ce1.item()
            ce_loss2 = loss_outputs_ce2.item()

            # Loss processing
            lcst, lce1, lce2 = (losses_vec.detach().cpu().numpy(
            ), losses_ce1.detach().cpu().numpy(), losses_ce2.detach().cpu().numpy())

            if model.model_name == 'Triplet_ClassNet':
                target_pns, label_pos, ebd_anc, ebd_pns, score_pos = triplet_ebdscr
                target, label1, outputs, outputs_ce1= (target_pns,), label_pos, (ebd_anc, ebd_pns), (score_pos,)
                target = (target_pns,)
                label1 = label_pos
                outputs = (ebd_anc, ebd_pns)
                outputs_ce1 = (score_pos,)

            target_source = [target, (label1,)]
            output_sources = [outputs, outputs_ce1]
            for k, metric_instance in enumerate(metric_instances):
                target, outputs = target_source[k], output_sources[k]
                metric_instance.eval_score(outputs, target, distance)

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
        train_loss, metrics = train_epoch(
            train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(
            epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += ' {}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(
            val_loader, model, loss_fn, cuda, metrics)
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
        train_loss, metrics = train_epoch(
            train_loader, model, loss_fn, optimizer, cuda, log_interval, metric_classes)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(
            epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += ' {}: {:.4f}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(
            val_loader, model, loss_fn, cuda, metric_classes)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += ' {}: {:.4f}'.format(metric.name(), metric.value())

        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metric_classes):

    gpu = 0
    metric_instances = []
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
            loss_outputs, distance, losses_vec = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (
                tuple, list) else loss_outputs

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
