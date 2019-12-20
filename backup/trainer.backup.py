import torch
import numpy as np
import ipdb
from utils import *
from losses import ContrastiveLoss_mod as const_loss
import copy
import operator 

def cp(x):
    return copy.deepcopy(x)

def minmax_norm(row):
    return (row - np.min(row))/(np.max(row) - np.min(row))
    # return (row - np.mean(row))/(np.std(row))

def get_pdf(losses, bins=None, n_bins=20):
    max_L = np.max(losses)
    # max_L = 20
    max_h = (max_L + max_L/n_bins)
    # max_h = 20
    if type(bins) != type(np.array(10)):
        bins = np.linspace(0, max_h, n_bins+1)
    hist_raw, bins = np.histogram(losses, bins)
    pdf = hist_raw/np.sum(hist_raw)
    if len(pdf) == n_bins == len(bins)-1:
        pass
    else:
        ipdb.set_trace()
    return pdf, hist_raw, bins

def get_KL_div(p, q):
    assert len(p) == len(q)
    eps = 1e-5 * np.ones_like(p)
    eps = 1e-10 * np.ones_like(p)
    return np.sum(np.dot(p, np.log(( p + eps) / (q+ eps) ) ))

def get_weighted_pdfs(losses1, losses2, mixed_bins=None, n_bins=100):
    itv = 1.0/n_bins
    max_ceil = 1.0 + itv
    mixed_pdf, mixed_hist_raw, mixed_bins = {}, {}, {}
    for mw in np.arange(0, max_ceil, itv):
        loss_mix = mw * losses1 + (1-mw) * losses2
        mixed_pdf[mw], mixed_hist_raw[mw], mixed_bins[mw] = get_pdf(loss_mix, bins=mixed_bins)
    return mixed_pdf, mixed_hist_raw, mixed_bins

def get_KL_values_from_pdfs(p_pdf_list, q_pdf_list):
    # Input is two dictionaries containing pdf for each mw
    assert len(p_pdf_list.items()) == len(q_pdf_list.items()), "Two dictionaries have different size."
    KL_values_dict = {}
    for mw, p_pdf in p_pdf_list.items():
        q_pdf = q_pdf_list[mw]
        # try:
        KL_values_dict[mw] = get_KL_div(p_pdf, q_pdf)
        # except:
            # ipdb.set_trace()
    return KL_values_dict

def get_KL_values_from_list_of_pdfs(p_pdf_list, q_pdf_list_from_epoches):
    # Input is two dictionaries containing pdf for each mw
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

def get_argmax_KL_dict(KL_values_dict):
    assert len(KL_values_dict.items()) != 0, "KL_values_dict is empty."
    argmax_mw = max(KL_values_dict.items(), key=operator.itemgetter(1))[0]
    return argmax_mw, KL_values_dict[argmax_mw]

'''
Minimum Variance Method

'''
def min_var(X):
    om = np.ones((X.shape[0],1))
    cov_mat = np.cov(X) 
    inv_com = np.linalg.inv(cov_mat) 
    p_mat = np.matmul(inv_com,om)/np.matmul(np.matmul(om.T,inv_com), om ) 
    p_mat = p_mat.T
    # p_mat = np.expand_dims(ptf_m, axis = 0)
    return p_mat[0]

def define_vars_for_MW_est(max_epoch=20):
    batch_pdf_cst = { i:{ x:[] for x in range(len(train_loader))   } for i in range(max_epoch) }
    batch_pdf_ce  = { i:{ x:[] for x in range(len(train_loader))   } for i in range(max_epoch) }
    mw_batch_list = [None] * len(train_loader)
    batch_hist_list = [batch_pdf_cst, batch_pdf_ce, mw_batch_list]
    return batch_hist_list

def fit_siam(gpu, train_loader, val_loader, model, loss_fn_tup, optimizer, scheduler, n_epochs, cuda, log_interval, mix_weight, MVLW, metric_classes=[], seed=0, start_epoch=0):
    for epoch in range(0, start_epoch):
        scheduler.step()

    batch_hist_list = define_vars_for_MW_est(max_epoch=20)
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()
         
        # Train stage
        np.random.seed(seed)
        print("\nTraining... ")
        train_loss, vcel1, vcel2, metrics, mix_weight, batch_hist_list, mix_weight_list = train_siam_epoch(gpu, train_loader, seed, epoch, model, loss_fn_tup, optimizer, cuda, log_interval, metric_classes, batch_hist_list, mix_weight, MVLW)
        message = '[seed: {} mixw: {:.5f}]  Epoch: {}/{}. Train set: Const loss: {:.4f} CE-loss1 {:.4f} CE-loss2 {:.4f} '.format(seed, mix_weight, epoch + 1, n_epochs, train_loss, vcel1, vcel2)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print("\n\n Testing... ")
        val_loss, vcel1, vcel2, metrics = test_siam_epoch(gpu, val_loader, epoch, model, loss_fn_tup, cuda, metric_classes, MVLW)
        val_loss /= len(val_loader)

        message += '\n[seed: {} mixw: {:.5f}] Epoch: {}/{}. Validation set: Const loss: {:.4f} CE-loss1 {:.4f} CE-loss2 {:.4f} '.format(seed, mix_weight, epoch + 1, n_epochs, val_loss, vcel1, vcel2)
        write_var = "{}, {:.4f}, {}, {}, {:.5f}, {:.4f}, {:.4f}".format(seed, mix_weight, epoch + 1, n_epochs, val_loss, vcel1, vcel2)

        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            write_var += ', {}'.format(metric.value())

        print(message)
    return write_var, mix_weight, mix_weight_list


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
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
            message += '\t{}: {}'.format(metric.name(), metric.value())
        
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def train_siam_epoch(gpu, train_loader, seed, epoch, model, loss_fn_tup, optimizer, cuda, log_interval, metric_classes, batch_hist_list, mix_weight, MVLW=0):
    # for metric in metrics:
        # metric.reset()
    metric_instances=[]
    for metric_class in metric_classes:
        metric_instance = metric_class()
        metric_instances.append(metric_instance)

    model.train()
    losses = []
    total_loss = 0
    
    loss_list=[ [], [] ]
    w_sum, w_sum_actual = 0, 0
    w1_sum, w2_sum =0 , 0
    total_sample = 0
    kl_mw_sum = 0
    KL_cum_list = []
    for batch_idx, (data, target, label1, label2) in enumerate(train_loader):

        iter_loss_list = [ [], [] ]
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda(gpu) for d in data)
            if target is not None:
                target = target.cuda(gpu)
                label1 = label1.cuda(gpu)
                label2 = label2.cuda(gpu)
                mix_weight = torch.tensor(mix_weight).cuda(gpu)


        optimizer.zero_grad()
        output1, output2, score1, score2 = model(*data)
        
        outputs = (output1, output2)
        outputs_ce1 = (score1,)
        outputs_ce2 = (score2,)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target
        
        loss_inputs_ce1 = outputs_ce1
        loss_inputs_ce2 = outputs_ce2
        if label1 is not None and label2 is not None:
            loss_inputs_ce1 += (label1,) 
            loss_inputs_ce2 += (label2,)
        
        loss_fn, loss_fn_ce = loss_fn_tup
        loss_outputs, distance, losses_const = loss_fn(*loss_inputs)
        
        loss_outputs_ce1, losses_ce1 = loss_fn_ce(*loss_inputs_ce1)
        loss_outputs_ce2, losses_ce2 = loss_fn_ce(*loss_inputs_ce2)

        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        # losses.append(loss.item())
        # total_loss += loss.item()
        # losses = []
        losses = [loss_outputs.item(), loss_outputs_ce1.item(), loss_outputs_ce2.item()]
        total_loss += loss_outputs.item() 
        ce_loss1 = loss_outputs_ce1.item() 
        ce_loss2 = loss_outputs_ce2.item()

        lcst, lce1, lce2 = (losses_const.detach().cpu().numpy(), losses_ce1.detach().cpu().numpy(), losses_ce2.detach().cpu().numpy()) 
        cov_ce1 = np.corrcoef(losses_const.detach().cpu().numpy(), losses_ce1.detach().cpu().numpy() +  losses_ce2.detach().cpu().numpy()) 
        cov_ce2 = np.corrcoef(losses_const.detach().cpu().numpy(), losses_ce2.detach().cpu().numpy()) 

        # print("cov_ce1", cov_ce1[1, 0], "cov_ce2:", cov_ce2[1,0])
        # csl_w, cel_w1, cel_w2 = 0.0, 1.0, 1.0

        # loss_mt = mix_weight * loss_outputs + (1-mix_weight)* (loss_outputs_ce1 +  loss_outputs_ce2)
        if epoch <= 20 and MVLW:
            # iter_LossCst = np.hstack(iter_loss_list[0])
            # iter_LossCE = np.hstack(iter_loss_list[1])
            iter_loss_cst = lcst
            iter_loss_ce =  lce1 + lce2
            # LossCst, LossCE = minmax_norm(iter_LossCst), minmax_norm(iter_LossCE) 
            LossCst, LossCE = lcst/np.max(lcst), (lce1 + lce2)/np.max(lce1+lce2)
            # LossCst, LossCE = iter_LossCst, iter_LossCE
            X = np.vstack((LossCst, LossCE))
            wm_norm = min_var(X)
            # wm_ = [ wm_norm[0]/np.max(iter_LossCst), wm_norm[1]/np.max(iter_LossCE)]
            # wm = [ wm_[0]/sum(wm_), wm_[1]/sum(wm_)]
            wm = wm_norm
            wm = [ wm[0]/np.max(lcst), wm[1]/np.max(lce1+lce2)]
            wm = [ wm[0]/sum(wm), wm[1]/sum(wm)]
            # print("max vals: l1 max:{:.4f} l2 max:{:.4f}".format(np.max(iter_loss_list), np.max(iter_LossCE)))
            # print("max vals: l1 max:{:.4f} l2 max:{:.4f}".format(np.max(lcst), np.max(lce1+lce2)))
            # print("Iter weight: w1:{:.4f} w2:{:.4f}".format(wm[0], wm[1]))
            # print("Weight actual: w1:{:.4f} w2:{:.4f}".format(wm_norm[0], wm_norm[1]))
            # mix_weight = wm[0]

            if epoch == 0:
                pdf_cst, hist_cst, org_bins_cst = get_pdf(iter_loss_cst, bins=None)
                pdf_ce, hist_ce, org_bins_ce = get_pdf(iter_loss_ce, bins=None)
                
                batch_pdf_cst, batch_pdf_ce, mw_batch_list = batch_hist_list
                
                if batch_idx == 0:
                    max_KL_mw = 0.5
                    KL_val = 0
                    prev_weight = 0.5
                    
                    org_mixed_pdf, mixed_hist_raw, org_mixed_bins = get_weighted_pdfs(iter_loss_cst, iter_loss_ce, mixed_bins=None)
                else:
                    new_mixed_pdf, _, _ = get_weighted_pdfs(iter_loss_cst, iter_loss_ce, org_mixed_bins)
                    KL_values_dict = get_KL_values_from_pdfs(new_mixed_pdf, org_mixed_pdf)
                    max_KL_mw, KL_val = get_argmax_KL_dict(KL_values_dict)

                    org_mixed_pdf, mixed_hist_raw, org_mixed_bins = get_weighted_pdfs(iter_loss_cst, iter_loss_ce, mixed_bins=None)
                    
                # batch_pdf_cst[batch_idx], batch_pdf_ce[batch_idx] = pdf_cst, pdf_ce
                # batch_hist_cst[batch_idx], batch_hist_ce[batch_idx] = hist_cst, hist_ce
                # batch_bins_cst[batch_idx], batch_bins_ce[batch_idx] = bins_cst, bins_ce
               
                # org_pdf_list_cst[batch_idx] = (org_mixed_pdf, mixed_bins)
                print(epoch, batch_idx, "Saving init parameters")

                batch_pdf_cst[epoch][batch_idx] = ( pdf_cst, hist_cst, org_bins_cst, org_mixed_pdf, org_mixed_bins, (cp(epoch), cp(batch_idx)) )
                batch_pdf_ce[epoch][batch_idx] = ( pdf_ce, hist_ce, org_bins_ce, org_mixed_pdf, org_mixed_bins, (cp(epoch), cp(batch_idx)) )
                print("batch_pdf_cst[{}][epoch]:".format(batch_idx), batch_pdf_cst[epoch][batch_idx][-1])
                # batch_hist_list = [batch_pdf_cst, batch_pdf_ce, mw_batch_list]
                print("-----------------0 batch_pdf_cst[batch_idx][epoch]:", batch_pdf_cst[epoch][batch_idx][-1])

            else:
                # batch_pdf_cst, batch_pdf_ce, batch_hist_cst, batch_hist_ce, batch_bins_cst, batch_bins_ce, org_pdf_list_cst, org_pdf_list_ce = batch_hist_list
                batch_pdf_cst, batch_pdf_ce, mw_batch_list = batch_hist_list

                ### Cumulative (all the prev epoches) KL div mode
                prev_mixed_pdf_list = []
                epoch_ceil = epoch
                for epoch_idx in range(0, epoch_ceil):
                    (_, org_hist_cst, org_bins_cst, org_mixed_pdf, org_mixed_bins, (ref_epoch, ref_batch_idx)) = batch_pdf_cst[epoch_idx][batch_idx]
                    (_, org_hist_ce, org_bins_ce, org_mixed_pdf, org_mixed_bins, (ref_epoch, ref_batch_idx)) = batch_pdf_ce[epoch_idx][batch_idx]
                    prev_mixed_pdf_list.append(org_mixed_pdf)
                    
                new_mixed_pdf, _, _ = get_weighted_pdfs(iter_loss_cst, iter_loss_ce, org_mixed_bins)
                KL_values_dict = get_KL_values_from_list_of_pdfs(new_mixed_pdf, prev_mixed_pdf_list)
                max_KL_mw, KL_val = get_argmax_KL_dict(KL_values_dict)
                ### ------------------------------------------------------
                

                ### Original (epoch 0) KL div mode --------

                # (_, org_hist_cst, org_bins_cst, org_mixed_pdf, org_mixed_bins, (ref_epoch, ref_batch_idx)) = batch_pdf_cst[0][batch_idx]
                # (_, org_hist_ce, org_bins_ce, org_mixed_pdf, org_mixed_bins, (ref_epoch, ref_batch_idx)) = batch_pdf_ce[0][batch_idx]
                # new_mixed_pdf, _, _ = get_weighted_pdfs(iter_loss_cst, iter_loss_ce, org_mixed_bins)
                # KL_values_dict = get_KL_values_from_pdfs(new_mixed_pdf, org_mixed_pdf)
                # max_KL_mw, KL_val = get_argmax_KL_dict(KL_values_dict)
                ### ------------------------------------------------------


                pdf_cst, hist_cst, new_bins_cst = get_pdf(iter_loss_cst, bins=org_bins_cst)
                pdf_ce, hist_ce, new_bins_ce = get_pdf(iter_loss_ce, bins=org_bins_ce)
                
                org_mixed_pdf, mixed_hist_raw, org_mixed_bins = get_weighted_pdfs(iter_loss_cst, iter_loss_ce, mixed_bins=None)
                batch_pdf_cst[epoch][batch_idx] = (pdf_cst, hist_cst, org_bins_cst, org_mixed_pdf, org_mixed_bins, (epoch, batch_idx))
                batch_pdf_ce[epoch][batch_idx] = (pdf_ce, hist_ce, org_bins_ce, org_mixed_pdf, org_mixed_bins, (epoch, batch_idx)) 
                # if batch_idx == 0:  
                w1_sum, kl_mw_sum, total_sample, prev_weight = mw_batch_list

               
                print("Org {}-{}:".format(ref_epoch, ref_batch_idx), org_hist_cst,"\nCur", hist_cst)
                print("Org {}-{}:".format(ref_epoch, ref_batch_idx), org_hist_ce, "\nCur:",hist_ce)
            print("w1_sum {} kl_mw_sum {} total_sample {}".format(w1_sum, kl_mw_sum, total_sample))
            # ipdb.set_trace()
            decay = epoch
            decay = 0
            w_sum += wm[0] * losses_const.shape[0]
            w1_sum +=  wm[0] * losses_const.shape[0] * np.exp(-1*decay)
            # w2_sum += wm[1] * losses_const.shape[0]
            kl_mw_sum += max_KL_mw * losses_const.shape[0] * np.exp(-1*decay)
            total_sample += losses_const.shape[0] * np.exp(-1*decay)
            # total_sample = len(train_loader)
            

            cum_MV_weight = round(float(w1_sum/total_sample), 4)
            cum_KL_weight = round(float(kl_mw_sum/total_sample), 4)
            if epoch == 0:   
                mix_weight = wm[0]
                mix_weight = max_KL_mw
                # mix_weight = 0.5
            else:
                # mix_weight = cum_KL_weight
                # mix_weight = cum_MV_weight
                # prev_weight = copy.deepcopy(cum_MV_weight)
                mix_weight = cp(prev_weight)
                # prev_weight = cp(cum_KL_weight)
            
            mw_batch_list = (w1_sum, kl_mw_sum, total_sample, cum_KL_weight)

            # ipdb.set_trace()
            mix_weight = torch.tensor(mix_weight).cuda(gpu)
            # print("Weight actual: w1:{:.4f} w2:{:.4f}".format(wm[0], wm[1]), "cumulative: w1:{:.4f} w2:{:.4f}".format(mix_weight, 1-mix_weight))
            print("{}-{} [seed {}] applied weight (mix_weight): {}".format(epoch, batch_idx, seed, mix_weight))
            print("{}-{} [seed {}] Weight actual: w1:{:.4f} ".format(epoch, batch_idx, seed, wm[0]), "Cumulative: w1:{:.4f} ".format(cum_MV_weight))
            # if epoch != 0:
            max_KL_mw = round(max_KL_mw, 4)
            KL_cum_list.append(KL_val)
            print("{}-{} [seed {}] max_KL_mw: {} KL_val: {} KL_mean:{}  cum_KL_weight: {}".format(epoch, batch_idx, seed, max_KL_mw, round(KL_val,4), round(np.mean(KL_cum_list), 4), cum_KL_weight))
       
        
        loss_mt = torch.mul(mix_weight, loss_outputs) + torch.mul(1-mix_weight, 1.0*(loss_outputs_ce1 +  loss_outputs_ce2))
        # loss_mt = loss_outputs_ce1  + loss_outputs_ce2

        # loss.backward()
        loss_mt.backward()
        optimizer.step()
        loss_list[0].extend(lcst) 
        loss_list[1].extend(lce1 + lce2)
        iter_loss_list[0].extend(lcst) 
        iter_loss_list[1].extend(lce1 + lce2)
        
        target_source = [target, (label1,)]
        output_sources = [outputs, outputs_ce1]
        for k, metric_instance in enumerate(metric_instances):
            met_target, met_outputs = target_source[k], output_sources[k]
            metric_instance.eval_score(met_outputs, met_target, loss_outputs, distance)
        
        if batch_idx % log_interval == 0:
            message = '[SIAM mixw: {:.1f}] Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(mix_weight.item(),
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metric_instances:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            # print(message)
            losses = []
        
        if epoch <= -1 and MVLW:
            # iter_LossCst = np.hstack(iter_loss_list[0])
            # iter_LossCE = np.hstack(iter_loss_list[1])
            iter_loss_cst = lcst
            iter_loss_ce =  lce1 + lce2
            # LossCst, LossCE = minmax_norm(iter_LossCst), minmax_norm(iter_LossCE) 
            LossCst, LossCE = lcst/np.max(lcst), (lce1 + lce2)/np.max(lce1+lce2)
            # LossCst, LossCE = iter_LossCst, iter_LossCE
            X = np.vstack((LossCst, LossCE))
            wm_norm = min_var(X)
            # wm_ = [ wm_norm[0]/np.max(iter_LossCst), wm_norm[1]/np.max(iter_LossCE)]
            # wm = [ wm_[0]/sum(wm_), wm_[1]/sum(wm_)]
            wm = wm_norm
            wm = [ wm[0]/np.max(lcst), wm[1]/np.max(lce1+lce2)]
            wm = [ wm[0]/sum(wm), wm[1]/sum(wm)]
            # print("max vals: l1 max:{:.4f} l2 max:{:.4f}".format(np.max(iter_loss_list), np.max(iter_LossCE)))
            # print("max vals: l1 max:{:.4f} l2 max:{:.4f}".format(np.max(lcst), np.max(lce1+lce2)))
            # print("Iter weight: w1:{:.4f} w2:{:.4f}".format(wm[0], wm[1]))
            print("Weight actual: w1:{:.4f} w2:{:.4f}".format(wm_norm[0], wm_norm[1]))
            # mix_weight = wm[0]
            w_sum += wm[0] * losses_const.shape[0]
            w1_sum += wm[0] * losses_const.shape[0]
            # w2_sum += wm[1] * losses_const.shape[0]
            mix_weight = wm[0]
        
    batch_hist_list = [batch_pdf_cst, batch_pdf_ce, mw_batch_list]
   
    if epoch == 0 and MVLW:
        
        # LossCst = np.hstack(loss_list[0])
        # LossCE = np.hstack(loss_list[1])

        # norm_LossCst, norm_LossCE = minmax_norm(LossCst), minmax_norm(LossCE) 
        # norm_LossCst, norm_LossCE = [ LossCst/np.max(LossCst), LossCE/np.max(LossCE)]
        # X = np.vstack((LossCst, LossCE))
        # X = np.vstack((norm_LossCst, norm_LossCE))
        # wm = min_var(X)
        # wr = copy.deepcopy(wm)
        # mix_weight = wm[0]
        # print("Weight: w1:{:.4f} w2:{:.4f}".format(wm[0], wm[1]))
        # w_sum = w_sum / len(train_loader.dataset)

        ###############
        # w1 = w1_sum / len(train_loader.dataset)
        # w2 = w2_sum / len(train_loader.dataset)
        # wm = [w1, w2]
        # mix_weight = w1


        # wm = [mix_weight, 1-mix_weight]
        # wm = [ wm[0]/np.max(LossCst), wm[1]/np.max(LossCE)]
        # wm = [ wm[0]/sum(wm), wm[1]/sum(wm)]
        # print("max vals: l1 max:{:.4f} l2 max:{:.4f}".format(np.max(LossCst), np.max(LossCE)))
        # print("===Epoch Level raw weight : w1:{:.4f} w2:{:.4f}".format(wr[0], wr[1]))
        # print("===Epoch Level mix weight : w1:{:.4f} w2:{:.4f}".format(wm[0], wm[1]))
        print("===Epoch Level mix weight : w1:{:.4f} w2:{:.4f}".format(mix_weight, (1-mix_weight)))

        # mix_weight = wm[0]
        
        mix_weight = torch.tensor(mix_weight).cuda(gpu)

    else:
        # mix_weight 
        pass

    # ipdb.set_trace()
    mix_weight_list = (str(seed), str(round(mix_weight.item(),4)), str(cum_KL_weight))
    total_loss /= (batch_idx + 1)
    return total_loss, ce_loss1, ce_loss2, metric_instances, mix_weight, batch_hist_list, mix_weight_list


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda(gpu) for d in data)
            if target is not None:
                target = target.cuda(gpu)


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics

def test_siam_epoch(gpu, val_loader, epoch, model, loss_fn_tup, cuda, metric_classes, MVLW):
    
    with torch.no_grad():
        metric_instances=[]
        for metric_class in metric_classes:
            metric_instance = metric_class()
            metric_instances.append(metric_instance)
        model.eval()
        val_loss = 0
        for batch_idx, (data, target, label1, label2) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda(gpu) for d in data)
                if target is not None:
                    target = target.cuda(gpu)
                    label1 = label1.cuda(gpu)
                    label2 = label2.cuda(gpu)

            outputs = model(*data)
            output1, output2, score1, score2 = model(*data)

            outputs = (output1, output2)
            outputs_ce1 = (score1,)
            outputs_ce2 = (score2,)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_inputs_ce1 = outputs_ce1
            loss_inputs_ce2 = outputs_ce2
            if label1 is not None and label2 is not None:
                loss_inputs_ce1 += (label1,) 
                loss_inputs_ce2 += (label2,)
            
            loss_fn, loss_fn_ce = loss_fn_tup
            loss_outputs, distance, losses_const = loss_fn(*loss_inputs)
            
            loss_outputs_ce1, losses_ce1 = loss_fn_ce(*loss_inputs_ce1)
            loss_outputs_ce2, losses_ce2 = loss_fn_ce(*loss_inputs_ce2)

            loss_outputs, distance, losses_const = loss_fn(*loss_inputs)
            # loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            # val_loss += loss.item()
            
            losses = [loss_outputs.item(), loss_outputs_ce1.item(), loss_outputs_ce2.item()]
            # val_loss = loss_outputs.item() + loss_outputs_ce1.item() + loss_outputs_ce2.item()
            val_loss += loss_outputs.item() 
            ce_loss1 = loss_outputs_ce1.item() 
            ce_loss2 = loss_outputs_ce2.item()

            # for metric in metrics:
                # metric(outputs, target, loss_outputs)
            # for metric_instance in metric_instances:
                # metric_instance.eval_score(outputs, target, loss_outputs, distance)
            target_source = [target, (label1,)]
            output_sources = [outputs, outputs_ce1]
            for k, metric_instance in enumerate(metric_instances):
                target, outputs = target_source[k], output_sources[k]
                metric_instance.eval_score(outputs, target, loss_outputs, distance)

    return val_loss, ce_loss1, ce_loss2, metric_instances


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda(gpu) for d in data)
                if target is not None:
                    target = target.cuda(gpu)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
