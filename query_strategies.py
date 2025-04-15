import numpy as np
import torch
import numpy as np
from utils import lab_conv
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
import math

def random_sampling(args, unlabeledloader, Len_labeled_ind_train, model, knownclass):
    model.eval()
    queryIndex = []
    labelArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (_, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((queryIndex, labelArr)).T
    np.random.shuffle(tmp_data)
    tmp_data = tmp_data.T
    queryIndex = tmp_data[0][:args.query_batch]
    labelArr = tmp_data[1]
    queryLabelArr = tmp_data[1][:args.query_batch]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def eaoa_sampling(args, unlabeledloader, Len_labeled_ind_train, model, model_ID, knownclass, use_gpu, query, trainloader_ID, trainloader_ID_w_OOD):
    model.eval()
    model_ID.eval()

    if trainloader_ID_w_OOD == None:
        eval_trainloader = trainloader_ID
    else:
        eval_trainloader = trainloader_ID_w_OOD

    labelArr_all = []
    if args.dataset == 'tinyimagenet':
        feat_all = torch.zeros([1, 4 * 128]).cuda()
    else:
        feat_all = torch.zeros([1, 128]).cuda()
    for batch_idx, (index, (data, labels)) in enumerate(eval_trainloader):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        _, features = model(data)
        labelArr_all += list(np.array(labels.cpu().data))
        feat_all = torch.cat([feat_all, features.data],0)
    feat_all = feat_all[1:]
    feat_all = F.normalize(feat_all, dim=1)

    queryIndex = []
    labelArr = []
    predArr = []
    uncertaintyArr = []
    aleatoricUncArr = []
    precision, recall = 0, 0
    if args.dataset == 'tinyimagenet':
        feat_unlab = torch.zeros([1, 4 * 128]).cuda()
    else:
        feat_unlab = torch.zeros([1, 128]).cuda()
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data = data.cuda()
        outputs, features = model(data)
        outputs_ID, _ = model_ID(data)

        softprobs = torch.softmax(outputs_ID, dim=1)
        _, pred = torch.max(softprobs, 1)
        if use_gpu:
            predTargets = torch.zeros_like(outputs_ID).cuda().scatter_(1, pred.view(-1,1), 1)
            bias = torch.zeros_like(outputs_ID).cuda().scatter_(1, pred.view(-1,1), -1e5) 
        else:
            predTargets = torch.zeros_like(outputs_ID).scatter_(1, pred.view(-1,1), 1)
            bias = torch.zeros_like(outputs_ID).scatter_(1, pred.view(-1,1), -1e5) 
        aleatoricUnc = -torch.logsumexp(outputs_ID, dim=1) + torch.logsumexp(outputs_ID*(1-predTargets)+bias, dim=1)

        if outputs.shape[1] == len(knownclass):
            energy = -torch.logsumexp(outputs, dim=1)
        else:
            energy = -torch.logsumexp(outputs[:,:-1], dim=1) + torch.log(1+torch.exp(outputs[:,-1]))
        Uncertainty = energy

        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))
        predArr += list(np.array(pred.cpu().data))
        uncertaintyArr += list(Uncertainty.cpu().detach().numpy())
        aleatoricUncArr += list(aleatoricUnc.cpu().detach().numpy())
        feat_unlab = torch.cat([feat_unlab, features.data],0)
    feat_unlab = feat_unlab[1:]
    feat_unlab = F.normalize(feat_unlab, dim=1)

    aleatoricUncArr = np.asarray(aleatoricUncArr)
    input_aleatoricUncArr = (aleatoricUncArr-aleatoricUncArr.min())/(aleatoricUncArr.max()-aleatoricUncArr.min())
    input_aleatoricUncArr = input_aleatoricUncArr.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_aleatoricUncArr)
    prob = gmm.predict_proba(input_aleatoricUncArr) 
    info_prob = prob[:,gmm.means_.argmax()] 

    uncertaintyArr = np.asarray(uncertaintyArr)
    input_uncertaintyArr = (uncertaintyArr-uncertaintyArr.min())/(uncertaintyArr.max()-uncertaintyArr.min())
    input_uncertaintyArr = input_uncertaintyArr.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_uncertaintyArr)
    prob = gmm.predict_proba(input_uncertaintyArr) 
    clean_prob = prob[:,gmm.means_.argmin()] 

    dists_all = torch.mm(feat_all, feat_unlab.t())
    _, top_k_index = dists_all.topk(250, dim=1, largest=True, sorted=True) ## Top-K similar scores and corresponding indexes
    dists_all, top_k_index = dists_all.cpu(), top_k_index.cpu()
    rknn_logits = torch.zeros(feat_unlab.shape[0], len(knownclass)+1, dtype=torch.long)
    for i in range(len(knownclass)):
        unique_indices, counts = torch.unique(top_k_index[np.asarray(labelArr_all)==i], return_counts=True)
        rknn_logits[unique_indices,i] = counts

    if trainloader_ID_w_OOD != None:
        unique_indices, counts = torch.unique(top_k_index[labelArr_all==len(knownclass)], return_counts=True)
        rknn_logits[unique_indices,len(knownclass)] = counts
        energy = -torch.logsumexp(rknn_logits.float()[:,:-1], dim=1) + torch.log(1+torch.exp(rknn_logits.float()[:,-1]))
    else:
        energy = -torch.logsumexp(rknn_logits.float()[:,:-1], dim=1)
    Uncertainty = energy
    uncertaintyArr = list(Uncertainty.cpu().detach().numpy()) 

    uncertaintyArr = np.asarray(uncertaintyArr)
    input_uncertaintyArr = (uncertaintyArr-uncertaintyArr.min())/(uncertaintyArr.max()-uncertaintyArr.min())
    input_uncertaintyArr = input_uncertaintyArr.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_uncertaintyArr)
    prob = gmm.predict_proba(input_uncertaintyArr) 
    clean_prob2 = prob[:,gmm.means_.argmin()]

    clean_prob *= clean_prob2

    queryIndex = torch.tensor(queryIndex)
    labelArr = torch.tensor(labelArr)
    predArr = torch.tensor(predArr)
    cleanArr = torch.tensor(clean_prob)
    infoArr = torch.tensor(info_prob)
    num_per_class = int(math.ceil(args.query_batch/args.known_class))

    sel_queryIndex, sel_queryLabelArr = [], []
    for i in range(args.known_class):
        temp_queryIndex = queryIndex[predArr == i]
        temp_labelArr = labelArr[predArr == i]
        temp_cleanArr = cleanArr[predArr == i]
        temp_infoArr = infoArr[predArr == i]

        _, indices = torch.topk(temp_cleanArr.view(-1), num_per_class*int(args.k1))
        temp_queryIndex = temp_queryIndex[indices]
        temp_labelArr = temp_labelArr[indices]
        temp_infoArr = temp_infoArr[indices]

        tmp_data = np.vstack((temp_queryIndex.numpy(), temp_labelArr.numpy()))
        tmp_data = np.vstack((tmp_data, temp_infoArr.numpy())).T
        tmp_data = tmp_data[(-tmp_data[:,2]).argsort()]
        tmp_data = tmp_data.T
        sel_queryIndex += list(tmp_data[0][:num_per_class]) 
        sel_queryLabelArr += list(tmp_data[1][:num_per_class]) 

    queryIndex = np.asarray(sel_queryIndex).astype(np.int32)
    queryLabelArr = np.asarray(sel_queryLabelArr)

    idx = torch.randperm(len(sel_queryIndex))
    queryIndex = queryIndex[idx][:args.query_batch]
    queryLabelArr = queryLabelArr[idx][:args.query_batch]

    labelArr = labelArr.numpy()

    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[np.where(queryLabelArr >= args.known_class)[0]], precision, recall
