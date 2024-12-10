import torch
from torch import nn
from model import Network
from metric import valid, evaluate
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
import warnings
from sklearn.cluster import KMeans


from utils.utils import neraest_labels
import time
from scipy.optimize import linear_sum_assignment
from clusteringPerformance import clusteringMetrics
import contrastive_loss
from spectral_clustering import KMeans as Kmeans
from spectral_clustering import spectral_clustering

st = time.time()

torch.set_num_threads(4)
# MNIST-USPS
# BDGP
# LableMe
# Fashion
Dataname = 'Scene15'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument('--neighbor_num', default=5, type=int)
parser.add_argument('--feature_dim', default=10, type=int)
parser.add_argument('--gcn_dim', default=128, type=int)
parser.add_argument('--tau', default=0.1, type=float)
parser.add_argument('--lambda1', default=1.0, type=float)
parser.add_argument('--lambda2', default=1.0, type=float)
parser.add_argument('--eta', default=1.0, type=float)
parser.add_argument('--neg_size', default=128, type=int)
parser.add_argument('--fine_epochs', default=50, type=int)
parser.add_argument('--instance_temperature', default=0.5, type=float)
parser.add_argument('--cluster_temperature', default=1.0, type=float)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings('ignore')

if args.dataset == "Fashion":
    args.mse_epochs = 200
    args.con_epochs = 200
    seed = 10
if args.dataset == 'LabelMe':
    args.mse_epochs = 200
    args.con_epochs = 200
    seed = 10

if args.dataset == 'CCV':
    args.mse_epochs = 2000
    args.con_epochs = 0
    args.fine_epochs = 0
    seed = 10
    args.learning_rate = 0.0005

if args.dataset == 'BDGP':
    args.mse_epochs = 200
    args.con_epochs = 200
    seed = 10
    args.neg_size = 100

if args.dataset == 'MNIST-USPS':
    args.mse_epochs = 200
    args.con_epochs = 500
    seed = 10

if args.dataset == 'Caltech-2V':
    args.mse_epochs = 1000
    args.con_epochs = 1000
    seed = 10
    args.learning_rate = 0.0006


if args.dataset == 'Caltech-3V':
    args.mse_epochs = 0
    args.con_epochs = 2000
    seed = 10
    args.learning_rate = 0.0005

if args.dataset == 'Caltech-4V':
    args.mse_epochs = 0
    args.con_epochs = 2000
    seed = 10
    args.batch_size = 1400
    args.learning_rate = 0.0001

if args.dataset == 'Caltech-5V':
    args.mse_epochs = 1200
    args.con_epochs = 800
    seed = 10
    args.batch_size = 1400
    args.learning_rate = 0.0005
if args.dataset == 'COIL20':
    args.mse_epochs = 500
    args.con_epochs = 0
    seed = 8
    args.learning_rate = 0.00001
if args.dataset =='MSRCv1':
    args.mse_epochs = 0
    args.con_epochs = 600
    args.fine_epochs = 0
    args.batch_size = 210
    seed = 10
    args.learning_rate = 0.00005
    # args.learning_rate = 0.00005
if args.dataset =='Caltech101-20':
    args.mse_epochs = 200
    args.con_epochs = 1800
    args.fine_epochs = 0
    args.batch_size = 2386
    seed = 10
    args.learning_rate = 0.0001
if args.dataset =='YouTubeFaces':
    args.mse_epochs = 200
    args.con_epochs = 1800
    args.fine_epochs = 0
    args.batch_size = 256
    seed = 10
    args.learning_rate = 0.0001
if args.dataset =='DHA':
    args.mse_epochs = 0
    args.con_epochs = 500
    args.fine_epochs = 0
    args.batch_size = 256
    seed = 3304
    args.learning_rate = 0.00008
if args.dataset == 'Scene15':
    args.mse_epochs = 0
    args.con_epochs = 2000
    args.fine_epochs = 0
    args.batch_size = 256
    seed =10
    args.learning_rate = 0.00005

if args.dataset == 'nus-wide':
    args.mse_epochs = 0
    args.con_epochs = 400
    args.fine_epochs = 0
    args.batch_size = 256
    seed = 10
    args.learning_rate = 0.000005

if args.dataset == 'flickr':
    args.mse_epochs = 0
    args.con_epochs = 500
    args.fine_epochs = 0
    args.batch_size = 256
    seed = 10
    args.learning_rate = 0.000005

if args.dataset == 'ESP-Game':
    args.mse_epochs = 0
    args.con_epochs = 500
    args.fine_epochs = 0
    args.batch_size = 256
    seed = 10
    args.learning_rate = 0.000005

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)
data_loader_test = torch.utils.data.DataLoader(dataset,batch_size=data_size,
    shuffle=True,
    drop_last=True,)

# 添加噪声函数
def add_noise(input, noise_factor=0.1):
    noise = torch.randn_like(input) * noise_factor
    return input + noise


def pretrain(epoch):
    tot_loss = 0.
    for batch_idx, (xs, labels, _) in enumerate(data_loader):

        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()

        xrs, hs, qs, gs = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(F.mse_loss(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

    return hs, labels,model.state_dict()


def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y


def model_train(epoch):
    total_loss = 0.
    # model.load_state_dict(torch.load('models/'+args.dataset+'_pretraining.pkl'))
    model.train()
    # Hs = []
    # labels_vector = []
    # for v in range(view):
    #     Hs.append([])
    for batch_idx, (xs, labels, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        # labels_vector.extend(labels.numpy())
        xrs, hs,qs,gs = model(xs)
        # fusion = torch.hstack(hs)
        # fusion = 0
        # for v in range(view):
            # fusion = fusion + hs[v]
            # hs[v] = hs[v].detach()
            # Hs[v].extend(hs[v].cpu().detach().numpy())
        # fusion = fusion/view

        # km = KMeans(n_clusters=class_num,n_init=10)
        # km.fit_predict(fusion.data.cpu().numpy())
        loss_instance = 0.
        loss_cluster = 0.
        for v in range(view):
            for w in range(v+1, view):
                loss_instance += criterion_instance(hs[v],hs[w])
                loss_cluster += criterion_cluster(qs[v],qs[w])
        # K-way normalized cuts or k-Means. Default: k-Means
        use_kmeans = True
        cluster_num = 400  # k, 50, 100, 200, 400
        # cluster_num = max(200, 2 * class_num)
        iter_num = 5
        k_eigen = class_num
        cld_t = 0.2
        cluster_labels = []
        centroids = []
        if use_kmeans:
            for v in range(view):
                cl_label, centroid = Kmeans(gs[v], K=cluster_num, Niters=iter_num)
                cluster_labels.append(cl_label)
                centroids.append(centroid)
        else:
            for v in range(view):
                cl_label, centroid = spectral_clustering(gs[v], K=k_eigen, clusters=cluster_num, Niters=iter_num)
                cluster_labels.append(cl_label)
                centroids.append(centroid)
                # instance-group discriminative learning

        criterion_cld = nn.CrossEntropyLoss().cuda()
        CLD_loss = 0
        for v in range(view):
            for w in range(view):
                if v != w:
                    affnity = torch.mm(gs[v], centroids[w].t())
                    CLD_loss = CLD_loss + criterion_cld(affnity.div_(cld_t), cluster_labels[v])

        CLD_loss = CLD_loss / view

        cross_loss = 0
        criterion_cross = nn.CrossEntropyLoss().cuda()
        # for v in range(view):
        #     for w in range(view):
        #         if v != w:
        #             affnity = torch.mm(hs[v], centroids[w].t())
        #             cross_loss = cross_loss + criterion_cross(affnity.div_(cld_t), cluster_labels[v])
        loss_cf = []
        loss_cc = []
        for v in range(view):
            for w in range(v, view):
                loss_cluster_feature = criterion_cluster.cluster_feature(hs[v], gs[w], centroid, cl_label,cluster_num)
                loss_cf.append(loss_cluster_feature.cpu().detach().numpy())
                # loss_cluster_center = criterion_cluster.cluster_orient(torch.Tensor(km.cluster_centers_).to(device),centroid)
                # loss_cc.append(loss_cluster_center.cpu().detach().numpy())
        # cross_loss = cross_loss / view
        loss_cf = sum(loss_cf)
        # loss_cc = sum(loss_cc)
        loss = loss_cluster  + CLD_loss + loss_instance + loss_cf
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # for v in range(view):
    #     Hs[v] = np.array(Hs[v])
    # cat_feature = np.concatenate(Hs, axis=1)
    # labels_vector = np.array(labels_vector).reshape(data_size)
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(total_loss / len(data_loader)))
    return hs, labels


def model_fine(epoch):
    total_loss = 0.

    for batch_idx, (xs, labels, _) in enumerate(data_loader):
        # xns = []
        for v in range(view):
            xn = add_noise(xs[v])
            # xns.append(xn)
            # xns[v] = xns[v].to(device)
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, hs, fusion_feautre, qs, p = model(xs)

        kmeans = KMeans(n_clusters=class_num, n_init=10)
        y_pred_km = kmeans.fit_predict(fusion_feautre.data.cpu().numpy())
        y_pred_km = y_pred_km.flatten()
        y_pred = y_pred_km
        cross_entropy = torch.nn.CrossEntropyLoss()
        loss_list = []
        for v in range(view):
            # loss_list.append(F.mse_loss(xs[v], xrs[v]))  # reconstruction loss
            # loss_list.append(criterion.forward_co_cluster(qs[v],p))
            q = qs[v].detach().cpu()
            q = torch.argmax(q, dim=1).numpy()
            p_hat = match(y_pred, q)
            loss_list.append(cross_entropy(qs[v], p_hat))
            # for t in range(v+1,view):
            #     loss_list.append(criterion.forward_co_cluster(qs[v],qs[t]))

            # loss_list.append(criterion.forword_debiased_instance(fusion_fea, hs[v], y_pred))
            #
            # loss_list.append(criterion.forword_feature(fusion_fea.T, hs[v].T))
            #
            # qn = neraest_labels(fusion_fea, qs[v]).to(device)
            # loss_list.append(criterion.forward_pui_label(p, qn))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(total_loss / len(data_loader)))
    return fusion_feautre, labels


accs = []
nmis = []
purs = []
aris = []

if not os.path.exists('./models'):
    os.makedirs('./models')

T = 1
for i in range(T):
    print("ROUND:{}".format(i + 1))

    # Network train
    model = Network(view, dims, args.feature_dim, class_num, device)
    # print(model)
    model = model.to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, device).to(device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, device).to(device)

    epoch = 1
    # pretrain
    # while epoch <= args.mse_epochs:
    #     zs, labels,model_save = pretrain(epoch)
    #     epoch += 1
    # torch.save(model_save, './models/' + args.dataset + '_pretraining.pkl')
    # #
    # hc = torch.cat(zs, dim=1)
    # kmeans = KMeans(n_clusters=class_num, n_init=10)
    # y_pred = kmeans.fit_predict(hc.data.cpu().numpy())
    # y_pred = y_pred.flatten()
    # labels = labels.flatten()
    # labels = labels.data.cpu().numpy()
    #
    # # nmi, ari, acc, pur = evaluate(labels, y_pred)
    # acc, nmi, ari, pur, fscore, precision, recall = clusteringMetrics(labels, y_pred)
    # print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f} F-score={:.4f} Precision={:.4f} Recall={:.4f}'.format(acc,
    #                                                                                                                nmi,
    #                                                                                                                ari,
    #                                                                                                                pur,
    #                                                                                                                fscore,
    #                                                                                                                precision,
    #                                                                                                                recall))
    best_acc = 0
    best_ari = 0
    best_nmi = 0
    best_pur = 0
    best_fsc = 0
    best_precision = 0
    best_recall = 0
    best_result =[]
    while epoch <= args.mse_epochs + args.con_epochs:
        fusion, labels = model_train(epoch)

        if epoch % 1 == 0:
            acc, nmi, ari, pur, fscore, precision, recall, acc_k, nmi_k, ari_k, pur_k, fscore_k, precision_k, recall_k,kmeans_vectors,cat_feature,labels_vector = valid(
                model, device, dataset, view, data_size, class_num,
                eval_h=False)
            if ari_k >=best_ari:
                np.save('./output/{}_label_ari.npy'.format(Dataname), kmeans_vectors)
                np.save('./output/{}_z_ari.npy'.format(Dataname),cat_feature)
                np.save('./output/{}_gnd_ari.npy'.format(Dataname),labels_vector)
                best_ari = ari_k
            if nmi_k >=best_nmi:
                np.save('./output/{}_label_nmi.npy'.format(Dataname), kmeans_vectors)
                np.save('./output/{}_z_nmi.npy'.format(Dataname),cat_feature)
                np.save('./output/{}_gnd_nmi.npy'.format(Dataname),labels_vector)
                best_nmi = nmi_k
            if pur_k >=best_pur:
                np.save('./output/{}_label_pur.npy'.format(Dataname), kmeans_vectors)
                np.save('./output/{}_z_pur.npy'.format(Dataname),cat_feature)
                np.save('./output/{}_gnd_pur.npy'.format(Dataname),labels_vector)
                best_pur = pur_k
            if recall_k >=best_recall:
                np.save('./output/{}_label_recall.npy'.format(Dataname), kmeans_vectors)
                np.save('./output/{}_z_recall.npy'.format(Dataname),cat_feature)
                np.save('./output/{}_gnd_recall.npy'.format(Dataname),labels_vector)
                best_recall = recall_k
            if fscore_k >=best_fsc:
                np.save('./output/{}_label_fsc.npy'.format(Dataname), kmeans_vectors)
                np.save('./output/{}_z_fsc.npy'.format(Dataname),cat_feature)
                np.save('./output/{}_gnd_fsc.npy'.format(Dataname),labels_vector)
                best_fsc = fscore_k
            if precision_k >=best_precision:
                np.save('./output/{}_label_recall.npy'.format(Dataname), kmeans_vectors)
                np.save('./output/{}_z_recall.npy'.format(Dataname),cat_feature)
                np.save('./output/{}_gnd_recall.npy'.format(Dataname),labels_vector)
                best_precision = precision_k
            if acc_k >= best_acc:
                np.save('./output/{}_label_acc.npy'.format(Dataname), kmeans_vectors)
                np.save('./output/{}_z_acc.npy'.format(Dataname),cat_feature)
                np.save('./output/{}_gnd_acc.npy'.format(Dataname),labels_vector)
                best_acc = acc_k
                best_result = []
                best_result.extend([acc, nmi, ari, pur, fscore, precision, recall, acc_k, nmi_k, ari_k, pur_k, fscore_k, precision_k, recall_k,epoch])
            print("Clustering results on semantic labels: ")
            print(
                'ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f} F-score={:.4f} Precision={:.4f} Recall={:.4f}'.format(
                    acc,
                    nmi,
                    ari,
                    pur,
                    fscore,
                    precision,
                    recall))
            print("Clustering results on kmeans clustering: ")
            print(
                'ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f} F-score={:.4f} Precision={:.4f} Recall={:.4f}'.format(
                    acc_k,
                    nmi_k,
                    ari_k,
                    pur_k,
                    fscore_k,
                    precision_k,
                    recall_k))
            print('-'*50)
        epoch += 1

        # while epoch <= args.mse_epochs + args.con_epochs + args.fine_epochs:
        #     fusion, labels = model_fine(epoch)
        #
        #     if epoch % 1 == 0:
        #         acc, nmi, ari, pur, fscore, precision, recall, acc_k, nmi_k, ari_k, pur_k, fscore_k, precision_k, recall_k = valid(
        #             model, device, dataset, view, data_size, class_num,
        #             eval_h=False)
        #
        #         print("Clustering results on semantic labels: ")
        #         print(
        #             'ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f} F-score={:.4f} Precision={:.4f} Recall={:.4f}'.format(
        #                 acc,
        #                 nmi,
        #                 ari,
        #                 pur,
        #                 fscore,
        #                 precision,
        #                 recall))
        #         print("Clustering results on kmeans clustering: ")
        #         print(
        #             'ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f} F-score={:.4f} Precision={:.4f} Recall={:.4f}'.format(
        #                 acc_k,
        #                 nmi_k,
        #                 ari_k,
        #                 pur_k,
        #                 fscore_k,
        #                 precision_k,
        #                 recall_k))
        #     epoch += 1
    print('best result: ACC=', best_result[0], 'NMI=', best_result[1], 'ARI=', best_result[2],
          'Pur=', best_result[3], 'fscore=', best_result[4], 'precision=', best_result[5], 'recall=',
          best_result[6])
    print('Epoch:',best_result[-1],'acc_k=', best_result[7], 'nmi_k=', best_result[8], 'ari_k',
          best_result[9], 'pur_k=', best_result[10], 'fscore_k=', best_result[11], 'precision_k=',
          best_result[12], 'recall_k=', best_result[13])
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print('dataset:', args.dataset)
