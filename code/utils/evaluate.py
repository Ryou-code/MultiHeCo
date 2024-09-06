import numpy as np
import torch
from .logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import torch
import numpy as np


##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################

def evaluate(embeds, ratio, idx_train, idx_val, idx_test, label, nb_classes, device, dataset, lr, wd
             , isTest=True):
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),
                                            multi_class='ovr'
                                            ))

    if isTest:
        print("\t[Classification] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
              .format(np.mean(macro_f1s),
                      np.std(macro_f1s),
                      np.mean(micro_f1s),
                      np.std(micro_f1s),
                      np.mean(auc_score_list),
                      np.std(auc_score_list)
                      )
              )
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    f = open("result_"+dataset+str(ratio)+".txt", "a")
    f.write(str(np.mean(macro_f1s))+"\t"+str(np.mean(micro_f1s))+"\t"+str(np.mean(auc_score_list))+"\n")
    f.close()
'''

def evaluate(embeds, ratio, idx_train, idx_val, idx_test, label, nb_classes, device, dataset, isTest=True):
    # 将设备和嵌入维度准备好
    hid_units = embeds.shape[1]
    embeds = embeds.to(device)
    true_labels = torch.argmax(label, dim=-1).cpu().numpy()

    nmi_scores = []
    ari_scores = []

    for _ in range(50):
        # K-means 聚类
        kmeans = KMeans(n_clusters=nb_classes, n_init=10, random_state=None)
        predicted_labels = kmeans.fit_predict(embeds.cpu().numpy())

        # 计算 NMI 和 ARI
        nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
        ari_score = adjusted_rand_score(true_labels, predicted_labels)

        nmi_scores.append(nmi_score)
        ari_scores.append(ari_score)

    if isTest:
        print("\t[Clustering] NMI_mean: {:.4f} var: {:.4f}  ARI_mean: {:.4f} var: {:.4f}"
              .format(np.mean(nmi_scores),
                      np.std(nmi_scores),
                      np.mean(ari_scores),
                      np.std(ari_scores)
                      )
              )
    else:
        return np.mean(nmi_scores), np.mean(ari_scores)

    # 将结果保存到文件
    f = open("result_clustering_"+dataset+str(ratio)+".txt", "a")
    f.write(f"NMI: {np.mean(nmi_scores):.4f} Var: {np.std(nmi_scores):.4f} ARI: {np.mean(ari_scores):.4f} Var: {np.std(ari_scores):.4f}\n")
    f.close()
'''