import numpy as np
import pandas as pd
import random
import torch
from GCN import GCN_Net
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
colors = list(mcolors.TABLEAU_COLORS.keys())
random.seed(123)

aadj=pd.DataFrame(np.loadtxt("interaction.txt"))
SD=pd.DataFrame(np.loadtxt("SD.txt"))
SM=pd.DataFrame(np.loadtxt("SM.txt"))

adj_list=[]
for index,row in aadj.iterrows():
    for i in range(len(aadj.iloc[0])):
        if row[i]==1:
            adj_list.append([index,i+495])#(miRNA节点下标,疾病节点下标)
feature_matrix=pd.DataFrame(0,index=range(878),columns=range(878))

for index in range(len(SM)):
    feature_matrix.iloc[index,:495]=SM.iloc[index]
for index in range(len(SM),len(SM)+len(SD)):
    feature_matrix.iloc[index,495:]=SD.iloc[index-495]


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=False)
label=[1]*len(adj_list)
label.extend([0]*len(adj_list))
random.shuffle(adj_list)
lenth=len(adj_list)
i=0
while i <lenth:
    neg_index=random.randint(495,877)
    if [adj_list[i][0],neg_index] not in adj_list:
        adj_list.append([adj_list[i][0],neg_index])
        i+=1
    else:
        continue
adj_list=np.array(adj_list)
label=np.array(label)
fold=-1
for train_index, val_index in skf.split(adj_list,label):
    fold+=1
    train_x,val_x=list(adj_list[train_index]),list(adj_list[val_index])
    train_y,val_y=list(label[train_index]),list(label[val_index])

    adj=[]
    for i in range(len(train_x)):
        if train_y[i]==1:
            adj.append(train_x[i])


    adj=torch.tensor(adj).to(torch.long).T.cuda()
    train_x=torch.tensor(train_x).to(torch.long).T.cuda()
    train_y=torch.tensor(train_y).to(torch.float32).cuda()
    val_x=torch.tensor(val_x).to(torch.long).T.cuda()
    val_y=torch.tensor(val_y).to(torch.float32).cuda()
    feature=torch.tensor(feature_matrix.to_numpy()).to(torch.float32).cuda()

    model = GCN_Net(len(feature[0]), 128, 64).cuda()
    model.train()
    opt = torch.optim.Adam(params=model.parameters(), lr=0.001,weight_decay=1e-4)
    loss_fn = torch.nn.BCELoss().cuda()

    
    epoch = 2000


    for i in range(epoch):
        y_hat = model(feature, adj, train_x)
        loss = loss_fn(y_hat.to(torch.float32), train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 100==0:
            print(loss)
    model.eval()

    from torchmetrics import AUROC
    from torchmetrics.classification import BinaryAccuracy, BinarySpecificity, BinaryPrecision
    Auc = AUROC(task="binary")
    with torch.no_grad():
        y_hat = model(feature, adj, val_x)
        Auc_value = Auc(y_hat.cpu(), val_y.cpu()).item()
        print(f"AUC:{Auc_value}")
        y_hat=y_hat.cpu()
        val_y=val_y.cpu()
        #_list=[i/100 for i in range(65,75)]
        #for i in _list:
        i=0.67
        Acc = BinaryAccuracy(i)
        Pcl = BinaryPrecision(i)
        Spc = BinarySpecificity(i)
        print(f"阈值为{i}时,Accuracy:{Acc(y_hat, val_y) * 100}%\n")
        print(f"阈值为{i}时,Specificity:{Spc(y_hat, val_y) * 100}%\n")
        print(f"阈值为{i}时,Precision:{Pcl(y_hat, val_y) * 100}%\n")

    fpr, tpr, thresholds = roc_curve(val_y, y_hat.cpu())
    roc_auc = auc(fpr, tpr)

    x = np.linspace(0, 1, 100)
    f = interp1d(fpr, tpr)
    tpr = f(x)
    fpr = x

    plt.plot(fpr, tpr, color=mcolors.TABLEAU_COLORS[colors[fold]], label=f'Fold {fold} ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color=mcolors.TABLEAU_COLORS[colors[fold]], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
plt.show()

