from numpy.core.defchararray import center
import torch, random
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import time

torch.manual_seed(42)
 
class RBFN(nn.Module):
    def __init__(self, centers, n_out=2):
        super(RBFN, self).__init__()
        self.n_out = n_out
        self.num_centers = centers.size(0) 
        self.dim_centure = centers.size(1) 
        self.centers = nn.Parameter(centers)
        self.beta = torch.ones(1, self.num_centers)*10
        self.linear = nn.Linear(self.num_centers+self.dim_centure, self.n_out, bias=True)
        self.initialize_weights()
 
 
    def kernel_fun(self, batches):
        n_input = batches.size(0)
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False)))
        return C
 
    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(torch.cat([batches, radial_val], dim=1))
        return class_score
 
    def initialize_weights(self, ):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
 
    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)
 
def cal_acc_recall(pre, gt):
    train_correct = (pre == gt).sum()
    train_acc = train_correct
    return train_acc / (len(pre))


if __name__ =="__main__":
    data = pd.read_csv(r'C:\Users\wangzy\Desktop\EE7207\data_train.csv')
    train_x = data.drop(['Label'], axis=1)
    estimator = KMeans(n_clusters=20)
    estimator.fit(train_x)
    centroids = estimator.cluster_centers_
    train_y = np.array(data[['Label']])
    train_x = np.array(train_x)
    data = torch.tensor(train_x, dtype=torch.float32)
    label = torch.tensor(train_y, dtype=torch.float32)
    label = label.view(300)
    label_temp = torch.ones(300)
    label = label.eq(label_temp.data).float()
    label = label.view(300,1)

    data_val = pd.read_csv(r'C:\Users\wangzy\Desktop\EE7207\data_val.csv')
    val_x = data_val.drop(['Label'], axis=1)
    val_y = np.array(data_val[['Label']])
    val_x = np.array(val_x)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_label = torch.tensor(val_y, dtype=torch.float32)
    val_label = val_label.view(30)
    val_label_temp = torch.ones(30)
    val_label = val_label.eq(val_label_temp.data).float()
    val_label = val_label.view(30,1)

    data_test = pd.read_csv(r'C:\Users\wangzy\Desktop\EE7207\data_test.csv')
    test_x = data_test.drop(['Label'], axis=1)
    test_x = np.array(test_x)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    
    stime=time.time()
    centers = torch.cat((data[0:4,:], data[295:299,:]))
    # Initialize centers by K-Means
    # centers = np.array(centroids)
    # centers = torch.tensor(centers, dtype=torch.float32)
    rbf = RBFN(centers,1)
    params = rbf.parameters()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params,lr=0.1,momentum=0.9)
 
    for i in range(1000):
        optimizer.zero_grad()
 
        y = rbf.forward(data)
        loss = loss_fn(y,label)
        loss.backward()
        optimizer.step()
        print(i,"\t",loss.data)
 
    # Evaluation
    # y = rbf.forward(val_x)
    # res = y.gt(0.5).float().view(30)
    # val_label = val_label.view(30)
    # print(res)
    # print(val_label)
    # correct = (res==val_label).sum().float()
    # print('Acc: {}'.format(correct/len(val_label)))
    # correct = (res[:22]==val_label[:22]).sum().float()
    # print('Recall: {}'.format(correct/22))
    # etime = time.time()
    # print('Time: {}'.format(etime-stime))

    # Inference
    y = rbf.forward(test_x)
    res = y.gt(0.5).float().view(len(test_x))
    res = res.tolist()
    res = [int(i) for i in res]
    for id, num in enumerate(res):
        if num == 0:
            res[id] = -1
    print(res)