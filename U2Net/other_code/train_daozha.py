import os

import torch
from torch.utils.data import Dataset, DataLoader
from models.net import U2NET
from utiles.dataset_daozha import MeterDataset
from utiles.loss_function import DiceLoss, FocalLoss

import torch.nn as nn
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

class Trainer(object):

    def __init__(self):
        self.net = U2NET(3, 2)
        # 输出两个mask
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        self.loss_function = FocalLoss(alpha=0.75)
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 训练模式，注意需要修改对应数据dataset的数据位置路径
        self.data_set_train = MeterDataset(mode='train')
        if os.path.exists('../weight/net.pt'):
            self.net.load_state_dict(torch.load('weight/net.pt', map_location='cpu'),False)
        self.data_set_val = MeterDataset(mode='val')
        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
        self.net = self.net.cuda()
        #self.net.to(self.device)

    def __call__(self):
        epoch_num = 200    #120
        batch_size_train = 10
        batch_size_val = 1
        ite_num = 0
        data_loader_train = DataLoader(self.data_set_train, batch_size_train, True, num_workers=2)
        data_loader_val = DataLoader(self.data_set_val, batch_size_val, False, num_workers=2)
        # loss_sum = 0
        # running_tar_loss = 0
        save_frq = 120
        # 输出权重保存路径
        model_dir = 'weight/net_daozha.pt'
        last_loss = []
        # 用来保存训练以及验证过程中信息
        results_file = "result_file/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        for epoch in range(epoch_num):
            '--------train---------'
            torch.set_grad_enabled(True)
            self.net.train()
            loss_sum = 0
            running_tar_loss = 0
            for i, (images, masks) in enumerate(data_loader_train):
                ite_num += 1
                images = images.cuda()
                masks = masks.cuda()
                d0, d1, d2, d3, d4, d5, d6 = self.net(images)
                print("d0_shape:",d0.shape)  #torch.Size([10, 2, 416, 416])
                print("masks_shape:",masks.shape)   #torch.Size([10, 2, 416, 416])
                loss, loss0 = self.calculate_loss(d0, d1, d2, d3, d4, d5, d6, masks)
                self.optimizer.zero_grad()
                # print(loss)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()
                running_tar_loss += loss0.item()
                del d0, d1, d2, d3, d4, d5, d6, loss
                print(
                    f'epoch:{epoch}; batch:{i + 1}; train loss:{loss_sum / (i + 1)}; tar:{running_tar_loss / (i + 1)}')
                # 将结果写入保存文件
                with open(results_file, "a") as f:
                    # 记录每个epoch对应的train_loss、lr以及验证集各指标
                    write_info =  f'epoch:{epoch}; batch:{i + 1}; train loss:{loss_sum / (i + 1)}; tar:{running_tar_loss / (i + 1)}\n'
                    f.write(write_info)
                #if ite_num % save_frq == 0:
                #    torch.save(self.net.module.state_dict(), model_dir)
            # '---------val----------'
            torch.set_grad_enabled(False)
            torch.cuda.empty_cache()
            self.net.eval()
            loss_val_all = 0
            loss0_val_all = 0

            for i,(images,masks) in enumerate(data_loader_val):
                images = images.cuda()
                masks = masks.cuda()
                d0, d1, d2, d3, d4, d5, d6 = self.net(images)
                loss_val, loss0_val = self.calculate_loss(d0, d1, d2, d3, d4, d5, d6, masks)
                loss_val_all = loss_val_all+loss_val
                loss0_val_all = loss0_val_all+loss0_val
                del d0, d1, d2, d3, d4, d5, d6, loss_val

            loss_val_all_mean = loss_val_all/len(data_loader_val)
            loss0_val_all_mean = loss0_val_all/len(data_loader_val)
            loss_sum_all = loss_val_all_mean + loss0_val_all_mean
            last_loss.append(loss_sum_all)
            if len(last_loss)>2:
                if loss_sum_all<last_loss[-2]:
                    #torch.save(self.net.module.state_dict(), model_dir.replace("net",str(epoch)+"_net"))
                    torch.save(self.net.module.state_dict(), model_dir)
            # '---------val----------'
            # self.net.eval()
            # for i,(images,masks) in enumerate(data_loader_val):
            #     images = images.to(self.device)
            #     masks = masks.to(self.device)
            #     d0, d1, d2, d3, d4, d5, d6 = self.net(images)

    def calculate_loss(self, d0, d1, d2, d3, d4, d5, d6, labels):
        loss0 = self.loss_function(d0, labels)
        loss1 = self.loss_function(d1, labels)
        loss2 = self.loss_function(d2, labels)
        loss3 = self.loss_function(d3, labels)
        loss4 = self.loss_function(d4, labels)
        loss5 = self.loss_function(d5, labels)
        loss6 = self.loss_function(d6, labels)
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        return loss, loss0


if __name__ == '__main__':
    trainer = Trainer()
    trainer()
