import time
import sys
import random
import csv
import os

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.transforms import functional as tvf
from torch import nn, optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model import feature_extractor, MLP_fc1, Multimodal_encoder, PersAM
import Dataset

#seed
def seed_set(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def train(model, rank, optimizer, train_loader, weight, n_class):
    model.train()
    train_class_loss = 0.0
    correct_num = 0
    loss_fn = nn.CrossEntropyLoss()

    for (input_tensor, slideID, class_label, interview_tensor, blood_tensor) in train_loader:
        input_tensor = input_tensor.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        interview_tensor = interview_tensor.to(rank, non_blocking=True)
        blood_tensor = blood_tensor.to(rank, non_blocking=True)
        for bag_num in range(input_tensor.shape[0]):
            optimizer.zero_grad()

            class_prob, class_hat, A, A_img, _, _, _ = model(input_tensor[bag_num], interview_tensor[bag_num], blood_tensor[bag_num])

            weight_ = torch.Tensor(weight[torch.argmax(class_label[bag_num])]).to(rank, non_blocking=True)
            class_loss = loss_fn(class_prob, torch.argmax(class_label[bag_num], dim=1))
            A = A[:, 0]
            A_ = (1 - A) * 0.05 + 0.95 #[0.95, 1]
            atten_prob = 1 - A_.prod(dim=-1)
            atten_loss = F.binary_cross_entropy(atten_prob, class_label[bag_num], weight=weight_)#

            loss = class_loss + atten_loss
            train_class_loss += class_loss.item() + atten_loss.item()

            loss.backward()
            optimizer.step()

            predicted = torch.argmax(class_prob, dim=1)
            correct_num += predicted.eq(torch.argmax(class_label[bag_num], dim=1).view_as(predicted)).sum().item()


    return train_class_loss, correct_num

def valid(model, rank, test_loader, weight, n_class):
    model.eval()
    test_class_loss = 0.0
    test_class_loss_ = 0.0
    correct_num = 0
    loss_fn = nn.CrossEntropyLoss()
    for (input_tensor, slideID, class_label, interview_tensor, blood_tensor) in test_loader:

        input_tensor = input_tensor.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        interview_tensor = interview_tensor.to(rank, non_blocking=True)
        blood_tensor = blood_tensor.to(rank, non_blocking=True)
        for bag_num in range(input_tensor.shape[0]):
            with torch.no_grad():
                class_prob, class_hat, A, A_img, _, _, _ = model(input_tensor[bag_num], interview_tensor[bag_num], blood_tensor[bag_num])

                weight_ = torch.Tensor(weight[torch.argmax(class_label[bag_num])]).to(rank, non_blocking=True)
                class_loss = loss_fn(class_prob, torch.argmax(class_label[bag_num], dim=1))
                A = A[:, 0]
                A_ = (1 - A) * 0.05 + 0.95 #[0.95, 1]
                atten_prob = 1 - A_.prod(dim=-1)
                atten_loss = F.binary_cross_entropy(atten_prob, class_label[bag_num], weight=weight_)

                test_class_loss += class_loss.item() + atten_loss.item()
                test_class_loss_ += class_loss.item()

                predicted = torch.argmax(class_prob, dim=1)
                correct_num += predicted.eq(torch.argmax(class_label[bag_num], dim=1).view_as(predicted)).sum().item()

    return test_class_loss, test_class_loss_, correct_num

def rotate(img):
    return tvf.rotate(img, angle=90*np.random.randint(0, 4))


def train_model(rank, world_size, cv_test, mag):
    setup(rank, world_size)

    ##################SETTING#######################################
    EPOCHS = 9
    BAG_NUM = 30
    BAG_SIZE = 100
    EPS = 0.05 #label smooth
    INTERVIEW_DIM = 18
    BLOOD_DIN = 10
    FEATURE_DIM = 512
    NUM_CV = 5
    ################################################################
    # devide dataset into TRAIN and VALID
    for seed in range(1): #seed = 0
        seed_set(seed=seed)
        train_all = []
        valid_all = []
        test_all = []
        train_label = []
        valid_label = []
        test_label = []
        list_subtype = ['DLBCL','FL','Reactive'] #use subtype
        num_data = []

        label_list = np.eye(len(list_subtype))
        label_list = np.where(label_list==1., 1-EPS, EPS).tolist()
        for enum, subtype in enumerate(list_subtype):
            list_id = np.loadtxt(f'./{subtype}.txt', delimiter=',', dtype='str').tolist()
            random.shuffle(list_id)
            num_e = len(list_id) // NUM_CV
            num_r = len(list_id) % NUM_CV
            tmp_all = []
            for cv in range(NUM_CV):
                tmp = []
                for i in range(num_e):
                    tmp.append(list_id.pop(0))
                if cv < num_r:
                    tmp.append(list_id.pop(0))
                tmp_all.append(tmp)
            train_tmp = tmp_all[cv_test%5] + tmp_all[(cv_test+1)%5] + tmp_all[(cv_test+2)%5]
            train_all += train_tmp
            valid_all += tmp_all[(cv_test+3)%5]
            test_all += tmp_all[(cv_test+4)%5]
            num_data.append(len(train_tmp))
            train_tmp = [label_list[enum] for _ in range(len(train_tmp))]
            valid_tmp = [label_list[enum] for _ in range(len(tmp_all[(cv_test+3)%5]))]
            test_tmp = [label_list[enum] for _ in range(len(tmp_all[(cv_test+4)%5]))]
            train_label += train_tmp
            valid_label += valid_tmp
            test_label += test_tmp

        #calculate weight
        n_class = len(num_data)
        num_all = sum(num_data)
        weight_pos = num_all/(2*np.array(num_data)) #each class is a binary classification problem
        weight_neg = num_all/(2*(num_all - np.array(num_data)))

        weight = np.eye(n_class)
        for i in range(n_class):
            weight[i] = np.where(weight[i]==1., weight_pos[i], weight_neg[i])

        df = pd.read_csv("clinical_record.csv") #read clinical record data
        train_index = []
        for slideID in train_all:
            index = df[df["No"] == int(slideID)].index
            if len(index) == 0:
                continue
            train_index.append(index[0])

        valid_index = []
        for slideID in valid_all:
            index = df[df["No"] == int(slideID)].index
            if len(index) == 0:
                continue
            valid_index.append(index[0])


        ##################################################
        #Complement missing values with the median of the training data
        #When used, it is necessary to implement a process
        #to normalize clinical_record.

        #Deleted due to processing that cannot be disclosed.
        ##################################################

        train_interview = train_table[columns_interview].values
        train_blood = train_table[columns_blood].values
        valid_interview = valid_table[columns_interview].values
        valid_blood = valid_table[columns_blood].values

        train_dataset = []
        for i, slideID in enumerate(train_all):
            train_dataset.append([slideID, train_label[i], train_interview[i], train_blood[i]])

        valid_dataset = []
        for i, slideID in enumerate(valid_all):
            valid_dataset.append([slideID, valid_label[i], valid_interview[i], valid_blood[i]])


        #Create a directory to store the log
        makedir('train_log')
        log = f'train_log/PersAM_log_{mag}_cv-{cv_test}_{EPS}_{seed}.csv'

        if rank == 0:
            #log header
            f = open(log, 'w')
            f_writer = csv.writer(f, lineterminator='\n')
            csv_header = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_loss_", "valid_acc", "time"]
            f_writer.writerow(csv_header)
            f.close()


        #make model
        feature_ex = feature_extractor()
        attention = Multimodal_encoder(feature_dim=FEATURE_DIM, num_class=n_class, device=rank)
        mlp_fc1 = MLP_fc1(INTERVIEW_DIM, FEATURE_DIM)
        mlp_fc2 = MLP_fc1(BLOOD_DIN, FEATURE_DIM)

        model = PersAM(feature_ex, attention, mlp_fc1, mlp_fc2, n_class)

        model = model.to(rank)
        process_group = torch.distributed.new_group([i for i in range(world_size)])

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
        ddp_model = DDP(model, device_ids=[rank])

        #learning rate
        params=[{"params": ddp_model.module.feature_ex.parameters(), "lr": 0.0001},
                {"params": ddp_model.module.attention.parameters(), "lr": 0.0002},
                {"params": ddp_model.module.MLP_fc1.parameters(), "lr": 0.0002/50},
                {"params": ddp_model.module.MLP_fc2.parameters(), "lr": 0.0002/50},
                ]
        optimizer = optim.SGD(params=params, momentum=0.9, weight_decay=1e-4, nesterov=True)


        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Lambda(rotate),
            torchvision.transforms.ToTensor(),
        ])

        transform_ = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])

        #start training
        for epoch in range(EPOCHS):
            train_loss = 0.0
            train_acc = 0.0
            valid_loss = 0.0
            valid_loss_ = 0.0 #only crossentropy
            valid_acc = 0.0
            start_t = time.time()

            #Due to re-create the bags each time, create a dataset for each epoch.
            train_Dataset = Dataset.mmDataset(
                train=True,
                transform=transform,
                dataset=train_dataset,
                mag=mag,
                bag_num=BAG_NUM,
                bag_size=BAG_SIZE,
                epoch=epoch
            )

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_Dataset , rank=rank)

            train_loader = torch.utils.data.DataLoader(
                kurume_train,
                batch_size=1,
                shuffle=False,
                pin_memory=False,
                num_workers=4,
                sampler=train_sampler
            )

            # lr scheduling
            if epoch == 3 or epoch == 6:
                for opt in optimizer.param_groups:
                    opt["lr"] = opt["lr"] * 0.1

            #train
            class_loss, acc = train(ddp_model, rank, optimizer, train_loader, weight, n_class)

            train_loss += class_loss
            train_acc += acc

            valid_Dataset = Dataset.mmDataset(
                train=True,
                transform=transform_,
                dataset=valid_dataset,
                mag=mag,
                bag_num=BAG_NUM,
                bag_size=BAG_SIZE,
                epoch=epoch
            )


            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_Dataset, rank=rank)
            valid_loader = torch.utils.data.DataLoader(
                kurume_valid,
                batch_size=1,
                shuffle=False,
                pin_memory=False,
                num_workers=4,
                sampler=valid_sampler
            )

            # valid
            class_loss, class_loss_, acc = valid(ddp_model, rank, valid_loader, weight, n_class)

            valid_loss += class_loss
            valid_loss_ += class_loss_
            valid_acc += acc

            train_acc /= float(len(train_loader.dataset))
            valid_acc /= float(len(valid_loader.dataset))
            elapsed_t = time.time() - start_t

            f = open(log, 'a')
            f_writer = csv.writer(f, lineterminator='\n')
            f_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_loss_, valid_acc, elapsed_t])
            f.close()
            #save model
            if rank == 0:
                makedir('model_params')
                model_params = f'./model_params/PersAM_{mag}_cv-{cv_test}_epoch-{epoch}_{EPS}_{seed}.pth'
                torch.save(ddp_model.module.state_dict(), model_params)

if __name__ == '__main__':

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    num_gpu = 8 #number of GPU used

    args = sys.argv
    cv_tests = [int(i) for i in args[1]]
    mag = "40x" # magnitude

    for cv_test in cv_tests:
        mp.spawn(train_model, args=(num_gpu, cv_test, mag), nprocs=num_gpu, join=True)
