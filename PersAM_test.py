import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import csv
import os
import Dataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import sys
import random


from model import feature_extractor, MLP_fc1, Correlation_encoder, PersAM


def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

#select best model
def select_cnn(log_file):
    train_log = np.loadtxt(log_file, delimiter=',', dtype='str')
    valid_loss = train_log[1:,3].astype(np.float32)
    loss_list = []
    total_epoch = valid_loss.shape[0]/8
    for i in range(int(total_epoch)):
        tmp = valid_loss[i*8:(i+1)*8]
        if i < 3:
            loss_list.append(1000000)
        else:
            loss_list.append(np.sum(tmp))
    return loss_list.index(min(loss_list))

def test(model, device, test_loader, output_file, n_class):
    model.eval()

    for (input_tensor, slideID, class_label, interview_tensor, blood_tensor, pos_list) in test_loader:
        input_tensor = input_tensor.to(device)
        interview_tensor = interview_tensor.to(device)
        blood_tensor = blood_tensor.to(device)
        for bag_num in range(input_tensor.shape[0]):
            with torch.no_grad():
                class_prob, class_hat, A, A_img, _, _, _ = model(input_tensor[bag_num], interview_tensor[bag_num], blood_tensor[bag_num])

            class_softmax = F.softmax(class_prob, dim=1).squeeze(0)
            class_softmax = class_softmax.tolist()
            class_label_ = torch.argmax(class_label[bag_num], dim=1)
            # Output bag classification results and attention_weight for each patch
            f = open(output_file, 'a')
            f_writer = csv.writer(f, lineterminator='\n')

            # [slideID, true label, pred labl] + [y_prob]
            slideid_tlabel_plabel = [slideID[bag_num], int(class_label_), class_hat] + class_softmax
            f_writer.writerow(slideid_tlabel_plabel)
            pos_x = []
            pos_y = []
            for pos in pos_list:
                pos_x.append(int(pos[0]))
                pos_y.append(int(pos[1]))
            f_writer.writerow(pos_x) # Write coordinates
            f_writer.writerow(pos_y) # Write coordinates
            A = A[:, 0, -n_class:]
            attention_weights = A.cpu().squeeze(0)
            attention_weights_list = attention_weights
            for i in range(n_class):
                att_list = []
                for att in attention_weights_list[i]:
                    att_list.append(float(att))

                # Write the attention weight for each instance
                f_writer.writerow(att_list)
            f.close()

def test_model(cv_test, mag, seed):

    ##################SETTING#######################################
    INTERVIEW_DIM = 18
    BLOOD_DIN = 10
    FEATURE_DIM = 512
    NUM_CV = 5
    ################################################################

    device = 'cuda:0'
    random.seed(seed)

    train_all = []
    valid_all = []
    test_all = []
    train_label = []
    valid_label = []
    test_label = []
    list_subtype = ['DLBCL','FL','Reactive'] #use subtype

    num_data = []

    label = 0
    label_list = np.eye(len(list_subtype)).tolist()
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
        label += 1

    n_class = len(num_data)
    num_all = sum(num_data)


    df = pd.read_csv("clinical_record.csv")
    train_index = []
    for i, slideID in enumerate(train_all):
        index = df[df["No"] == int(slideID)].index
        if len(index) == 0:
            continue
        train_index.append(index[0])

    test_index = []
    for i, slideID in enumerate(test_all):
        index = df[df["No"] == int(slideID)].index
        if len(index) == 0:
            continue
        test_index.append(index[0])


    ##################################################
    #Complement missing values with the median of the training data
    #And normalize each column

    #Deleted due to processing that cannot be disclosed.
    ##################################################

    test_interview = test_table[columns_interview].values
    test_blood = test_table[columns_blood].values

    test_dataset = []
    for i, slideID in enumerate(test_all):
        test_dataset.append([slideID[, test_label[i], test_interview[i], test_blood[i]])

    #name of log file
    log = f'train_log/PersAM_log_{mag}_cv-{cv_test}_0.05_{seed}.csv'

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # select best model
    epoch_cnn = select_cnn(log)
    print(epoch_cnn)
    model_cnn = f'./model_params/PersAM_{mag}_cv-{cv_test}_epoch-{epoch_cnn}_0.05_{seed}.pth'

    #Create a directory to store the results
    makedir('test_result')
    result = f'test_result/PersAM_log_{mag}_cv-{cv_test}_{seed}_new.csv'

    #make model
    feature_ex = feature_extractor()
    attention = Correlation_encoder(feature_dim=FEATURE_DIM, num_class=n_class, device=device)
    mlp_fc1 = MLP_fc1(INTERVIEW_DIM, FEATURE_DIM)
    mlp_fc2 = MLP_fc1(BLOOD_DIN, FEATURE_DIM)

    model = PersAM(feature_ex, attention, mlp_fc1, mlp_fc2, n_class)
    model.load_state_dict(torch.load(model_cnn,map_location='cpu'))
    model = model.to(device)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])

    test_Dataset = Dataset.mmDataset(
        train=False,
        transform=transform,
        dataset=test_dataset,
        mag=mag,
        bag_num=50,
        bag_size=100,

    )

    test_loader = torch.utils.data.DataLoader(
        test_Dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
    )

    test(model, device, test_loader, result, n_class)
    print("finish")

if __name__ == '__main__':

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    args = sys.argv
    cv_test = int(args[1]) # fold used for test
    seed = int(args[2]) # seed
    mag = "40x" # magnitude

    test_model(cv_test, mag, seed)
