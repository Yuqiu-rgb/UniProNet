import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import UniProNet_model
import argparse
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, roc_auc_score
import os
from torch.optim import *
from read_data import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True



def test_eval(test, test_embedding, test_labels, model):
    Result, _ = model(test, test_embedding)
    Result_softmax = F.softmax(Result, dim=1)  # Apply softmax to the output
    _, predicted = torch.max(Result_softmax, 1)
    correct = 0
    correct += (predicted == test_labels).sum().item()
    return 100 * correct / Result.shape[0], Result_softmax


def test_eval_str(test, test_embedding, test_str_embedding, test_labels, model):
    Result, _ = model(test, test_embedding, test_str_embedding)
    Result_softmax = F.softmax(Result, dim=1)  # Apply softmax to the output
    _, predicted = torch.max(Result_softmax, 1)
    correct = 0
    correct += (predicted == test_labels).sum().item()
    return 100 * correct / Result.shape[0], Result_softmax


def addbatch(data, label, batchsize):
    data = TensorDataset(data, label)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=True)

    return data_loader


def addbatcht(data, embedding, str_embedding, label, batchsize):
    data = TensorDataset(data, embedding, str_embedding, label)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=True)

    return data_loader


def save_model_test(model_dict, fold, auc, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # filename = 'test_AUC[{:.3f}], {}.pt'.format(test_auc, save_prefix)
    filename = 'fold{}_BERT_model.pt'.format(fold)
    save_path_pt = os.path.join(save_dir, filename)
    print('save_path_pt', save_path_pt)
    torch.save(model_dict, save_path_pt, _use_new_zipfile_serialization=False)
    print('Save Model Over: {}, AUC: {:.3f}\n'.format(save_prefix, auc))


def training(fold, model, device, epochs, criterion, optimizer,
             traindata,
             val, val_embedding, val_str_embedding, val_labels, scheduler, **kwargs):
    running_loss = 0
    max_performance = 0
    ReduceLR = False
    model.train()
    for epoch in range(epochs):
        if epoch < warmup_steps:
            scheduler.step()
        elif epoch == warmup_steps:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=0.5, patience=1,
                                                                   threshold=1e-3,
                                                                   threshold_mode='abs')
            ReduceLR = True
        for step, (inputs, embedding, str_embedding, labels) in enumerate(traindata):
            # print ("inputs.shape=",inputs.shape)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            embedding = embedding.to(device, non_blocking=True)
            str_embedding = str_embedding.to(device, non_blocking=True)
            model = model.to(device, non_blocking=True)
            outputs, _ = model(inputs, embedding, str_embedding)
            loss = get_val_loss(outputs, labels, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        acc, val_result = test_eval_str(val, val_embedding, val_str_embedding, val_labels, model)
        auc = roc_auc_score(val_labels.cpu().detach().numpy(),
                            val_result[:, 1].cpu().detach().numpy())
        if auc - max_performance > 1e-4 and epoch > 50:
            print("best_model_save")
            '''PO'''
            # save_model_test(model.state_dict(), fold, auc, '../model/PO_train',
            #                 parameters.learn_name)
            # # Open the file using "append" mode to add content to it
            # with open("../model/PO_train/save_result.txt", "a") as f:
            #     # Format the content to be written as a string
            #     result_str = "save model: epoch {} - iteration {}: average loss {:.3f} val_acc {:.3f} val_auc {:.3f} learning rate {:.2e}\n".format(
            #         epoch + 1, step + 1, running_loss, acc, auc,
            #         optimizer.param_groups[0]['lr'])
            #     #Write the string to the file
            #     f.write(result_str)
            # '''C'''
            # save_model_test(model.state_dict(), fold, auc, '../model/C_train',
            #                 parameters.learn_name)
            # # Open the file using "append" mode to add content to it
            # with open("../model/C_train/save_result.txt", "a") as f:
            #     # Format the content to be written as a string
            #     result_str = "save model: epoch {} - iteration {}: average loss {:.3f} val_acc {:.3f} val_auc {:.3f} learning rate {:.2e}\n".format(
            #         epoch + 1, step + 1, running_loss, acc, auc,
            #         optimizer.param_groups[0]['lr'])
            #     #Write the string to the file
            #     f.write(result_str)
            '''C2'''
            save_model_test(model.state_dict(), fold, auc, '../model/C2_train',
                            parameters.learn_name)
            # Open the file using "append" mode to add content to it
            with open("../model/C2_train/save_result.txt", "a") as f:
                # Format the content to be written as a string
                result_str = "save model: epoch {} - iteration {}: average loss {:.3f} val_acc {:.3f} val_auc {:.3f} learning rate {:.2e}\n".format(
                    epoch + 1, step + 1, running_loss, acc, auc,
                    optimizer.param_groups[0]['lr'])
                #Write the string to the file
                f.write(result_str)

            print(result_str, "\n")
            max_performance = auc
            # # if epoch >= warmup_steps:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.2

        if epoch % 5 == 0:
            print(
                "epoch {} - iteration {}: average loss {:.3f} val_acc {:.3f} val_auc {:.3f} learning rate {:.2e}".format(
                    epoch + 1, step + 1, running_loss, acc, auc, optimizer.param_groups[0]['lr']))
        running_loss = 0
        # if ReduceLR:
        #     scheduler.step(auc)


def train_validation(parameters,
                     x_train_encoding, x_train_embedding, x_train_str_embedding, train_label,
                     device, epochs, criterion, k_fold, learning_rate, batchsize):
    # skf = KFold(n_splits=k_fold,shuffle=True,random_state=15)
    skf = StratifiedShuffleSplit(n_splits=k_fold, random_state=42)  # 42
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_encoding, train_label.cpu())):
        model = UniProNet_model(parameters).to(device, non_blocking=True)

        print('**' * 10, 'Fold', fold + 1, 'Processing...', '**' * 10)
        x_train = x_train_encoding[train_idx].to(device, non_blocking=True)
        x_train_label = train_label[train_idx].to(device, non_blocking=True)
        x_val = x_train_encoding[val_idx].to(device, non_blocking=True)
        x_val_label = train_label[val_idx].to(device, non_blocking=True)
        x_val_label.index = range(len(x_val_label))

        embedding = x_train_embedding[train_idx].to(device, non_blocking=True)
        x_val_embedding = x_train_embedding[val_idx].to(device, non_blocking=True)
        str_embedding = x_train_str_embedding[train_idx].to(device, non_blocking=True)
        x_val_str_embedding = x_train_str_embedding[val_idx].to(device, non_blocking=True)

        train_data_loader = addbatcht(x_train, embedding, str_embedding, x_train_label, batchsize)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate, weight_decay=1e-4)
        warmup_scheduler = WarmupScheduler(optimizer, warmup_steps)
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        training(fold, model, device, epochs, criterion, optimizer,
                 train_data_loader,
                 x_val, x_val_embedding, x_val_str_embedding, x_val_label, warmup_scheduler,
                 args=parameters)



def BERT_encoding(txt_array, test_array):
    txt_seq_length = 0
    test_seq_length = 0
    txt_number = len(txt_array)
    test_number = len(test_array)
    for i in range(txt_number):
        txt_array[i] = txt_array[i] + '2'
        if len(txt_array[i]) > txt_seq_length:
            txt_seq_length = len(txt_array[i])
    for i in range(test_number):
        if len(test_array[i]) > test_seq_length:
            test_seq_length = len(test_array[i])
    seq_length = max(txt_seq_length, test_seq_length)
    x = np.zeros([txt_number, seq_length])
    nuc_d = {'0': 0, '2': 2, 'B': 4, 'Q': 5, 'I': 6, 'D': 7, 'M': 8, 'V': 9, 'G': 10, 'K': 11,
             'Y': 12, 'P': 13, 'H': 14, 'Z': 15, 'W': 16, 'U': 17, 'A': 18, 'N': 19, 'F': 20,
             'R': 21, 'S': 22, 'C': 23, 'E': 24, 'L': 25, 'T': 26, 'X': 27}
    for seq_index in range(txt_number):
        seq = txt_array[seq_index].upper()
        seq = seq[0 - seq_length:].ljust(seq_length, '0')

        for n, base in enumerate(seq):
            x[seq_index][n] = nuc_d[base]
    x = np.array(x, dtype='int64')
    CLS = np.ones([txt_number, 1])
    SEP = 2 * np.ones([txt_number, 1])
    Pad = np.zeros([txt_number, 1])
    x = np.concatenate([CLS, x, SEP, Pad], axis=1)
    x = torch.LongTensor(x)
    return x

class WarmupScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=0,
                        choices=list(range(torch.cuda.device_count())),
                        help='ordinal number of the GPU to use for computation')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Using {}".format(device))

    '''PO'''
    # train, test = data_read_po()
    # x_train, x_test = train.iloc[:, 1], test.iloc[:, 1]
    # train_label, test_label = train.iloc[:, 0], test.iloc[:, 0]
    #
    # x_train_encoding = BERT_encoding(x_train, x_test).to(device, non_blocking=True)
    # x_test_encoding = BERT_encoding(x_test, x_train).to(device, non_blocking=True)
    #
    # x_train_embedding, x_test_embedding = embedding_load_pO()
    # x_train_str_embedding, x_test_str_embedding = embedding_str_load_pO()
    '''C'''
    # train, test = data_read_pc()
    # x_train, x_test = train.iloc[:, 1], test.iloc[:, 1]
    # train_label, test_label = train.iloc[:, 0], test.iloc[:, 0]
    #
    # x_train_encoding = BERT_encoding(x_train, x_test).to(device, non_blocking=True)
    # x_test_encoding = BERT_encoding(x_test, x_train).to(device, non_blocking=True)
    #
    # x_train_embedding, x_test_embedding = embedding_load_pc()
    # x_train_str_embedding, x_test_str_embedding = embedding_str_load_pc()
    '''C2'''
    train, test = data_read_pc2()
    x_train, x_test = train.iloc[:, 1], test.iloc[:, 1]
    train_label, test_label = train.iloc[:, 0], test.iloc[:, 0]

    x_train_encoding = BERT_encoding(x_train, x_test).to(device, non_blocking=True)
    x_test_encoding = BERT_encoding(x_test, x_train).to(device, non_blocking=True)

    x_train_embedding, x_test_embedding = embedding_load_pc2()
    x_train_str_embedding, x_test_str_embedding = embedding_str_load_pc2()

    '''---'''
    train_label = torch.tensor(np.array(train_label, dtype='int64')).to(device, non_blocking=True)
    test_label = torch.tensor(np.array(test_label, dtype='int64')).to(device, non_blocking=True)

    import config

    parameters = config.get_train_config()
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = UniProNet_model.LossFunction()
    
    print("train.shape = ", x_train_encoding.shape)
    print("train_label.shape = ", train_label.shape)

    torch.manual_seed(142)  # 142，3407.42
    traindata = addbatch(x_train_encoding, train_label, 256)
    warmup_steps = 5
    train_validation(parameters,
                     x_train_encoding,
                     x_train_embedding,
                     x_train_str_embedding,
                     train_label,
                     device,
                     100,
                     criterion,
                     5,
                     1e-4,
                     256)
    print('--end--')
