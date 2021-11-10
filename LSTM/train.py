from net import Net
import numpy as np
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
import argparse
import time
from prepare_data import OurDataset
from torch.utils.data import Dataset, DataLoader
import shutil
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', required=True, help='path to dataset')
    parser.add_argument('-l', '--labels', required=True, help='path to labels')
    parser.add_argument('-g', '--gpu', default=-100, type=int, help='GPU id to use.')
    parser.add_argument('-lr', '--learningRate', default=0.01, type=float, help='initial learning rate', dest='lr')
    parser.add_argument('-w', '--window', default=3, type=int, help='window')
    parser.add_argument('-e', '--epochs', default=70, type=int, help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=27, type=int, help='mini-batch size (default: 27)')
    parser.add_argument('-v', '--evaluation', default=10, type=int, help='steps for evaluation (default: 10)')
    args = parser.parse_args()
    return(args)


def save_checkpoint(state, is_best, filename):  # .tar
    # print('Saving '+filename+' ...')
    # torch.save(state, filename+'.pth')
    if is_best:
        # print('Saving best model ...')
        # shutil.copyfile(filename+'.pth', 'best_model_'+filename+'.pth')  #
        print('Saving '+filename+' ...')
        torch.save(state, 'best_model_'+filename+'.pth')


def train_model():

    args = vars(parse_args())

    if args['gpu']==(-100):
        device = torch.device('cpu')
    else:
        cudnn.benchmark = True
        torch.cuda.empty_cache()
        device = torch.device('cuda:'+str(args['gpu']))

    raw_data = np.genfromtxt(args['data'], delimiter=',')
    true_label = np.genfromtxt(args['labels'], delimiter=',')

    train_data = raw_data[:int(0.8*len(raw_data))]
    train_label = true_label[:int(0.8*len(true_label))]

    valid_data = raw_data[int(0.8*len(raw_data)):]
    valid_label = true_label[int(0.8*len(true_label)):]

    dataset_train = OurDataset(train_data,train_label,args['window'], device=device)
    dataset_valid = OurDataset(valid_data,valid_label,args['window'], device=device)

    train_loader = DataLoader(dataset_train, batch_size=args['batch_size'], shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset_valid, batch_size=args['batch_size'], shuffle=True, num_workers=0)

    model = Net(device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), args['lr'])
    criterion = nn.CrossEntropyLoss()

    total_elapsed_time = 0.0

    best_acc = 0
    best_train_score = np.infty
    train_losses, valid_losses = [], []
    train_accss, valid_accss = [], []

    for epoch in range(args['epochs']):
        print('epoch',str(epoch))
        dataset_train.permute()
        start_time = time.time()

        train_loss = 0.0
        train_corrects = 0
        train_total = 0

        val_acs, val_ls = [], []

        for steps, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()

            outcomes = model.forward(batch_x)
            outcomes = torch.reshape(outcomes,(-1,12))

            _, preds = torch.max(outcomes,1)

            loss = criterion(outcomes, batch_y.long())
            loss.backward()
            optimizer.step()

            train_total += len(batch_y)
            train_loss += loss.item()*len(batch_x)
            train_corrects += int(torch.sum(preds==batch_y))

            # EVALUATION
            if steps % args['evaluation'] == 0:
                val_loss = 0.0
                val_corrects = 0
                val_total = 0
                model.eval()

                with torch.no_grad():
                    for batch_x_val, batch_y_val in valid_loader:
                        outcomes_val = model.forward(batch_x_val)
                        outcomes_val = torch.reshape(outcomes_val,(-1,12))

                        _val, val_preds = torch.max(outcomes_val,1)

                        loss_valid = criterion(outcomes_val, batch_y_val.long())

                        val_total += len(batch_y_val)
                        val_loss += loss_valid.item()*len(batch_x_val)
                        val_corrects += int(torch.sum(val_preds==batch_y_val))


                    epoch_val_loss = val_loss/val_total
                    epoch_val_acc = np.float(val_corrects)/val_total

                    val_acs.append(epoch_val_acc)
                    val_ls.append(epoch_val_loss)

                    print('Validation Loss: {:6.4f}, Validation Accuracy: {:6.2f}%;'.format(epoch_val_loss,epoch_val_acc*100))

                model.train()

        epoch_train_loss = train_loss/train_total
        epoch_train_acc = np.float(train_corrects)/train_total
        epoch_elapsed_time = time.time()-start_time
        total_elapsed_time += epoch_elapsed_time

        train_losses.append(epoch_train_loss)
        train_accss.append(epoch_train_acc)

        valid_losses.append(np.average(val_ls))
        valid_accss.append(np.average(val_acs))

        is_best = epoch_train_acc > best_acc
        best_acc = max(epoch_train_acc, best_acc)

        print('Train Loss: {:6.4f}, Total Accuracy: {:6.2f}%, Time Consumption: {:6.2f} seconds;'.format(epoch_train_loss,epoch_train_acc*100, epoch_elapsed_time))

        save_checkpoint({'epoch': epoch, 'arch': 'LSTM', 'state_dict': model.state_dict(), 'best_acc': best_acc,
            'optimizer' : optimizer.state_dict()}, is_best,'LSTM')

    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.savefig('Losses.png')
    # plt.show()

    plt.figure()
    plt.plot(train_accss, label='Training accuracy')
    plt.plot(valid_accss, label='Validation accuracy')
    plt.legend(frameon=False)
    plt.savefig('Accuracy.png')

    return(total_elapsed_time)



total_time_train = train_model()

print('\nThe total time consdumption of this trainings is {:.2f} seconds.'.format(total_time_train))
