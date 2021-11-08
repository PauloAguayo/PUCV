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

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', required=True, help='path to dataset')
    parser.add_argument('-l', '--labels', required=True, help='path to labels')
    parser.add_argument('-g', '--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('-lr', '--LearningRate', default=0.01, type=float, help='initial learning rate', dest='lr')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('-w', '--window', default=3, type=int, help='window')
    parser.add_argument('-e', '--epochs', default=70, type=int, help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=27, type=int, help='mini-batch size (default: 64)')
    args = parser.parse_args()
    return(args)


# def train_model(raw_data, true_label, model, crit, opti, n_epoch):
def train_model():

    args = vars(parse_args())

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
    lm = 10

    best_acc1 = 0
    best_train_score = np.infty
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    for epoch in range(args['epochs']):
        print('epoch',str(epoch))
        dataset_train.permute()
        start_time = time.time()

        train_loss = 0.0
        train_corrects = 0
        train_total = 0

        for steps, (batch_x, batch_y) in enumerate(train_loader):
            # print('Batch',str(steps),'of',str(len(train_loader)))
            optimizer.zero_grad()

            outputs = model.forward(batch_x)
            outputs = torch.reshape(outputs,(-1,12))

            _, preds = torch.max(outputs,1)

            loss = criterion(outputs, batch_y.long())
            loss.backward()
            optimizer.step()

            train_total += len(batch_y)
            train_loss += loss.item()*len(batch_x)
            train_corrects += int(torch.sum(preds==batch_y))

            if steps % lm == 0:
                # print('Evaluation')
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for em,(inputs, labels) in enumerate(valid_loader):
                        # inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        logps = torch.reshape(logps,(-1,12))
                        batch_loss = criterion(logps, labels.long())
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # train_losses.append(running_loss/len(train_loader))
                # test_losses.append(test_loss/len(test_loader))
                #
                # score = float(accuracy/len(test_loader))
                # train_score = float(running_loss/print_every)
                # print(f"Epoch {epoch+1}/{args['epochs']}.. "
                #       f"Train loss: {running_loss/print_every:.3f}.. "
                #       f"Test loss: {test_loss/len(test_loader):.3f}.. "
                #       f"Test accuracy: {accuracy/len(test_loader):.3f}")
                # running_loss = 0
                model.train()



        epoch_train_loss = train_loss/train_total
        epoch_train_acc = np.float(train_corrects)/train_total
        epoch_elapsed_time = time.time()-start_time
        total_elapsed_time += epoch_elapsed_time

        print('Train Loss: {:6.4f}, Total Accuracy: {:6.2f}%, Time Consumption: {:6.2f} seconds;'.format(epoch_train_loss,epoch_train_acc*100, epoch_elapsed_time))

    return(total_elapsed_time)


total_time_train = train_model()

print('\nThe total time consdumption of this trainings is {:.2f} seconds.'.format(total_time_train))
