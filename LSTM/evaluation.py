from net import Net
import numpy as np
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import argparse
import time
import os
from datetime import datetime
from prepare_data import OurDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', required=True, help='path to dataset')
    parser.add_argument('-l', '--labels', required=True, help='path to labels')
    parser.add_argument('-g', '--gpu', default=-100, type=int, help='GPU id to use.')
    parser.add_argument('-lr', '--learningRate', default=0.01, type=float, help='initial learning rate', dest='lr')
    parser.add_argument('-w', '--window', default=3, type=int, help='window')
    parser.add_argument('-e', '--epochs', default=70, type=int, help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=27, type=int, help='mini-batch size (default: 27)')
    args = parser.parse_args()
    return(args)


def save_checkpoint(state, is_best, filename, dir):  # .tar
    if is_best:
        print('Saving '+filename+' ...')
        torch.save(state, os.path.join(dir,'best_model_'+filename+'.pth'))
        return('best_model_'+filename+'.pth')


def train_model():

    args = vars(parse_args())

    class_array = [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,20.]
    n_classes = len(class_array)

    if args['gpu']==(-100):
        device = torch.device('cpu')
    else:
        cudnn.benchmark = True
        torch.cuda.empty_cache()
        device = torch.device('cuda:'+str(args['gpu']))

    date = datetime.today().strftime('%Y-%m-%d')
    destination_folder = os.path.join('results',date,str(args['window'])+'W')

    if not os.path.exists(destination_folder):
        os.makedirs(os.path.join(destination_folder,str(0)))
        new_dir = os.path.join(destination_folder,str(0))
    else:
        for iter in range(100000):
            new_dir = os.path.join(destination_folder,str(iter))
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                break


    raw_data = np.genfromtxt(args['data'], delimiter=',')
    true_label = np.genfromtxt(args['labels'], delimiter=',')

    train_data = raw_data[:int(0.7*len(raw_data))]
    train_label = true_label[:int(0.7*len(true_label))]

    eval_data = raw_data[int(0.7*len(raw_data)):]
    eval_label = true_label[int(0.7*len(true_label)):]

    dataset_train = OurDataset(train_data,train_label,args['window'], device=device, characters=class_array)
    dataset_eval = OurDataset(eval_data,eval_label,args['window'], device=device, characters=class_array)

    train_loader = DataLoader(dataset_train, batch_size=args['batch_size'], shuffle=True, num_workers=0)
    eval_loader = DataLoader(dataset_eval, batch_size=len(eval_label), shuffle=True, num_workers=0)

    model = Net(device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), args['lr'])
    criterion = nn.CrossEntropyLoss()

    total_elapsed_time = 0.0

    best_acc = 0
    train_losses, train_accss = [], []

    for epoch in range(args['epochs']):
        print('epoch',str(epoch))
        dataset_train.permute()
        start_time = time.time()

        train_loss = 0.0
        train_corrects = 0
        train_total = 0

        for steps, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()

            outcomes = model.forward(batch_x)
            outcomes = torch.reshape(outcomes,(-1,n_classes))

            _, preds = torch.max(outcomes,1)

            loss = criterion(outcomes, batch_y.long())
            loss.backward()
            optimizer.step()

            train_total += len(batch_y)
            train_loss += loss.item()*len(batch_x)
            train_corrects += int(torch.sum(preds==batch_y))


                # model.train()

        epoch_train_loss = train_loss/train_total
        epoch_train_acc = np.float(train_corrects)/train_total
        epoch_elapsed_time = time.time()-start_time
        total_elapsed_time += epoch_elapsed_time

        train_losses.append(epoch_train_loss)
        train_accss.append(epoch_train_acc)

        is_best = epoch_train_acc > best_acc
        best_acc = max(epoch_train_acc, best_acc)

        print('Train Loss: {:6.4f}, Total Accuracy: {:6.2f}%, Time Consumption: {:6.2f} seconds;'.format(epoch_train_loss,epoch_train_acc*100, epoch_elapsed_time))

        filename = save_checkpoint({'epoch': epoch, 'arch': 'LSTM', 'state_dict': model.state_dict(), 'best_acc': epoch_train_acc,
            'optimizer' : optimizer.state_dict()}, is_best,'LSTM_'+str(args['window'])+'w', new_dir)

    filename = save_checkpoint({'epoch': epoch, 'arch': 'LSTM', 'state_dict': model.state_dict(), 'best_acc': best_acc,
        'optimizer' : optimizer.state_dict()}, is_best,'final_LSTM_'+str(args['window'])+'w', new_dir)

    model.load_state_dict(torch.load(os.path.join(new_dir,'best_model_LSTM_'+str(args['window'])+'w'+'.pth'), map_location=device)['state_dict'])
    model.to(device)

    ev_loss = 0.0
    ev_corrects = 0
    ev_total = 0
    model.eval()

    # with torch.no_grad():
    #     for batch_x_ev, batch_y_ev in eval_loader:
    #         outcomes_ev = model.forward(batch_x_ev)
    #         outcomes_ev = torch.reshape(outcomes_ev,(-1,n_classes))
    #
    #         _ev, ev_preds = torch.max(outcomes_ev,1)
    #
    #         loss_eval = criterion(outcomes_ev, batch_y_ev.long())
    #
    #         ev_total += len(batch_y_ev)
    #         ev_loss += loss_eval.item()*len(batch_x_ev)
    #         ev_corrects += int(torch.sum(ev_preds==batch_y_ev))
    #
    #
    #     epoch_ev_loss = ev_loss/ev_total
    #     epoch_ev_acc = np.float(ev_corrects)/ev_total
    #
    #     ev_acs.append(epoch_ev_acc)
    #     ev_ls.append(epoch_ev_loss)
    #
    #     print('Evaluation Loss: {:6.4f}, Evaluation Accuracy: {:6.2f}%;'.format(epoch_ev_loss,epoch_ev_acc*100))
    eval_losses, eval_accss = [], []


    ev_acs, ev_ls = [], []


    with torch.no_grad():
        for batch_x_eval, batch_y_eval in eval_loader:
            outcomes_eval = model.forward(batch_x_eval)
            outcomes_eval = torch.reshape(outcomes_eval,(-1,n_classes))

            _eval, eval_preds = torch.max(outcomes_eval,1)
            batch_y_eval = batch_y_eval.to('cpu')
            eval_preds = eval_preds.to('cpu')

            accuracy = accuracy_score(batch_y_eval, eval_preds)
            f1 = f1_score(batch_y_eval, eval_preds, average='macro')
            precision = precision_score(batch_y_eval, eval_preds, average='macro')
            recall = recall_score(batch_y_eval, eval_preds, average='macro')
            cm = confusion_matrix(batch_y_eval, eval_preds, labels=[class_array.index(k) for k in class_array])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_array)

            plt.figure()
            disp.plot()
            plt.savefig(os.path.join(new_dir,'ConfusionMatrix_'+str(args['window'])+'w.png'))

    print('Accuracy = {:6.4f}%, Precision = {:6.4f}%, Recall = {:6.4f}%, F1 = {:6.4f}%;'.format(accuracy*100,precision*100, recall*100, f1*100))

    file1 = open(os.path.join(new_dir,'metrics.txt'),"w")
    L = ['Accuracy = {:6.4f}% \n'.format(accuracy*100), 'Precision = {:6.4f}% \n'.format(precision*100), 'Recall = {:6.4f}% \n'.format(recall*100), 'F1 = {:6.4f}% \n'.format(f1*100)]
    file1.writelines(L)
    file1.close()


    plt.figure()
    plt.plot(train_losses, label='Training loss')
    # plt.plot(eval_losses, label='Evaluation loss')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(new_dir,'Losses_'+str(args['window'])+'w.png'))

    plt.figure()
    plt.plot(train_accss, label='Training accuracy')
    # plt.plot(eval_accss, label='Evaluation accuracy')
    plt.axis([0, 1, 0, args['epochs']])
    plt.legend(frameon=False)
    plt.savefig(os.path.join(new_dir,'Accuracy_'+str(args['window'])+'w.png'))

    return(total_elapsed_time)




total_time_train = train_model()

print('\nThe total time consumption of this trainings is {:.2f} seconds.'.format(total_time_train))
