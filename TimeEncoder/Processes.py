import torch
import torch.nn as nn
from Models_complex import Embedder, Estimator
from CustomLoss import WMAPELoss #SMAPELoss
import time
import os
import numpy as np
from utils import Logs

def train(train_loader, test_loader, learn_rate, hidden_dim, n_out, n_layers, batch_size, device, EPOCHS, n_steps, n_epochs, save_path, model_type, true_model,seq_len):

    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2] #2

    # Instantiating the models
    model_embedder = Embedder(input_dim, hidden_dim, n_layers, batch_size, device, model_type)
    model_estimator = Estimator(input_dim, hidden_dim, n_out, n_layers, batch_size, device, model_type, seq_len)

    if true_model!=False:
        true_model_embedder = os.path.join(true_model,'t_encoder_embedder_best.pth')
        true_model_estimator = os.path.join(true_model,'t_encoder_estimator_best.pth')
        model_embedder.load_state_dict(torch.load(true_model_embedder))
        model_estimator.load_state_dict(torch.load(true_model_estimator))

    model_embedder.to(device)
    model_estimator.to(device)

    criterion = nn.MSELoss()
    # criterion = SMAPELoss()
    optional_criterion = WMAPELoss()

    optimizer = torch.optim.Adam(list(model_estimator.parameters())+list(model_embedder.parameters()), lr=learn_rate)#, differentiable=True)

    best_loss_train = 10000000000
    best_loss_test = 10000000000

    logs = Logs(save_path, learn_rate, hidden_dim, n_out, n_layers, batch_size, EPOCHS, model_type, true_model, seq_len, best_loss_train, best_loss_test, model_embedder, model_estimator)#, optimizer)

    print("--> Starting Training")
    print("-->", str(model_type), "net")
    epoch_times = []
    epoch_train_losses = []
    epoch_test_losses = []

    # Start training loop
    for epoch in range(1,EPOCHS+1):
        start_time = time.perf_counter()
        model_embedder.train()
        model_estimator.train()
        avg_loss = 0.
        optional_avg_loss = 0.
        train_counter = 0
        for batch, label in train_loader:
            h_e = model_embedder.init_hidden()
            h_r = model_estimator.init_hidden()
            label = torch.reshape(label, (batch.size(0),1))
            train_counter += 1

            if model_type == "GRU":
                h_e = h_e.data
                h_r = h_r.data
            else:
                h_e = tuple([e.data for e in h_e])
                h_r = tuple([e.data for e in h_r])

            out_e, h_e = model_embedder(batch.to(device).float(), h_e)

            out_r, h_r = model_estimator(out_e.to(device).float(), h_r)

            train_loss = criterion(label.to(device).float(), out_r.to(device).float())
            optional_train_loss = optional_criterion(label.to(device).float(), out_r.to(device).float())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            avg_loss += train_loss.item()
            optional_avg_loss += optional_train_loss.item()

            if train_counter%n_steps == 0:
                print("Epoch {}......Step: {}/{}....... Average Train Estimator Loss for Epoch: {}   -->   WMAPE = %{}".format(epoch, train_counter, len(train_loader), avg_loss/train_counter, optional_avg_loss/train_counter*100))

        model_embedder.eval()
        model_estimator.eval()
        with torch.no_grad():
            test_loss = 0
            optional_test_loss = 0
            test_counter = 0

            for x_test, y_test in test_loader:
                test_counter += 1
                y_test = torch.reshape(y_test, (x_test.size(0),1))

                h_e = model_embedder.init_hidden()
                h_r = model_estimator.init_hidden()

                if model_type == "GRU":
                    h_e = h_e.data
                    h_r = h_r.data
                else:
                    h_e = tuple([e.data for e in h_e])
                    h_r = tuple([e.data for e in h_r])

                out_e, h_e = model_embedder(x_test.to(device).float(), h_e)

                out_r, h_r = model_estimator(out_e.to(device).float(), h_r)

                test_loss += criterion(y_test.to(device).float(), out_r.to(device).float()).item()
                optional_test_loss += optional_criterion(y_test.to(device).float(), out_r.to(device).float()).item()

            validation_loss = float(test_loss/test_counter)
            optional_validation_loss = float(optional_test_loss/test_counter)

        current_time = time.perf_counter()
        print("Epoch {}/{} Train Estimator Done, Total Loss: {}   -->   WMAPE = %{}".format(epoch, EPOCHS, avg_loss/len(train_loader), optional_avg_loss/train_counter*100))
        print("Epoch {}/{} Test Estimator Done, Total Loss: {}   -->   WMAPE = %{}".format(epoch, EPOCHS, validation_loss, optional_validation_loss*100))
        print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
        epoch_train_losses.append(avg_loss/len(train_loader))
        epoch_test_losses.append(validation_loss)

        best_loss_train, best_loss_test = logs.best_model(avg_loss/len(train_loader), validation_loss, model_estimator, model_embedder)#, optimizer)

        if epoch%n_epochs == 0:
            sv_path_e = save_path.split('.')[0]+'_embedder'+'_'+str(epoch)+'.'+save_path.split('.')[1]
            torch.save(model_embedder.state_dict(), sv_path_e)
            sv_path_r = save_path.split('.')[0]+'_estimator'+'_'+str(epoch)+'.'+save_path.split('.')[1]
            torch.save(model_estimator.state_dict(), sv_path_r)

    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    sv_path_e = save_path.split('.')[0]+'_embedder'+'_'+str(epoch)+'.'+save_path.split('.')[1]
    torch.save(model_embedder.state_dict(), sv_path_e)
    sv_path_r = save_path.split('.')[0]+'_estimator'+'_'+str(epoch)+'.'+save_path.split('.')[1]
    torch.save(model_estimator.state_dict(), sv_path_r)
    logs.losses(epoch_train_losses,epoch_test_losses)
    return(model_estimator)

def evaluate(model, test_data, device):
    model.eval()
    outputs = []
    targets = []
    start_time = time.perf_counter()
    for x, label in test_data:
        inp = torch.from_numpy(np.array(x))
        labs = torch.from_numpy(np.array(label))
        h = model.init_hidden()
        out, h = model(inp.to(device).float(), h)
        outputs.append(out.cpu().detach().numpy().reshape(-1))
        targets.append(labs.numpy().reshape(-1))
    print("Evaluation Time: {}".format(str(time.perf_counter()-start_time)))
    sMAPE = 0
    count = 0
    for i in range(len(outputs)):
        for y,y__ in zip(targets[i],outputs[i]):
            count+=1
            sMAPE += abs(y__-y)/(abs(y)+abs(y__))/2
    print("sMAPE: {}%".format(sMAPE/count*100))
    return outputs, targets, sMAPE/count*100
