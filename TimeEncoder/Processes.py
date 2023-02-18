import torch
import torch.nn as nn
from Models import TimeEncoder
# from Models_d import TimeEncoder
import time
import os
import numpy as np



def details(path,learn_rate, hidden_dim, n_out, n_layers, batch_size, EPOCHS, model_type, true_model, score):
    path_ = r"{}".format(path)
    lr = 'learning rate = '+str(learn_rate)
    hd = 'hidden dim = '+str(hidden_dim)
    no = 'n neurons output = '+str(n_out)
    nl = 'n layers = '+str(n_layers)
    bs = 'batch size = '+str(batch_size)
    e = 'epochs = '+str(EPOCHS)
    mt = 'model type = '+str(model_type)
    tm = 'previous model = '+str(true_model)
    be = 'best error = '+str(score)
    lines = [lr, hd, no, nl, bs, e, mt, tm, be]
    new_path = path_.split("\\")[:-1]
    with open(r"{}".format(os.path.join(*new_path,'details.txt')), 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

def best_model(val,reference,path, model, learn_rate, hidden_dim, n_out, n_layers, batch_size, EPOCHS, model_type, true_model):
    if val<reference:
        save_path = path.split('.')[0]+'_best.'+path.split('.')[1]
        torch.save(model, save_path)
        reference = val
        details(save_path,learn_rate, hidden_dim, n_out, n_layers, batch_size, EPOCHS, model_type, true_model, reference)
    return(reference)

def train(train_loader, learn_rate, hidden_dim, n_out, n_layers, batch_size, device, EPOCHS, n_steps, n_epochs, save_path, model_type, true_model):
    #global learn_rate, hidden_dim, n_out, n_layers, batch_size, EPOCHS, model_type, true_model

    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]

    # Instantiating the models
    if true_model!=False:
        model = torch.load(true_model)
    else:
        model = TimeEncoder(input_dim, hidden_dim, n_out, n_layers, batch_size, device, model_type)

    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    model.train()
    # print("Starting Training of {} model".format(model_type))
    print("--> Starting Training")
    print("-->", str(model_type), "net")
    epoch_times = []
    best_loss = 10000000000
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        start_time = time.perf_counter()
        h = model.init_hidden()
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            label = torch.reshape(label, (x.size(0),1))
            counter += 1
            #h = h.data
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter%n_steps == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
                best_loss = best_model(float(avg_loss/counter),best_loss,save_path,model, learn_rate, hidden_dim, n_out, n_layers, batch_size, EPOCHS, model_type, true_model)
        current_time = time.perf_counter()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
        best_loss = best_model(float(avg_loss/len(train_loader)),best_loss,save_path, model, learn_rate, hidden_dim, n_out, n_layers, batch_size, EPOCHS, model_type, true_model)
        if epoch%n_epochs == 0:
            sv_path = save_path.split('.')[0]+'_'+str(epoch)+'.'+save_path.split('.')[1]
            torch.save(model, sv_path)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    sv_path = save_path.split('.')[0]+'_'+str(epoch)+'.'+save_path.split('.')[1]
    torch.save(model, sv_path)
    return model

def evaluate(model, test_data, device):#, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.perf_counter()
    # for i in test_x.keys():
    for x, label in test_data:
        inp = torch.from_numpy(np.array(x))
        labs = torch.from_numpy(np.array(label))
        h = model.init_hidden()
        out, h = model(inp.to(device).float(), h)
        # outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        # targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
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
    return outputs, targets, sMAPE
