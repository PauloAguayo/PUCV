import torch
import torch.nn as nn
from Models import TimeEncoder
import time


def best_model(val,reference,path, model):
    if val<reference:
        save_path = path.split('.')[0]+'_best.'+path.split('.')[1]
        torch.save(model, save_path)
        reference = val
    return(reference)


def train(train_loader, learn_rate, hidden_dim, n_out, n_layers, batch_size, device, EPOCHS, n_steps, n_epochs, save_path):#, model_type="GRU"):

    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    # Instantiating the models
    # if model_type == "GRU":
    #     model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    # else:
    #     model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)

    # model = TimeEncoder(input_dim, hidden_dim, n_out, n_layers, batch_size, device)
    model = torch.load('models/model_7/t_encoder_10_20_best.pth')
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    # print("Starting Training of {} model".format(model_type))
    print("Starting Training")
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
            h = h.data
            # if model_type == "GRU":
            #     h = h.data
            # else:
            #     h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter%n_steps == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
                best_model(float(avg_loss/counter),best_loss,save_path,model)
        current_time = time.perf_counter()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
        best_model(float(avg_loss/len(train_loader)),best_loss,save_path, model)
        if epoch%n_epochs == 0:
            save_path = save_path.split('.')[0]+'_'+str(epoch)+'.'+save_path.split('.')[1]
            torch.save(model, save_path)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    save_path = save_path.split('.')[0]+'_'+str(epoch)+'.'+save_path.split('.')[1]
    torch.save(model, save_path)
    return model

def evaluate(model, test_data):#, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.perf_counter()
    # for i in test_x.keys():
    for x, label in test_data:
        inp = torch.from_numpy(np.array(x))
        labs = torch.from_numpy(np.array(label))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        # outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        # targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
        outputs.append(out.cpu().detach().numpy().reshape(-1))
        targets.append(labs.numpy().reshape(-1))
    print("Evaluation Time: {}".format(str(time.perf_counter()-start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i]-targets[i])/(targets[i]+outputs[i])/2)/len(outputs)
    print("sMAPE: {}%".format(sMAPE*100))
    return outputs, targets, sMAPE
