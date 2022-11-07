from functools import reduce
import torch
import torch.jit
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import DataLoader

dtype = torch.float

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Swish(nn.Module):
    
    def forward(self, x):
        return x * torch.sigmoid(x)

def parse_nonlinearity(non_linearity):
    """Parse non-linearity."""
    if hasattr(nn, non_linearity):
        return getattr(nn, non_linearity)
    elif hasattr(nn, non_linearity.capitalize()):
        return getattr(nn, non_linearity.capitalize())
    elif hasattr(nn, non_linearity.upper()):
        return getattr(nn, non_linearity.upper())
    elif non_linearity.lower() == "swish":
        return Swish
    else:
        raise NotImplementedError(f"non-linearity {non_linearity} not implemented")

def parse_layers(layers, in_dim, non_linearity):
    """Parse layers of nn."""
    nonlinearity = parse_nonlinearity(non_linearity)
    layers_ = list()
    in_dim = in_dim[0]

    for layer in layers:
        layers_.append(nn.Linear(in_dim, layer))
        layers_.append(nonlinearity())
        in_dim = layer

    return nn.Sequential(*layers_), in_dim

class FeedForwardNN(nn.Module):
    
    def __init__(
        self,
        in_dim,
        out_dim,
        layers=(200, 200),
        non_linearity="swish",
    ):
        super().__init__()
        self.kwargs = {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "layers": layers,
            "non_linearity": non_linearity,
        }

        self.hidden_layers, in_dim = parse_layers(layers, in_dim, non_linearity)
        self.output_shape = out_dim
        linear_layer = nn.Linear(
            in_dim, reduce(lambda x, y: x * y, list(out_dim))
        )
        final_non_linearity = nn.Softplus()
        self.final_layer = nn.Sequential(
            linear_layer,
            final_non_linearity,
            )

    def forward(self, x):
        hidden_output = self.hidden_layers(x)
        output = self.final_layer(hidden_output)
        return output

class NeuralConstitutiveModel(nn.Module):

    def __init__(
        self,
        in_dim_model_1 = (2, ),
        in_dim_model_2 = (3, ),
        layers_model_1 = (256, 256),
        layers_model_2 = (256, 256),
        ):
        super(NeuralConstitutiveModel, self).__init__()
        self.in_dim_model_1 = in_dim_model_1
        self.in_dim_model_2 = in_dim_model_2
        self.network_1, self.out_dim_1 = parse_layers(layers_model_1, in_dim=in_dim_model_1, non_linearity="swish")

        self.network_2, self.out_dim_2 = parse_layers(layers_model_2, in_dim=in_dim_model_2, non_linearity="swish")

        self.final_input_dim = self.out_dim_1 + self.out_dim_2

        self.final_network = FeedForwardNN(
            in_dim = (self.final_input_dim, ),
            out_dim = (1, ),
            layers = (256, 256, 256, 256, 256),
            )

    def forward(self, x):
        
        input_1 = x[..., :self.in_dim_model_1[0]]
        input_2 = x[..., self.in_dim_model_1[0]:self.in_dim_model_1[0] + self.in_dim_model_2[0]]
        
        out_1 = self.network_1(input_1)
        out_2 = self.network_2(input_2)

        input_final_model = torch.cat((out_1, out_2), dim=-1)

        out = self.final_network(input_final_model)

        return out        

def loadDataSplitTest(n_tiling_params, filename, shuffle = True, ignore_unconverging_result = True):
    all_data = []
    all_label = [] 
    
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")[:]]
        if (ignore_unconverging_result):
            if (item[-1] > 1e-6 or math.isnan(item[-1])):
                continue
            if (item[-5] < 1e-6 or item[-5] > 10):
                continue
            # if (np.abs(item[-3] - 1.001) < 1e-6 or np.abs(item[-3] - 0.999) < 1e-6):
            #     continue
        data = item[0:n_tiling_params]
        for i in range(2):
            data.append(item[n_tiling_params+i])
        data.append(2.0*item[n_tiling_params+2])
        # data.append(item[n_tiling_params+2])
        label = item[n_tiling_params+3:n_tiling_params+5]
        label.append(1.0 * item[n_tiling_params+5])
        label.append(item[n_tiling_params+6])
        
        all_data.append(data)
        all_label.append(label)
        
    print("#valid data:{}".format(len(all_data)))
    # exit(0)
    start = 0
    end = -1
    all_data = np.array(all_data[start:]).astype(np.float32)
    all_label = np.array(all_label[start:]).astype(np.float32) 
    
    
    indices = np.arange(all_data.shape[0])
    if (shuffle):
        np.random.shuffle(indices)
    
    all_data = all_data[indices]
    all_label = all_label[indices]
    
    return all_data, all_label


def relativeL2(y_true, y_pred):
    if (y_true.shape[1] > 1):
        stress_norm = torch.linalg.norm(y_true, ord=2, dim=1)
        stress_norm = stress_norm.unsqueeze(-1)
        
        stress_norm = stress_norm.repeat(1, 3)
        
        y_true_normalized = torch.divide(y_true, stress_norm)
        y_pred_normalized = torch.divide(y_pred, stress_norm)
        
        return torch.mean(torch.square(y_true_normalized - y_pred_normalized))
    else:
        y_true_normalized = torch.ones(y_true.shape).to(device)
        y_pred_normalized = torch.divide(y_pred, y_true)
        return torch.mean(torch.square(y_true_normalized - y_pred_normalized))

def generator(train_data, train_label):    
    indices = np.arange(train_data.shape[0])
    while True:
        np.random.shuffle(indices)
        yield train_data[indices], train_label[indices]

def trainStep(n_tiling_params, opt, model_inputs, 
                            model_supervision, model):
    opt.zero_grad()
    psi = model(model_inputs)
    
    loss_energy = relativeL2(model_supervision[:, -1:], psi)
    
    grad = torch.autograd.grad(psi, model_inputs, 
            grad_outputs=torch.ones(psi.shape[0], 1).to(device),
            retain_graph=True, only_inputs=True)[0]
    loss_grad = relativeL2(model_supervision[:, 0:3], grad[:,2:])
    total_loss  = loss_energy + loss_grad
    total_loss.backward()
    opt.step()
    return loss_grad, loss_energy

def testStep(n_tiling_params, model_inputs, 
                model_supervision, model):
    
    psi = model(model_inputs)
    
    loss_energy = relativeL2(model_supervision[:, -1:], psi)
    
    grad = torch.autograd.grad(psi, model_inputs, 
            grad_outputs=torch.ones(psi.shape[0], 1).to(device),
            retain_graph=False, only_inputs=True)[0]
    loss_grad = relativeL2(model_supervision[:, 0:3], grad[:,2:])
    total_loss  = loss_energy + loss_grad
    return loss_grad, loss_energy

def train(n_tiling_params, model_name, 
        train_data, train_label, validation_data, validation_label):

    model = NeuralConstitutiveModel()

    
    model = model.to(device)

    n_epoch = 100
    batch_size = 10000

    opt = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=1e-4
    )
    
    validation_data_tensor = torch.tensor(validation_data, requires_grad=True).to(device)
    validation_label_tensor = torch.tensor(validation_label).to(device)
    
    for epoch in range(n_epoch):
        
        lambdas, sigmas = next(generator(train_data, train_label))
        if batch_size == -1:
            batch = 1
        else:
            batch = int(np.floor(len(lambdas) / batch_size))
        
        train_loss_grad = 0.0
        train_loss_e = 0.0
        g_norm_sum = 0.0
        for i in range(batch):
            mini_bacth_lambdas = torch.tensor(lambdas[i * batch_size:(i+1) * batch_size], requires_grad=True).to(device)
            mini_bacth_sigmas = torch.tensor(sigmas[i * batch_size:(i+1) * batch_size]).to(device)

        
            grad, e = trainStep(n_tiling_params, opt, 
                            mini_bacth_lambdas, 
                            mini_bacth_sigmas, model)
            
            train_loss_grad += grad
            train_loss_e += e
        validation_loss_grad, validation_loss_e = testStep(n_tiling_params, 
                                                validation_data_tensor, 
                                                validation_label_tensor, model)

        print("epoch: {}/{} train_loss_grad: {} train_loss e: {}, validation_loss_grad:{} loss_e:{}".format(epoch, n_epoch, \
                train_loss_grad, train_loss_e, \
                validation_loss_grad, validation_loss_e))

    torch.cuda.empty_cache()
    model = model.double()
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save('model_scripted.pt') # Save
    # torch.save(model.module.state_dict(), "test")


        
    
if __name__ == "__main__":
    
    n_tiling_params = 2
    
    full_data = "/home/yueli/Documents/ETH/SandwichStructure/Server/all_data_IH21_shuffled.txt"  
    
    data_all, label_all = loadDataSplitTest(n_tiling_params, full_data, False, True)
    
    five_percent = int(len(data_all) * 0.05)
    
    train_data =  data_all[:-five_percent]
    train_label =  label_all[:-five_percent]

    validation_data = data_all[-five_percent:]
    validation_label = label_all[-five_percent:]
    model_name = "IH21"

    train(n_tiling_params, model_name, 
        train_data, train_label, validation_data, validation_label)