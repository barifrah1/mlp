import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch as torch
from torch import nn
from torch.nn import functional as F
import pdb
from tqdm import tqdm


id_num_str = '312425036'  # input("Please Enter your Israeli ID?")
if (len(id_num_str) != 9):
    print('ID should contain 9 digits')
if (id_num_str.isdigit() is False):
    print('ID should contain only digits')
id_num = list(id_num_str[-3:])
random_num = sum(list(map(int, id_num)))
random_num

np.random.seed(random_num)
torch.manual_seed(random_num)
x, y = make_moons(500, noise=0.2, random_state=random_num)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=random_num)

### START CODE HERE ###
# 2 dimensional graph: purple points - class 0 , yellow points - class1
"""
zline = y_train
xline = x_train[:, 0]
yline = x_train[:, 1]
plt.scatter(xline, yline, c=zline, cmap='viridis', linewidth=0.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.colorbar()
# plt.show()
"""
### START CODE HERE ###
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)
#x_train = x_train.double()


class MlpArgs():
    def __init__(self, lr, num_epochs):
        self.lr = lr
        self.num_epochs = num_epochs


class MlpLogisticClassifier(nn.Module):
    def __init__(self, in_features):
        super(MlpLogisticClassifier, self).__init__()
        """model_layers = [(nn.Linear(in_features, 1, bias=True))]
        model_layers.append(nn.Sigmoid())
        self.fc_module = nn.Sequential(*model_layers)"""
        self.linear = nn.Linear(in_features, 1, bias=True)

    def forward(self, x):
        x = x.float()
        x = torch.sigmoid(self.linear(x))
        return x

# infer function


def infer(net, data, criterion):
    net.eval()
    running_loss = 0
    auc = 0
    predictions = len(data[1])*[None]
    for i in range(len(data[0])):
        x = data[0][i]
        y = data[1][i].float()
        with torch.no_grad():
            pred = net(x).float()
            predictions[i] = pred
            loss = criterion(torch.tensor(pred.item()), y).item()
        running_loss += loss
    auc = roc_auc_score(data[1], predictions)
    return [running_loss / len(data[0]), auc]


def training_loop(
    args,
    net,
    train,
    validation=None,
    test=None,
    criterion_func=nn.BCELoss,
    optimizer_func=torch.optim.SGD,
):
    """The training runs here.
    args: a class instance that contains the arguments
    net: the network we're training
    tr_loader, val_loader, test_loader: dataloaders for the train, validation and test sets
    criterion: the loss function
    optimizer: the optimizer to be used"""
    criterion = criterion_func()
    optimizer = optimizer_func(net.parameters(), lr=args.lr)
    tr_loss, val_loss, auc_per_epoch, test_auc_per_epoch = [
        None] * args.num_epochs, [None] * args.num_epochs, [None] * args.num_epochs, [None] * args.num_epochs
    test_loss, untrained_test_loss = None, None
    predictions = len(train[1]) * [None]
    # Note that I moved the inferences to a function because it was too much code duplication to read.
    if test:
        untrained_test_loss = infer(net, test, criterion)
    for epoch in range(args.num_epochs):
        net.train()
        running_tr_loss = 0
        for i in tqdm(range(len(train[0]))):
            x = train[0][i]
            y = train[1][i].float()
            optimizer.zero_grad()
            pred = net(x)
            predictions[i] = pred
            loss = criterion(pred, torch.tensor([y]))
            loss.backward()
            optimizer.step()
            running_tr_loss += loss
        tr_loss[epoch] = running_tr_loss.item() / len(train[0])
        auc_per_epoch[epoch] = roc_auc_score(train[1], predictions)
        if validation:
            val_loss[epoch], test_auc_per_epoch[epoch] = infer(
                net, validation, criterion)

    if test:
        test_loss = infer(net, test, criterion)

    print(f"Done training for {args.num_epochs} epochs.")
    print(
        f"The BCELoss is {untrained_test_loss[0]:.2e} before training and {test_loss[0]:.2e} after training."
    )
    print(
        f"The auc is {untrained_test_loss[1]:.2e} before training and {test_loss[1]:.2e} after training."
    )
    print(
        f"The training and validation losses are "
        f"\n\t{tr_loss}, \n\t{val_loss}, \n\tover the training epochs, respectively."
    )

    print(
        f"training auc by epochs "
        f"\n\t{auc_per_epoch} \n\tover the training epochs, respectively."
    )
    print(
        f"test auc by epochs "
        f"\n\t{test_auc_per_epoch} \n\tover the training epochs, respectively."
    )

    return tr_loss, val_loss, test_loss, untrained_test_loss, auc_per_epoch, test_auc_per_epoch


def plot_loss_graph(loss_list, train_or_val: str):
    loss_values = loss_list
    epochs = range(1, len(loss_values)+1)

    plt.plot(epochs, loss_values, label=f'{train_or_val} Loss by epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_auc_graph(auc, train_or_val: str):
    epochs = range(1, len(auc)+1)
    plt.plot(epochs, auc, label=f'{train_or_val} auc by epoch')
    plt.xlabel('Epochs')
    plt.ylabel('auc')
    plt.legend()

    plt.show()


def plot_decision_boundary(bias, a1, a2, X, Y):
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn.linear_model

    # Retrieve the model parameters.
    b = bias
    w1, w2 = a1, a2
    # Calculate the intercept and gradient of the decision boundary.
    c = -b/w2
    m = -w1/w2

    # Plot the data and the classification with the decision boundary.
    xmin, xmax = -2, 3
    ymin, ymax = -2, 3
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    plt.plot(xd, yd, 'k', lw=1, ls='--')
    plt.fill_between(xd, yd, ymin, color='tab:orange', alpha=0.2)
    plt.fill_between(xd, yd, ymax, color='tab:blue', alpha=0.2)
    plt.scatter(*X[Y == 0].T, s=8, alpha=0.5)
    plt.scatter(*X[Y == 1].T, s=8, alpha=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.ylabel(r'$x_2$')
    plt.xlabel(r'$x_1$')

    plt.show()


in_features = x_train.shape[1]
net = MlpLogisticClassifier(in_features)
args = MlpArgs(1e-2, 20)
tr_loss, val_loss, test_loss, untrained_test_loss, auc_per_epoch, test_auc_per_epoch = training_loop(args,
                                                                                                     net,
                                                                                                     (x_train,
                                                                                                      y_train),
                                                                                                     (x_test,
                                                                                                      y_test),
                                                                                                     (x_test,
                                                                                                      y_test),
                                                                                                     nn.BCELoss
                                                                                                     )
plot_loss_graph(tr_loss, "training")
plot_loss_graph(val_loss, "validation")
plot_auc_graph(auc_per_epoch, "training")
plot_auc_graph(test_auc_per_epoch, "validation")
par = []
for names, params in net.named_parameters():
    par.append(params)
bias = par[1].item()
a1 = par[0][0][0].item()
a2 = par[0][0][1].item()

plot_decision_boundary(bias, a1, a2, x, y)
print("end!")
