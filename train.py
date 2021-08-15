import os, time, argparse

from utils.datasets import create_dataloader
from tqdm import tqdm
import torch
from torch import nn
from models.vgg import VGG
from models.alexnet import AlexNet
from utils.visual import plot_loss, plot_lr, plot_accuracy
from torchvision import models
from config.hyper import load_config


def train_loop(model, dataloader, lossfn, optimizer, epoch, epochs):
    model.train()
    pbar = tqdm(dataloader)
    print(('\n' + '%12s' * 3) % ('Epoch', 'gpu_mem', 'loss'))
    for X, y in pbar:
        # gpu spped up
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        # froward propagation
        pred = model(X)
        loss = lossfn(pred, y)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        
        # gpu mem retrieve
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        # process bar logs
        s = ('%12s' * 3) % ('%g/%g' % (epoch + 1, epochs), mem, '%.7g'% loss)
        pbar.set_description(s)
    return loss

def val_loop(model, dataloader, loss_fn, epoch, epochs):
    model.eval()
    batch = len(dataloader)
    size = len(dataloader.dataset)
    pbar = tqdm(dataloader)
    total_loss = 0.
    correct = 0
    with torch.no_grad():
        for X, y in pbar:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            pred = model(X)
            loss = loss_fn(pred, y).item()
            total_loss += loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%12s' * 3) % ('%g/%g' % (epoch + 1, epochs), mem, '%.7g'% loss)
            pbar.set_description(s)
    correct /= size
    print(f'Accuracy: {(100*correct):>0.1f}% \n')
    return total_loss / batch, 100 * correct



def main(opt):
    print(f'Parameters: {opt}')


    # args
    batch_size, img_size, epochs, workers, data, weights, lr, optimizer_type, name, class_num, hyper_config \
        = opt.batch_size, opt.img_size, opt.epochs, opt.workers, opt.data, opt.weights, opt.lr, opt.optimizer, \
            opt.name, opt.class_num, opt.hyper
    
    # check and make the path if not exists
    save_to = os.path.join('runs/train', name)
    if os.path.exists(save_to):
        name = time.strftime('%Y%m%d-%H-%M-%S', time.localtime())
        print(f'{save_to} is exists, choose a new name: {name}')
        save_to = os.path.join('runs/train', name)

    if not os.path.exists(save_to):
        os.makedirs(save_to)
    print(f'All results will be saved in {save_to}')

    


    # data
    train_loader = create_dataloader(data, train=True, batch_size=batch_size, num_workers=workers)
    val_loader = create_dataloader(data, train=False, batch_size=batch_size, num_workers=workers)

    # model
    if weights and weights != '':
        print(f'loading model from {weights}')
        model = torch.load(weights)
    else:
        # model = AlexNet(out_features=class_num)
        model = VGG(num_layers=16, out_features=class_num)
        # model = models.vgg16(pretrained=True)
        # model.classifier = nn.Sequential(
        #     nn.Linear(512 * 49, 4096),
        #     nn.Dropout(p=0.5),
        #     nn.ReLU(inplace=True),    
        #     nn.Linear(4096, 4096),
        #     nn.Dropout(p=0.5),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, class_num)
        # )
    
    print(model)
    
    # loss, optimizer
    lossfn = nn.CrossEntropyLoss()

    # use cuda if it is available
    if torch.cuda.is_available():
        model = model.cuda()
        lossfn = lossfn.cuda()  
    
    # hyper config
    hyper = load_config(hyper_config)
    optimizer = hyper.getOptimizer(model.parameters())
    scheduler = hyper.geLrScheduler(optimizer)
    # if optimizer_type == 'sgd':
    #     optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # elif optimizer_type == 'adam':
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    # # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # training
    best_loss = 0
    lrs = []
    train_losses = []
    val_losses = []
    accuracy = []
    for epoch in range(epochs):
        # training and validating
        train_loss = train_loop(model, train_loader, lossfn, optimizer, epoch, epochs)
        val_loss, accu = val_loop(model, val_loader, lossfn, epoch, epochs)
        
        # record the training
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
        accuracy.append(accu)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(accu)
        else:
            scheduler.step()
            

        if best_loss == 0 or best_loss > val_loss:
            best_loss = val_loss
            torch.save(model, os.path.join(save_to, 'best.pth'))
        
        if (epoch + 1) % 5 == 0:
            plot_loss(train_losses, val_losses, save_to)
            plot_lr(lrs, save_to)
            plot_accuracy(accuracy, save_to)
    torch.save(model, os.path.join(save_to, 'final.pth'))
    print(f'All results are saved in {save_to}')\

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/caltech256', help='data for training')
    parser.add_argument('--hyper', type=str, default='config/hyper.yaml', help='hyper parameters: including optimizer and scheduler')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weights', type=str, default='', help='data for training')
    parser.add_argument('--batch-size', type=int, default=64, help='batch-size')
    parser.add_argument('--img-size', type=int, default=224, help='input image size')
    parser.add_argument('--class-num', type=int, default=256, help='number of categories')
    parser.add_argument('--workers', type=int, default=0, help='number of workers')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer type, adam or sgd')
    parser.add_argument('--name', type=str, default='exp', help='name of the traning')
    opt = parser.parse_args()

    main(opt)
    