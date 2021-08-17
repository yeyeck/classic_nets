import os
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import ExponentialLR, StepLR, ReduceLROnPlateau, CosineAnnealingLR
import yaml


class HyperConfig(object):
    def __init__(self, cfg):
        super(HyperConfig, self).__init__()
        self.cfg = cfg
        print('Hyper Configurations: ')
        print(self.cfg)
    

    def getOptimizer(self, params):
        lr = self.cfg['lr']
        optimizer_conf = self.cfg['optimizer']
        optimizer_type = optimizer_conf['type']
        if  optimizer_type == 'Adam':
            betas = tuple(optimizer_conf['betas'])
            return Adam(params, lr=lr, betas=betas)
        elif optimizer_type == 'SGD':
            momentum = optimizer_conf['momentum']
            return SGD(params, lr=lr, momentum=momentum)
        elif optimizer_type == 'RMSprop':
            alpha = optimizer_conf['alpha']
            return RMSprop(params, lr=lr, alpha=alpha)
        else:
            return SGD(params, lr=lr, momomentum=0.9)
    
    def geLrScheduler(self, optimizer):
        scheduler_conf = self.cfg['scheduler']
        scheduler_type = scheduler_conf['type']
        if scheduler_type == 'ReduceLROnPlateau':
            mode = scheduler_conf['mode']
            factor = scheduler_conf['factor']
            patience = scheduler_conf['patience']
            cooldown = scheduler_conf['cooldown']
            min_lr = scheduler_conf['min_lr']
            return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, cooldown=cooldown, min_lr=min_lr, verbose=True)
        elif scheduler_type == 'ReduceLROnPlateau':
            gamma = scheduler_conf['gamma']
            return ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_type == 'StepLR':
            gamma = scheduler_conf['gamma']
            step_size = scheduler_conf['step_size']
            return StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'CosineAnnealingLR':
            T_max = scheduler_conf['T_max']
            eta_min = scheduler_conf['eta_min']
            return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)



