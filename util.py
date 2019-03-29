import torch
from torch import nn, optim
from torch.autograd import Variable, grad

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    code taken from https://github.com/jxhe/vae-lagging-encoder/blob/master/modules/utils.py
    """ 
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)

def calc_mutual_info(model, test_loader, step, alpha, device):
    mi = 0
    num_examples = 0
    for datum in test_loader:
        batch_data, _ = datum
        batch_data = batch_data.to(device)
        batch_size = batch_data.size(0)
        num_examples += batch_size
        mutual_info = model.module.calc_mutual_info(batch_data, step, alpha)
        mi += mutual_info * batch_size

    return mi / num_examples