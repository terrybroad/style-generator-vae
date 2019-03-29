import argparse
import random

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable, grad
from torchvision import datasets, transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

import util
from data import sampler_loader, sample_data
from model import Encoder, StyledGenerator
from tensorboardX import SummaryWriter
from tqdm import tqdm

writer = SummaryWriter()
device = 'cpu'

def train(generator, discriminator, step_it, train_loader, val_loader, options, g_optimizer, e_optimizer):
    alpha_step = 1. / ( float(options.n_iters_per_step) * 1.)

    aggressive_flag = args.aggressive

    mse_criterion = nn.MSELoss()
    mse_criterion.to(device)
    util.requires_grad(generator, True)
    util.requires_grad(encoder, True)
    generator.train()
    encoder.train()

    pre_mi = best_mi = mi_not_improved = 0
    starting_it = n_epochs = 0

    if args.checkpoint is not None:
        starting_it = checkpoint['iter']
        n_epochs = checkpoint['n_epochs']
        pre_mi = checkpoint['pre_mi']
        best_mi = checkpoint['best_mi']
        mi_not_improved = checkpoint['mi_not_improved']
        prev_aggressive_flag = checkpoint['aggressive_flag']
        if(prev_aggressive_flag != aggressive_flag):
            print("overwriting aggressive flag from checkpoint state")

    for step in range( step_it, step_it + options.num_steps ):
        tr_data_loader = sample_data(train_loader, 4 * 2 ** step )
        valid_data_loader = sample_data(valid_loader, 4 * 2 ** step )
        dataset = iter(tr_data_loader)
        fixed_image, label = next(dataset)
        utils.save_image(
            fixed_image,
            'sample/'+str(step)+'_fixed.png',
            nrow=5,
            normalize=True,
            range=(-1, 1))
        
        for i in range(starting_it, options.n_iters_per_step ):
            n_iters = (step - step_it) * options.n_iters_per_step + i
            encoder.zero_grad()
            generator.zero_grad()
            g_optimizer.zero_grad()
            e_optimizer.zero_grad()

            alpha = min( 1, alpha_step * float(i) )
            
            try:
                real_image, label = next(dataset)
            except (OSError, StopIteration):
                dataset = iter(tr_data_loader)
                real_image, label = next(dataset)
            
            b_size = real_image.size(0)
            real_image = real_image.to(device)

            z, kld = encoder( real_image, step=step, alpha=alpha)
            reconstructed_image = generator(z, step=step, alpha=alpha)
            
            mse = mse_criterion( real_image, reconstructed_image )
            loss = mse + args.beta * kld.mean()
            loss.backward()

            if aggressive_flag:
                if not (n_iters % args.aggr_tr_ratio == 0):
                    e_optimizer.step()
                else:
                    g_optimizer.step()
            else:
                e_optimizer.step()
                g_optimizer.step()

            if (n_iters % len(tr_data_loader.dataset)) == 0:
                if n_iters != 0:
                    n_epochs += 1
            
            if aggressive_flag and (n_iters % args.mi_valid_count) == 0:
                with torch.no_grad():
                    cur_mi = util.calc_mutual_info(encoder, valid_data_loader, step, alpha, device)
                
                if cur_mi - best_mi < 0:
                    mi_not_improved += 1
                    if mi_not_improved == 5:
                        aggressive_flag = False
                        print("STOP BURNING")
                else:
                    best_mi = cur_mi

                pre_mi = cur_mi

            print("step: %d, itr: %d, mse: %.3f, kld: %.3f, loss: %.3f, mutual_info: %.3f" % (step, i, mse, kld.mean(), loss, pre_mi) )
            writer.add_scalar('data/mse', mse, n_iters)
            writer.add_scalar('data/kld', kld.mean(), n_iters)
            writer.add_scalar('data/loss', loss, n_iters)
            writer.add_scalar('data/mutual_info', pre_mi, n_epochs)
            writer.add_scalar('params/alpha', alpha, n_iters)
            writer.add_scalar('params/step_itr', step, n_iters)
            writer.add_scalar('params/image_size', 4 * 2 ** step, n_iters)
            writer.add_scalar('params/n_epochs', n_epochs, n_iters )

            if (i + 1) % 1000 == 0:
                with torch.no_grad():
                    z_f, kld_f = encoder( fixed_image, step=step, alpha=alpha)
                    reconstructed_fixed = generator(z_f, step=step, alpha=alpha)
                    utils.save_image(
                    reconstructed_fixed,
                    'sample/'+str(step)+'_'+str(i + 1).zfill(6)+'_reconstructed.png',
                    nrow=5,
                    normalize=True,
                    range=(-1, 1))

            if (i + 1) % 10000 == 0:
                torch.save({
                 'step': step,
                 'iter' : i,
                 'n_epochs' : n_epochs,
                 'random_seed' : random_seed,
                 'valid_size' : args.valid_size,
                 'aggressive_flag': aggressive_flag,
                 'pre_mi': pre_mi,
                 'best_mi': best_mi,
                 'mi_not_improved': mi_not_improved, 
                 'generator_state_dict':   generator.module.state_dict(),
                 'encoder_state_dict':   encoder.module.state_dict(),
                 'g_optimizer_state_dict': g_optimizer.state_dict(),
                 'e_optimizer_state_dict': e_optimizer.state_dict()
                }, 'checkpoint/'+str(step)+'_'+str(i + 1).zfill(6)+'.model')
        
        starting_it = 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggressive inference VAE with style-based generator')

    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--init-size', default=8, type=int, help='initial image size')
    parser.add_argument('--n_iters_per_step', type=int, default=100000, help='number of iterations per step')
    parser.add_argument('--num_steps', type=int, default=5, help='number of step increments increasing in resolution')
    parser.add_argument('--batch_size', type=int, default=15, help='batch size')
    parser.add_argument('--cuda', default=True, help='enables CUDA training')
    parser.add_argument('--checkpoint', default=None, type=str, help="path to checkpoint")
    parser.add_argument('--code_size', default=512, type=int, help="latent code dim")
    parser.add_argument('--beta', default = 0.2, type=float, help="beta parameter for KLD")
    parser.add_argument('--aggressive', default=True, help='apply aggressive training')
    parser.add_argument('--aggr_tr_ratio', default=100, type=int, help="ratio of inference to generator updates in aggressive inference training")
    parser.add_argument('--mi_valid_count', default=5000, type=int, help="number of iterations between calculating mutual information on validation set")
    parser.add_argument('--valid_size', default = 0.1, type=float, help="ratio of validation set size to training set size")


    args = parser.parse_args()
    
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False
        print("Cuda is no found on device so defaulting to running code on CPU")

    if args.valid_size < 0 or args.valid_size > 1:
        raise ValueError('valid_size must be between 0 and 1')

    device = torch.device('cuda' if args.cuda else 'cpu')

    generator = nn.DataParallel(StyledGenerator(args.code_size)).to(device)
    encoder = nn.DataParallel(Encoder(args.code_size)).to(device)


    g_optimizer = optim.Adam(generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group({'params': generator.module.style.parameters(), 'lr': args.lr * 0.01})
    e_optimizer = optim.Adam(
        encoder.module.parameters(), lr=args.lr, betas=(0.0, 0.99))

    step = args.init_size // 4 - 1
    random_seed = np.random.seed()

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        generator.module.load_state_dict(checkpoint['generator_state_dict'])
        encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        e_optimizer.load_state_dict(checkpoint['e_optimizer_state_dict'])
        step = checkpoint['step']
        args.valid_size = checkpoint['valid_size']
        random_seed = checkpoint['random_seed']

    data = datasets.ImageFolder(args.path)
    num_train = len(data)
    indices = list(range(num_train))
    split = int(np.floor(args.valid_size * num_train))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = sampler_loader(args.path, args.batch_size, train_sampler)
    valid_loader = sampler_loader(args.path, args.batch_size, valid_sampler)

    train(generator, encoder, step, train_loader, valid_loader, args, g_optimizer, e_optimizer )

    writer.export_scalars_to_json("%s/all_scalars.json" % 'samples')
    writer.close()
