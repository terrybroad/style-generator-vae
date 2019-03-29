import argparse
import random

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable, grad
from torchvision import datasets, transforms, utils

import util
from data import celeba_loader, sample_data
from model import Encoder, StyledGenerator
from tensorboardX import SummaryWriter
from tqdm import tqdm

writer = SummaryWriter()
device = 'cpu'

def train(generator, discriminator, step_it, loader, options, g_optimizer, e_optimizer, starting_it = 0 ):
    
    alpha_step = 1. / ( float(options.n_iters_per_step) * 1.)

    mse_criterion = nn.MSELoss()
    mse_criterion.to(device)
    util.requires_grad(generator, True)
    util.requires_grad(encoder, True)
    generator.train()
    encoder.train()

    for step in range( step_it, step_it + options.num_steps ):
        data_loader = sample_data(loader, 4 * 2 ** step )
        dataset = iter(data_loader)
        fixed_image, label = next(dataset)
        utils.save_image(
            fixed_image,
            'sample/'+str(step)+'_fixed.png',
            nrow=3,
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
                dataset = iter(data_loader)
                real_image, label = next(dataset)
            
            b_size = real_image.size(0)
            real_image = real_image.to(device)

            z, kld = encoder( real_image, step=step, alpha=alpha)
            reconstructed_image = generator(z, step=step, alpha=alpha)
            
            mse = mse_criterion( real_image, reconstructed_image )
            loss = mse + args.beta * kld.mean()
            loss.backward()
            e_optimizer.step()
            g_optimizer.step()
            print("step: %d, itr: %d, mse: %f, kld: %f, loss total: %f" % (step, i, mse, kld.mean(), loss) )
            writer.add_scalar('data/mse', mse, n_iters)
            writer.add_scalar('data/kld', kld.mean(), n_iters)
            writer.add_scalar('data/loss', loss, n_iters)
            writer.add_scalar('params/alpha', alpha, n_iters)
            writer.add_scalar('params/step_itr', step, n_iters)
            writer.add_scalar('params/image_size', 4 * 2 ** step, n_iters)

            if (i + 1) % 1000 == 0:
                with torch.no_grad():
                    z_f, kld_f = encoder( fixed_image, step=step, alpha=alpha)
                    reconstructed_fixed = generator(z_f, step=step, alpha=alpha)
                    utils.save_image(
                    reconstructed_fixed,
                    'sample/'+str(step)+'_'+str(i + 1).zfill(6)+'_reconstructed.png',
                    nrow=3,
                    normalize=True,
                    range=(-1, 1))

            if (i + 1) % 10000 == 0:
                torch.save({
                 'step': step,
                 'iter' : i,
                 'generator_state_dict':   generator.module.state_dict(),
                 'encoder_state_dict':   encoder.module.state_dict(),
                 'g_optimizer_state_dict': g_optimizer.state_dict(),
                 'e_optimizer_state_dict': e_optimizer.state_dict()
                }, 'checkpoint/'+str(step)+'_'+str(i + 1).zfill(6)+'.model')
        
        starting_it = 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE with style-based generator')

    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--init-size', default=8, type=int, help='initial image size')
    parser.add_argument('-d', '--data', default='celeba', type=str,
                        choices=['celeba'],
                        help=('Specify dataset. '
                            'Currently only CelebA supported'))
    parser.add_argument('--n_iters_per_step', type=int, default=100000, help='number of iterations per step')
    parser.add_argument('--num_steps', type=int, default=5, help='number of step increments increasing in resolution')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--cuda', default=True, help='enables CUDA training')
    parser.add_argument('--checkpoint', default=None, type=str, help="path to checkpoint")
    parser.add_argument('--code_size', default=512, type=int, help="latent code dim")
    parser.add_argument('--beta', default = 0.2, type=float, help="beta parameter for KLD")

    args = parser.parse_args()
    
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False
        print("Cuda is no found on device so defaulting to running code on CPU")


    device = torch.device('cuda' if args.cuda else 'cpu')

    generator = nn.DataParallel(StyledGenerator(args.code_size)).to(device)
    encoder = nn.DataParallel(Encoder(args.code_size)).to(device)


    g_optimizer = optim.Adam(generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group({'params': generator.module.style.parameters(), 'lr': args.lr * 0.01})
    e_optimizer = optim.Adam(
        encoder.module.parameters(), lr=args.lr, betas=(0.0, 0.99))

    step = args.init_size // 4 - 1
    starting_it = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        generator.module.load_state_dict(checkpoint['generator_state_dict'])
        encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        e_optimizer.load_state_dict(checkpoint['e_optimizer_state_dict'])
        step = checkpoint['step']
        starting_it = checkpoint['iter']

    if args.data == 'celeba':
        loader = celeba_loader(args.path, args.batch_size)
    else:
        raise ValueError('Only celeba is support as dataset currently: %s' % opt.model)


    train(generator, encoder, step, loader, args, g_optimizer, e_optimizer, starting_it=starting_it )

    writer.export_scalars_to_json("%s/all_scalars.json" % 'samples')
    writer.close()
