import torch
import numpy as np
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from optparse import OptionParser
from model import UNet
import os
import dataset
import random

def trainNet(net, data_dir, sample_dir, cpt_dir, epochs=100, gpu=True, train=True, pth=None):

    criterion = torch.nn.MSELoss()
    if gpu:
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    if train:
        train_dataset = dataset.InpaintingDataSet(os.path.join(data_dir, 'train.png'), 1600)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=16,
                                                        shuffle=True,
                                                        num_workers=0)
        print('train items:', len(train_dataset))

        for epoch in range(0, epochs):
            print('Epoch %d/%d' % (epoch + 1, epochs))
            print('Training...')
            net.train()

            epoch_loss = 0

            for i, (inputs, targets) in enumerate(train_data_loader):
            
                optimizer.zero_grad()

                inputs_in = torch.transpose(inputs, 1, 3)

                if gpu:
                    inputs_in = Variable(inputs_in.cuda())
                    targets_in = Variable(targets.cuda())

                else:
                    inputs_in = Variable(inputs_in)
                    targets_in = Variable(targets)

                out = net.forward(inputs_in)

                loss = criterion(out, torch.transpose(targets_in, 1, 3))

                epoch_loss += loss.item()
                
                loss.backward()
                optimizer.step()

                #print('Training sample %d / %d - Loss: %.6f' % (i+1, 100, loss.item()))
                
            print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / (i+1)))

            if (epoch+1) == 1 or (epoch+1) == 5 or (epoch+1) == 10 or (epoch+1) == 50 or (epoch+1) == 100:
                showSample(inputs[0], targets[0], torch.transpose(out, 1, 3)[0], (epoch+1), sample_dir, train=True)

                torch.save(net.state_dict(), os.path.join(cpt_dir, 'CP%d.pth' % (epoch + 1)))
                print('Checkpoint %d saved !' % (epoch + 1))

    else:
        test_dataset = dataset.InpaintingDataSet(os.path.join(data_dir, 'test.png'), 5, train=False)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0)
        print('test items:', len(test_dataset))

        print('Testing', pth)
        net.load_state_dict(torch.load(os.path.join(cpt_dir, pth)))
        net.eval()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_data_loader):

                inputs_in = torch.transpose(inputs, 1, 3)

                if gpu:
                    inputs_in = Variable(inputs_in.cuda())
                else:
                    inputs_in = Variable(inputs_in)

                out = net.forward(inputs_in)

                showSample(inputs[0], targets[0], torch.transpose(out, 1, 3)[0], pth[2:-4], sample_dir, train=False)


def getArgs():
    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('--data-dir', dest='data_dir', default='./inpainting_set', help='data directory')
    parser.add_option('--sample-dir', dest='sample_dir', default='./samples', help='sample directory')
    parser.add_option('--cpt-dir', dest='cpt_dir', default='./checkpoints', help='checkpoint directory')
    parser.add_option('--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('--test', action='store_false', default=True, help='testing mode')
    parser.add_option('--pth', default='CP100.pth', help='pth')

    (options, args) = parser.parse_args()
    return options

"""
def showSample(input, target, out, epoch, sample_dir, train):
    t = 'train'
    if not train:
        t = 'test'

    plt.subplot(1, 3, 1).set_title(str(epoch)+'_'+t+'_gt')
    plt.imshow(target)
    plt.subplot(1, 3, 2).set_title(str(epoch)+'_'+t+'_in')
    plt.imshow(input[:,:,0:3])
    plt.subplot(1, 3, 3).set_title(str(epoch)+'_'+t+'_out')
    plt.imshow(out.cpu().detach().numpy())

    plt.show()
    plt.savefig(os.path.join(sample_dir,str(epoch)+'_'+t+'.png'))
"""

def showSample(input, target, out, epoch, sample_dir, train):
    t = 'train'
    if not train:
        t = 'test'

    # Eğer input (H, W, C) şeklindeyse (yani Channel son), permute etmeye gerek yok
    # Eğer input (C, H, W) şeklindeyse (PyTorch default), permute(1,2,0) yapman lazım

    if input.shape[0] == 4:  # (4, H, W) yani 4 kanal varsa (inpainting inputu)
        input_img = input[0:3, :, :].permute(1, 2, 0).cpu().detach().numpy()  # sadece RGB
    else:  # zaten (H, W, 3) olabilir
        input_img = input[:, :, 0:3].cpu().detach().numpy()

    if target.shape[0] == 3:
        target_img = target.permute(1, 2, 0).cpu().detach().numpy()
    else:
        target_img = target.cpu().detach().numpy()

    if out.shape[0] == 3:
        output_img = out.permute(1, 2, 0).cpu().detach().numpy()
    else:
        output_img = out.cpu().detach().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title(f'{epoch}_{t}_gt')
    plt.imshow(np.clip(target_img, 0, 1))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f'{epoch}_{t}_in')
    plt.imshow(np.clip(input_img, 0, 1))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f'{epoch}_{t}_out')
    plt.imshow(np.clip(output_img, 0, 1))
    plt.axis('off')

    plt.tight_layout()

    plt.savefig(os.path.join(sample_dir, f'{epoch}_{t}.png'))
    plt.close()



if __name__ == '__main__':
    args = getArgs()

    net = UNet()

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    if not os.path.exists(args.cpt_dir):
        os.makedirs(args.cpt_dir)

    trainNet(net=net,
        data_dir=args.data_dir,
        sample_dir=args.sample_dir,
        cpt_dir=args.cpt_dir,
        epochs=args.epochs,
        gpu=args.gpu,
        train=args.test,
        pth=args.pth)