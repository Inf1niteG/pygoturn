# necessary imports
import os, sys
import time
import copy
import datasets
from datasets import ALOVDataset, ILSVRC2014_DET_Dataset
import argparse
import model
import torch
from torch.autograd import Variable
from torchvision import transforms
from helper import ToTensor, Normalize, show_batch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from helper import *
from multiprocessing.dummy import Pool as ThreadPool

# constants
use_gpu = torch.cuda.is_available()
kSaveModel = 20000 # save model after every 20000 steps
batchSize = 50 # number of samples in a batch
kGeneratedExamplesPerImage = 10; # generate 10 synthetic samples per image in a dataset
transform = transforms.Compose([Normalize(), ToTensor()])
enable_tensorboard = False
if enable_tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()

args = None
parser = argparse.ArgumentParser(description='GOTURN Training')
parser.add_argument('-n', '--num-batches', default=500000, type=int, help='number of total batches to run')
parser.add_argument('-lr', '--learning-rate', default=1e-6, type=float, help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay factor')
parser.add_argument('--momentum', default=0.9, type=float, help='optimizer momentum')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='optimizer momentum')
parser.add_argument('--lr-decay-step', default=100000, type=int, help='steps after which learning rate decays')
parser.add_argument('-save', '--save-directory', default='../saved_checkpoints/exp3/', type=str, help='path to save directory')
parser.add_argument('-lshift', '--lambda-shift-frac', default=5, type=float, help='lambda-shift for random cropping')
parser.add_argument('-lscale', '--lambda-scale-frac', default=15, type=float, help='lambda-scale for random cropping')
parser.add_argument('-minsc', '--min-scale', default=-0.4, type=float, help='min-scale for random cropping')
parser.add_argument('-maxsc', '--max-scale', default=0.4, type=float, help='max-scale for random cropping')
parser.add_argument('-seed', '--manual-seed', default=800, type=int, help='set manual seed value')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch-size', default=50, type=int, help='number of samples in batch (default: 50)')
parser.add_argument('--save-freq', default=20000, type=int, help='save checkpoint frequency (default: 20000)')

def main():

    global args, batchSize, kSaveModel, use_gpu
    args = parser.parse_args()
    print(args)
    batchSize = args.batch_size
    kSaveModel = args.save_freq
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.manual_seed)

    # load datasets
    alov = ALOVDataset('../data/alov300/imagedata++/',
                       '../data/alov300/alov300++_rectangleAnnotation_full/',
                       transform)
    imagenet = ILSVRC2014_DET_Dataset('../data/imagenet_img/',
                                       '../data/imagenet_bbox/',
                                       transform,
                                       args.lambda_shift_frac,
                                       args.lambda_scale_frac,
                                       args.min_scale,
                                       args.max_scale)
    # list of datasets to train on
    datasets = [alov, imagenet]

    # load model
    net = model.GoNet()
    loss_fn = torch.nn.L1Loss(size_average = False)
    if use_gpu:
        net = net.cuda()
        loss_fn = loss_fn.cuda()
    
    # initialize optimizer
    trainable_weights = []
    trainable_bias = []

    for name, param in net.classifier.named_parameters():
        if 'weight' in name:
            trainable_weights.append(param)
        elif 'bias' in name:
            trainable_bias.append(param)

    optimizer = optim.SGD(
        [
            {
                'params': trainable_weights,
                'lr': args.learning_rate * 10
            },
            {
                'params': trainable_bias,
                'lr': args.learning_rate * 20
            }
        ],
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if os.path.exists(args.save_directory):
        print('Directory %s already exists' % (args.save_directory))
    else:
        os.makedirs(args.save_directory)

    # start training
    net = train_model(net, datasets, loss_fn, optimizer)

    # save trained model
    path = os.path.join(args.save_directory, 'final_model.pth')
    torch.save(net.state_dict(), path)

# batch formation regimen pseudocode
# curr_idx = 0
# running_batch = np.zeros((50,4))
# while 1:
#    temp_batch = tr_batch()
#    print "curr_idx = ", curr_idx
#    if curr_idx + 11 <= 50:
#        running_batch[curr_idx:curr_idx+11] = temp_batch
#        curr_idx = curr_idx + 11
#    elif (curr_idx + 11) > 50:
#        count_in = 50-curr_idx
#        print "count_in = ", count_in
#        running_batch[curr_idx:curr_idx+count_in] = temp_batch[:count_in]
#        model_batch = running_batch
#        curr_idx = (curr_idx+11)%50
#        running_batch = np.zeros((50,4))
#        running_batch[:curr_idx] = temp_batch[count_in:]
#        break

def get_training_batch(running_batch_idx, running_batch, dataset):
    done = 0
    N = kGeneratedExamplesPerImage+1
    train_batch = None

    x1_batch, x2_batch, y_batch = make_transformed_samples(dataset, args)
    if running_batch_idx + N <= batchSize:
        running_batch['previmg'][running_batch_idx:running_batch_idx+N,:,:,:] = x1_batch
        running_batch['currimg'][running_batch_idx:running_batch_idx+N,:,:,:] = x2_batch
        running_batch['currbb'][running_batch_idx:running_batch_idx+N,:] = y_batch
        running_batch_idx = (running_batch_idx+N)
    elif running_batch_idx + N > batchSize:
        done = 1
        count_in = batchSize-running_batch_idx
        # print "count_in =", count_in
        if count_in > 0:
            running_batch['previmg'][running_batch_idx:running_batch_idx+count_in,:,:,:] = x1_batch[:count_in,:,:,:]
            running_batch['currimg'][running_batch_idx:running_batch_idx+count_in,:,:,:] = x2_batch[:count_in,:,:,:]
            running_batch['currbb'][running_batch_idx:running_batch_idx+count_in,:] = y_batch[:count_in,:]
            running_batch_idx = (running_batch_idx+N) % batchSize
            train_batch = running_batch
            running_batch = {'previmg': torch.Tensor(batchSize, 3, 227, 227),
                            'currimg': torch.Tensor(batchSize, 3, 227, 227),
                            'currbb': torch.Tensor(batchSize, 4)}
            running_batch['previmg'][:running_batch_idx,:,:,:] = x1_batch[count_in:,:,:,:]
            running_batch['currimg'][:running_batch_idx,:,:,:] = x2_batch[count_in:,:,:,:]
            running_batch['currbb'][:running_batch_idx,:] = y_batch[count_in:,:]
        else:
            train_batch = running_batch
            running_batch_idx = 0
            running_batch['previmg'][running_batch_idx:running_batch_idx+N,:,:,:] = x1_batch
            running_batch['currimg'][running_batch_idx:running_batch_idx+N,:,:,:] = x2_batch
            running_batch['currbb'][running_batch_idx:running_batch_idx+N,:] = y_batch
            running_batch_idx = (running_batch_idx+N)

    return running_batch, train_batch, done, running_batch_idx
            

def make_transformed_samples(dataset, args):

    idx = np.random.randint(dataset.len, size=1)[0]
    orig_sample = dataset.get_orig_sample(idx) # unscaled original sample (single image and bb)
    true_sample = dataset.get_sample(idx) # cropped scaled sample (two frames and bb)
    true_tensor = transform(true_sample)

    origimg = orig_sample['image']
    origbb = orig_sample['bb']
    x1_batch = torch.Tensor(kGeneratedExamplesPerImage + 1, 3, 227, 227)
    x2_batch = torch.Tensor(kGeneratedExamplesPerImage + 1, 3, 227, 227)
    y_batch = torch.Tensor(kGeneratedExamplesPerImage + 1, 4)

    # initialize batch with the true sample
    x1_batch[0,:,:,:] = true_tensor['previmg']
    x2_batch[0,:,:,:] = true_tensor['currimg']
    y_batch[0,:] = true_tensor['currbb']

    for i in range(kGeneratedExamplesPerImage):
        sample = {'image': origimg, 'bb': origbb}
        prevbb = random_crop(sample,
                             args.lambda_scale_frac,
                             args.lambda_shift_frac,
                             args.min_scale,
                             args.max_scale)

        # Crop previous image with height and width twice the prev bounding box height and width
        # Scale the cropped image to (227,227,3)
        crop_curr = transforms.Compose([CropCurr()])
        scale = Rescale((227,227))
        transform_prev = transforms.Compose([CropPrev(), scale])
        prev_img = transform_prev({'image':origimg, 'bb':origbb})['image']
        # Crop current image with height and width twice the prev bounding box height and width
        # Scale the cropped image to (227,227,3)
        curr_obj = crop_curr({'image':origimg, 'prevbb':prevbb, 'currbb':origbb})
        curr_obj = scale(curr_obj)
        curr_img = curr_obj['image']
        currbb = curr_obj['bb']
        currbb = np.array(currbb)
        sample = {'previmg': prev_img,
                'currimg': curr_img,
                'currbb' : currbb
                }
        sample = transform(sample)
        x1_batch[i+1,:,:,:] = sample['previmg']
        x2_batch[i+1,:,:,:] = sample['currimg']
        y_batch[i+1,:] = sample['currbb']

    return x1_batch, x2_batch, y_batch

def train_model(model, datasets, criterion, optimizer):

    global args, writer
    since = time.time()
    curr_loss = 0
    lr = args.learning_rate
    flag = False
    start_itr = 0
    running_batch_idx = 0
    running_batch = {'previmg': torch.Tensor(batchSize, 3, 227, 227),
                     'currimg': torch.Tensor(batchSize, 3, 227, 227),
                     'currbb': torch.Tensor(batchSize, 4)}
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)

    # resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_itr = checkpoint['itr']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            running_batch_idx = checkpoint['running_batch_idx']
            running_batch = checkpoint['running_batch']
            lr = checkpoint['lr']
            np.random.set_state(checkpoint['np_rand_state'])
            torch.set_rng_state(checkpoint['torch_rand_state'])
            print("=> loaded checkpoint '{}' (iteration {})"
                  .format(args.resume, checkpoint['itr']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if not os.path.isdir(args.save_directory):
        os.makedirs(args.save_directory)

    itr = start_itr
    st = time.time()
    while itr < args.num_batches:

        model.train()
        if args.resume and os.path.isfile(args.resume) and itr == start_itr and (not flag):
            checkpoint = torch.load(args.resume)
            i = checkpoint['dataset_indx']
            flag = True
        else:
            i = 0

        # train on datasets
        # usually ALOV and ImageNet
        while i < len(datasets):
            dataset = datasets[i]
            i = i+1
            running_batch, train_batch, done, running_batch_idx = get_training_batch(running_batch_idx, running_batch, dataset)
            if done:
                scheduler.step()
                
                x1 = train_batch['previmg']
                x2 = train_batch['currimg']
                y = train_batch['currbb']
                # wrap them in Variable
                if use_gpu:
                    x1, x2, y = Variable(x1.cuda()), \
                        Variable(x2.cuda()), Variable(y.cuda(), requires_grad=False)
                else:
                    x1, x2, y = Variable(x1), Variable(x2), Variable(y, requires_grad=False)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                output = model(x1, x2)
                loss = criterion(output, y)

                # backward + optimize
                loss.backward()
                optimizer.step()

                # statistics
                curr_loss = loss.item()
                end = time.time()
                itr = itr + 1
                print('[training] step = %d/%d, loss = %f, time = %f' % (itr, args.num_batches, curr_loss, end-st))
                sys.stdout.flush()
                st = time.time()

                if enable_tensorboard:
                    writer.add_scalar('train/batch_loss', curr_loss, itr)

                if itr > 0 and itr % kSaveModel == 0:
                    path = os.path.join(args.save_directory,
                                        'model_itr_' + str(itr) + '_loss_' + str(round(curr_loss, 3)) + '.pth.tar')
                    save_checkpoint({
                    'itr': itr,
                    'np_rand_state': np.random.get_state(),
                    'torch_rand_state': torch.get_rng_state(),
                    'l1_loss': curr_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'running_batch_idx': running_batch_idx,
                    'running_batch': running_batch,
                    'lr': lr,
                    'dataset_indx': i
                    }, path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    if enable_tensorboard:
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()
    return model

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

if __name__ == "__main__":
    main()
