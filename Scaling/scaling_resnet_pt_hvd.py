from __future__ import print_function
import os
import argparse
import time
import socket
import wandb

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.utils.data.distributed

import horovod.torch as hvd

# This code train a ResNet model on random data and is used to measure scaling
# efficiency of data parallel training on various machines.
# This version uses Horovod to perform distributed training and was adapted from
# a version created by Denis Boyda at ALCF (2022) which used DDP

def init_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Scaling Example')

    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    parser.add_argument('--arch', metavar='ARCH', default='resnet18', # 152
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names))
    parser.add_argument('--name', default='test')
    parser.add_argument('--nnodes', type=int)
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--backend', default='nccl', choices=['nccl', 'mpi', 'gloo'],
                        help='Whether this is running on cpu or gpu')
    parser.add_argument('--data_fdir', default='/lus/grand/projects/datascience/boyda/data/imagenet_1k/',
                        help='Path of the imageNet data folder')
    args = parser.parse_args()
    return args


def init_MPI():
    # Mpi4py
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    local_rank = rank % torch.cuda.device_count()

    return rank, local_rank, size


def init_HVD():
    # Horovod
    #import horovod.torch as hvd
    hvd.init()
    hrank = hvd.rank()
    hrankl = hvd.local_rank()
    hsize = hvd.size()

    return hrank, hrankl, hsize


def main():
    args = init_args()
    rank, local_rank, size = init_MPI()
    hrank, local_hrank, hsize = init_HVD()
    #if rank == 0:
    #    wandb.init(project="scaling", entity="alcf-datascience", name=args.name)
    #    wandb.config.update(args)
    print(f"Rank {rank}, local rank {local_rank}, size {size}")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(int(local_rank))
    torch.cuda.manual_seed(args.seed)


    model = models.__dict__[args.arch]().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr * size)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters())

    pNum = sum(p.numel() for p in model.parameters())
    print("# parameters", pNum)

    def metric_average(val):
        avg = hvd.allreduce(val,average=True)
        return avg

    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def train():
        
        model.train()

        data = torch.randn(args.batch_size, 3, 224, 224, device='cuda')
        target = torch.zeros(args.batch_size, device='cuda', dtype=int)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        acc = accuracy(output, target)[0][0]
        loss = loss

        return loss, acc

    for epoch in range(args.epochs):
        t = time.time()
        loss, acc = train()
        loss = metric_average(loss).item()
        acc = metric_average(acc).item()
        t = time.time() - t

        #if rank == 0:
        #    wandb.log({
        #        'loss': loss,
        #        'acc': acc,
        #        'time': t,
        #        'throughput': args.batch_size * size / t
        #        })
        if rank == 0 and epoch>0:
            print("Epoch", epoch+1, ":")
            print("--> throughput:", args.batch_size * size / t)
            print("--> time:", t)
            print("--> loss:", loss)
            print("--> accuracy:", acc)
            print()

    

if __name__ == '__main__':
    assert torch.cuda.is_available()
    main()
