import sys
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='')

    #  experiment settings
    # name of the experiment
    parser.add_argument('--dataset', default='video', type=str,
                        help='name of dataset to train upon')
    parser.add_argument('--root_dir', default='../filtered_ds/', type=str,
                            help='name of dataset to train upon')
    parser.add_argument('--csv_file', default='videos_df', type=str,
                            help='name of dataset to train upon')
    parser.add_argument('--crop_size', default=224, type=int,
                            help='name of dataset to train upon')
    parser.add_argument('--batch_size', default=4, type=int,
                            help='name of dataset to train upon')
    parser.add_argument('--workers', default=4, type=int,
                            help='name of dataset to train upon')
    parser.add_argument('--embed_dim', default=512, type=int,
                            help='name of dataset to train upon')
    parser.add_argument('--pretrained', default=True, type=bool,
                            help='name of dataset to train upon')
    parser.add_argument('--dropout', default=0.3, type=float,
                            help='name of dataset to train upon')
    parser.add_argument('--model_name', default='resnet50', type=str,
                            help='name of dataset to train upon')
    parser.add_argument('--num_classes', default=2, type=int,
                            help='name of dataset to train upon')
    parser.add_argument('--lr', default=1e-3, type=float,
                            help='name of dataset to train upon')
    parser.add_argument('--epochs', default=50, type=int,
                            help='name of dataset to train upon')
    parser.add_argument('--log_interval', default=10, type=int,
                            help='name of dataset to train upon')
    parser.add_argument('--save_model_path', default='./checkpoints/', type=str,
                            help='name of dataset to train upon')


    args = parser.parse_args()
    """
    # update args
    args.data_dir = '{}/{}'.format(args.root_dir, args.dataset)
    args.log_dir = '{}/runs/{}/'.format(args.data_dir, args.name)
    args.res_dir = '%s/runs/%s/res' % (args.data_dir, args.name)
    args.out_pred_dir = '%s/runs/%s/pred' % (args.data_dir, args.name)
    
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    """
    assert args.root_dir is not None
    assert args.num_classes > 0
    
    print(' '.join(sys.argv))
    print(args)

    return args
