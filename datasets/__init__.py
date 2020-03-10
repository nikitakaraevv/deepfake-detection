from datasets.video_dataset import VideoDataset

def get_dataset(args):
    """get_dataset
    :param name:
    """
    return {
        'video' : VideoDataset,
        # feel free to add new datasets here
    }[args.dataset]