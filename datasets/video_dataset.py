
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/00_video_dataset.ipynb

import os
import PIL
import torch
import numpy as np
from path import Path
import pandas as pd
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, split,
                 phase=None,
                 video_len=30,
                 crop_size=160,
                 label_col='label',
                 transform=True,
                 ):

        self.video_dataframe = pd.read_csv(csv_file)
        self.video_dataframe = self.video_dataframe[self.video_dataframe.is_valid==(split=='test')].reset_index(drop=True)
        self.phase = split if phase is None else phase
        self.root_dir = Path(root_dir)
        self.lbl_cl = label_col
        self.video_len = video_len
        self.crop_size = crop_size
        self.transform=transform
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


    def __len__(self):
        return len(self.video_dataframe)

    def apply_transforms(self, video):
        if not self.transform: return video
        if 'train' == self.phase:
            # Resize
            resize = transforms.Resize(size=(self.crop_size+20, self.crop_size+20))
            video = [resize(im) for im in video]

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                video[0], output_size=(self.crop_size, self.crop_size))
            video = [TF.crop(im, i, j, h, w) for im in video]

            # Random horizontal flipping
            if random.random() > 0.5:
                video = [TF.hflip(im) for im in video]

            #    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            # Rotation
            angles = [-12, -6, 0, 6, 12]
            angle = random.choice(angles)
            video = [TF.rotate(im, angle) for im in video]


            #rotation_transform = MyRotationTransform()
            hue_factor = random.uniform(-0.2,0.2)
            video = [TF.adjust_hue(im, hue_factor) for im in video]
        else:
            resize = transforms.Resize(size=(self.crop_size, self.crop_size))
            video = [resize(im) for im in video]

        # Transform to tensor

        return [TF.normalize(TF.to_tensor(im),self.mean, self.std) for im in video]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_dir = self.root_dir/str(self.video_dataframe['name'][idx])
        sorted_frames = os.listdir(video_dir)
        sorted_frames.sort(key= lambda x : int(x.split('_')[-1][:-4]))

        video = ([PIL.Image.open(video_dir/im) for im in sorted_frames])

        assert len(video)>=self.video_len
        video=video[:self.video_len]
        video = torch.stack(self.apply_transforms(video))

        label = self.video_dataframe[self.lbl_cl][idx]

        data = (video, label)

        return tuple(data)

    def unprocess_video(self, video, plot=False):
        def unprocess_image(im):
            im = im.squeeze().numpy().transpose((1, 2, 0))
            im = self.std * im + self.mean
            im = np.clip(im, 0, 1)
            im = im * 255
            return PIL.Image.fromarray(im.astype(np.uint8))

        video = [unprocess_image(im) for im in video]

        if plot:
            video_len = len(video)
            fig = plt.figure(figsize=(3,3*video_len))
            for idx in range(video_len):
                ax = fig.add_subplot(video_len, 1, idx+1)
                ax.imshow(video[idx])
        else:
            return video