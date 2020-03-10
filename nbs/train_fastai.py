from fastai.vision import *
import pandas as pd

data_path = Path('../../filtered_ds/')
new_df = pd.read_csv(data_path/'images_df')

data = ((ImageList.from_df(df=new_df, 
                           path=data_path)
                 .split_from_df()
                 .label_from_df(cols='label')
                 .transform(get_transforms(max_rotate=3.,max_warp=0.,p_affine=0.), size=160)
                 .databunch())
                 .normalize(imagenet_stats))
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(1)
learn.export()