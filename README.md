# deepfake-detection

## Project structure

The project is structured as following:

```bash
.
├── datasets
|   └── dataset selector
|   └── video_dataset.py # loading and pre-processing gtsrb data
├── models
|   └── architecture selector
|   └── resnet_rnn.py # highly compressed CNN
├── nbs
|   └── 00_video_dataset.ipynb # notebook used to create the dataset script
|   └── 01_model.ipynb # notebook used to create the resnet_rnn script
|   └── 02_train.ipynb 
|   └── Colab_ds_preproc.ipynb # preprocessing of the original dataset in Google Colab
|   └── notebook2script.py
|   └── train_fastai.py # fastai model was trained using this script
├── args.py # parsing all command line arguments for experiments
├── trainer.py # main file from the project serving for calling all necessary functions for training and testing
```

## Launching
Experiments can be launched by calling `train.py` and a set of input arguments to customize the experiments. You can find the list of available arguments in `args.py` and some default values. Note that not all parameters are mandatory for launching and most of them will be assigned their default value if the user does not modify them.

Here is a typical launch command and some comments:

- `python train.py --batch_size 16 --crop_size 224 --csv_file videos_df --dataset video --dropout 0.3 --embed_dim 512 --epochs 50  --log_interval 10  --lr 0.001 --model_name resnet152 --num_classes 2 --pretrained True --workers 4`
  
## Output
For each experiment weights of all the networks are saved to the folder `checkpoints` after each epoch as well as all the train and valiation losses and accuracy values.
