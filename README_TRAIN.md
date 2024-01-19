# Data Preparation

## 1. Using InsightFace Dataset

InsightFace provides a variety of labeled face dataset preprocessed to 112x112 size.

[insightface link](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)

The unzipped dataset looks as follows

```
example: faces_vgg_112x112
└── train.rec   
├── train.idx                                                                                       │
├── train.lst                                                                                       │
├── agedb_30.bin                                                                                    │
├── calfw.bin                                                                                       │
├── cfp_ff.bin                                                                                      │
├── cfp_fp.bin                                                                                      │
├── cplfw.bin                                                                                       │
├── lfw.bin                                                                                         │
```

`train.rec` contains all the training dataset images and `rec` format combines all data to a single file
whilst allowing indexed access.
`rec` file is good when one does not one to create millions of individual image files in storage.

We provide a training code that utilizes this `rec` file directly without converting to `jpg` format.
But if one ones to convert to `jpg` images and train, refer to the next section.

#### Dataset preparation steps

1. Download the dataset from [insightface link](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
2. Unzip it to a desired location, `DATASET_ROOT`  _ex)_ `/data/`.
3. The result folder we will call `DATASET_NAME`, ex) `faces_vgg_112x112`.
4. For preprocessing run
   1. `python convert.py --rec_path <DATASET_ROOT>/<DATASET_NAME> --make_validation_memfiles`

* Note you cannot turn on `--train_data_subset` option. For this you must expand the dataset to images
  (refer to below section).

## 2. Using Image Folder Dataset

Another option is to extract out all images from the InsightFace train.rec file.
It uses the directory as label structure, and you can swap it with your own dataset.

#### Dataset preparation steps for InsightFace dataset

1. Download the dataset from [insightface link](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
2. Unzip it to a desired location, `DATASET_ROOT`  _ex)_ `/data/`.
3. The result folder we will call `DATASET_NAME`, ex) `faces_vgg_112x112`.
4. For preprocessing run
   1. `python convert.py --rec_path <DATASET_ROOT>/<DATASET_NAME> --make_image_files --make_validation_memfiles`

# Train

Just run

```
python main.py
```

Options are available in configs, you may want to check https://hydra.cc/docs/intro/ for more information.

- For training with .rec file or Image folder, change the `use_mxrecord` in config/data/XXX.yaml
- For validation on SCface and tinyface, only .rec file is acceptable
- For your local or remote machine setting, just change the config/platform
- The config/logger/csv is deprecated
