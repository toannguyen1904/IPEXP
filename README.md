## Note

Change the following points so that this source code could execute correctly:
- ``train_img_path``, ``val_img_path``, ``test_img_path`` in **dataset/pre_data.py** <br />
- ``root`` and ``save_path`` in **config.py**

## Training, validating and testing
To run the model, firstly, please download the ``images`` folder of the dataset to **dataset/ip102_v1.1**. Secondly, run ``python train.py -d`` folowed by the name of the sub-dataset you want to train: ``meta``, ``non_meta``, ``all``, ``two_classes``.
