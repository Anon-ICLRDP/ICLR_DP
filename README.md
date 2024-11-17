# Instructions:

# Datasets

For tasks that require the CelebA dataset, download and unzip the dataset from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html in the directory in which the experiment is to be run (FileName: img_align_celeba.zip). Additionally, attain list_eval_partition.csv and metadata.csv.

For tasks that require the Waterbirds dataset, download and unpack the tarball dataset from https://github.com/kohpangwei/group_DRO#waterbirds in the directory in which the experiment is to be run.

For tasks that require the MultiNLI dataset, please see - https://github.com/kohpangwei/group_DRO.

For tasks that require the HardImageNet dataset, download and unzip the dataset from https://mmoayeri.github.io/HardImageNet/. Move the folders hardImageNet and HardImageNet_Images to the relevant directory.

# Requirements:

torch==2.3.1
torchvision==0.18.1
numpy==1.22.0
pandas=1.3.5

# Data Pruning - CelebA (Built on top of code provided by Liu et. al. - https://github.com/anniesch/jtt/tree/master)

Unzip CelebA zip file and move img_align_celeba to celebA/data. Move metadata.csv and list_eval_partition.csv to celebA/data as other3.csv and other2.csv respectively.

1) Generate required metadata and list_eval_partition.

```
python3 remove_required.py
```

2) Train with the full dataset. Extract error norms as file '9el2n.pkl' (Note: you can stop running the script before the 11th epoch.)
```
python3 generate_downstream.py --exp_name CelebA_sample_exp --dataset CelebA --n_epochs 25 --lr 1e-3 --weight_decay 1e-4 --method ERM
bash results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_25_lr_1e-03_weight_decay_0.0001/job.sh
```

3) Comment out lines 137-158 in train.py

4) Rename metadata.csv and list_eval_partition in celebA/data as metadata_main.csv and list_eval_partition_main.csv

5) Prune Data

For pruning only samples with spurious features (Figure 4)

```
python3 test_pickle_oracle.py # For Hardest
```

For pruning all samples (Figure 6)

```
python3 test_pickle.py # For Hardest
```

In celebA/data, run:

```
python3 get_el2n.py
```

Depending on desired data sparsity, respective files as shown for 50% sparsity:

```
mv metadata_50prune.csv metadata.csv
mv list_eval_partition_50prune.csv list_eval_partition.csv
```

6) Train on pruned dataset

```
python3 generate_downstream.py --exp_name CelebA_sample_exp --dataset CelebA --n_epochs 25 --lr 1e-3 --weight_decay 1e-4 --method ERM
bash results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_25_lr_1e-03_weight_decay_0.0001/job.sh
```

7) Evaluate (Make sure to set the right paths in the result folder)

```
python3 get_all_testing_valtuned.py
```


# Data Pruning - Waterbirds (Built on top of code provided by Kirichenko et. al. - https://github.com/PolinaKirichenko/deep_feature_reweighting)

Unpack the Waterbirds tarball dataset

1) Train classifier: (Note: you can stop running the script before the 2nd epoch.)

```
python3 train_classifier.py  --output_dir=outputs --pretrained_model --num_epochs=100 --weight_decay=1e-3 --batch_size=32 --init_lr=1e-3 --eval_freq=1 --data_dir=waterbird_complete95_forest2water2 --test_wb_dir=waterbird_complete95_forest2water2 --augment_data --seed=1
```

2) Comment out lines 258 - 277

3) For Data Pruning:

```
python3 testing_trainloader_el2n.py
python3 change_metadata.py
```

Point metadata directory in wb_data.py to newly created metadata (new_metadata_95.csv).

Eg: metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv")) # Change metadata.csv to point to newly generated metadata.


4) Train on Pruned dataset

```
python3 train_classifier.py  --output_dir=outputs --pretrained_model --num_epochs=100 --weight_decay=1e-3 --batch_size=32 --init_lr=1e-3 --eval_freq=1 --data_dir=waterbird_complete95_forest2water2 --test_wb_dir=waterbird_complete95_forest2water2 --augment_data --seed=1
```

5_ Evaluate

```
python3 test.py
```


# Data Pruning - MultiNLI (Built on top of code provided by Sagawa et. al. - https://github.com/kohpangwei/group_DRO)

For information regarding data formatting, please follow - https://github.com/kohpangwei/group_DRO

1) Train classifier:  (Note: you can stop running the script before the 6th epoch.)

```
python3 run_expt.py -s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr 2e-05 --batch_size 32 --n_epochs 20 --save_step 1000 --save_best --save_last
```

2) Move to multinli/data directory. Rename metadata_random.csv and metadata_random_original.csv.

```
python3 change_metadata.py
```
Move back to root dir.

In train.py, comment out lines 90-121.

In data/multinli_dataset.py, uncomment lines 12, 13, 100.


3) Train on Pruned dataset

```
python3 run_expt.py -s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr 2e-05 --batch_size 32 --n_epochs 20 --save_step 1000 --save_best --save_last
```

5_ Evaluate

```
python3 test_all.py
```


# Data Pruning - HardImageNet 

1) Create dataset and train classifier (Note: you can stop running the script before the 2nd epoch.)

```
python3 create_sets_spind.py
python3 ResNet20_maine.py
```

2) Prune the dataset: (To alter the percentage pruned, change fraction in line 116 of create_sets_spind2.py)

```
python3 create_sets_spind2.py # For Hardest
```

3) Train on pruned dataset:

```
python3 ResNet20_main_retrain.py
```

4) Evaluate:

```
python3 test.py
```
