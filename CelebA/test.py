import pandas as pd

d2 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_nonspurious/model_outputs')
d3 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_nonspurious/model_outputs')

print(d1.shape)
print(d2.shape)
print(d3.shape)

print(d3.columns)
# avg_acc_group:1
for i in range(d3.shape[0]):
    if d3.loc[i, 'Male'] != d3.loc[i, 'y_true_None_epoch_1_val']:
        print(d3.loc[i, 'Male'], d3.loc[i, 'y_true_None_epoch_1_val'])
        print(d3.loc[i, 'indices_None_epoch_1_val'])
        print(1/0)
