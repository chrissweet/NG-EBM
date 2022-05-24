# NG-EBM - Non-generative Energy Based Models
This repository is forked from JEM - Joint Energy Models (Grathwohl et. al. 2019) and contains modified code to train and evaluate our Non-generative Energy Based Models method.

We would like to thank the JEM authors for not only an excellent paper but for providing their code which simplified the implementation of NG-EBM and allowed us to contrast the methods.

A pretrained NG-EBM model on CIFAR10 can be found [here](http://www.crc.nd.edu/~csweet1/NG-EBM_CIFAR10_MODEL.pt).

## Usage
### Training
To train a model on CIFAR10 or CIFAR100, modified for the NG-EBM paper (choice for ```--dataset``` is ```cifar10``` or ```cifar100```):
```markdown
python3 train_wrn_ebm.py --lr .0001 --dataset cifar10 --optimizer adam --energy_variance_loss 1.0 --energy_derivative_loss 1.0 --p_x_weight 0.0 --p_y_given_x_weight 1.0 --p_x_y_weight 0.0 --sigma .03 --width 10 --depth 28 --save_dir /YOUR/SAVE/DIR --warmup_iters 1000
```

Seed can be chosen with ```--seed [int]```. Seeds chosen for the paper were 0101, 0202, 0303, 0404, and 0505.

### Evaluation

To evaluate the classifier on cifar10 (for cifar100 use ```--dataset cifar100_test --n_classes 100```):
```markdown
python3 eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval test_clf --dataset cifar_test
```
To do OOD detection (on cifar100, with ```pxgrad``` [Approximate Mass] scoring function). Score functions available ```px```, ```py```, ```pxgrad```. Example for cifar10, for cifar100 use ```--ood_dataset svhn --n_classes 100```:
```markdown
python3 eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval OOD --score_fn pxgrad --ood_dataset cifar_100
```
To generate a histogram of OOD scores like Table 4 in paper. Example for cifar10, for cifar100 use ```--datasets cifar100 svhn --n_classes 100```:
```markdown
python3 eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval logp_hist --datasets cifar10 svhn --save_dir /YOUR/HIST/FOLDER
```
To calculate energies for a data set and save as ```energies.csv``` (for cifar10, for cifar100 use ```--dataset cifar100_test --n_classes 100```):
```markdown
python3 eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval pri_energy --dataset cifar_test --save_dir /YOUR/SAVE/DIR 
```
To calculate the binned calibration and save as ```calibration.csv``` (for cifar10, for cifar100 use ```--dataset cifar100_test --n_classes 100```):
```markdown
python3 eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval calibration --dataset cifar_test --save_dir /YOUR/SAVE/DIR
```

### Attacks

Note that this requires foolbox 1.8.

To run Linf attacks on NG-EBM. Example for cifar10, for cifar100 use ```--dataset cifar100 --n_classes 100```:
```markdown
python3 attack_model.py --start_batch 0 --end_batch 6 --load_path /PATH/TO/YOUR/MODEL.pt --exp_name /YOUR/EXP/NAME --n_steps_refine 1 --distance Linf --random_init --n_dup_chains 5 --base_dir /PATH/TO/YOUR/EXPERIMENTS/DIRECTORY --attack_tries 1
```
To run L2 attacks on NG-EBM. Example for cifar10, for cifar100 use ```--dataset cifar100 --n_classes 100```:
```markdown
python3 attack_model.py --start_batch 0 --end_batch 6 --load_path /cloud_storage/BEST_EBM.pt --exp_name rerun_ebm_1_step_5_dup_l2_no_sigma_REDO --n_steps_refine 1 --distance L2 --random_init --n_dup_chains 5 --sigma 0.0 --base_dir /cloud_storage/adv_results --attack_tries 1
 ```
To plot the attack file
```markdown
python3 attack_plot.py --attack_file ./experiment_attack/adversarials_batch_0_cfar10_L2.npy --distance L2 --dataset cifar10 --batch_size 50 --num_batch 5
 ```

## Other dependancies
python 3.8.0

pytorch 1.9.0

numpy 1.21.2

pillow 8.2.0
