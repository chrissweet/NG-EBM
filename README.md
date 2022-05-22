# NG-EBM - Non-generative Energy Based Models
This repository is forked from JEM - Joint Energy Models (Grathwohl et. al. 2019) and contains modified code to train and evaluate our Non-generative Energy Based Models method.

We would like to thank the JEM authors for not only an excellent paper but for providing their code which simplified the implementation of NG-EBM and allowed us to contrast the methods.

A pretrained NG-EBM model on CIFAR10 can be found [here](http://www.crc.nd.edu/~csweet1/NG-EBM_CIFAR10_MODEL.pt).

## Usage
### Training
To train a model on CIFAR10 modified for the NG-EBM paper
```markdown
python train_wrn_ebm.py --lr .0001 --dataset cifar10 --optimizer adam --energy_variance_loss 1.0 --energy_derivative_loss 1.0 --p_x_weight 0.0 --p_y_given_x_weight 1.0 --p_x_y_weight 0.0 --sigma .03 --width 10 --depth 28 --save_dir /YOUR/SAVE/DIR --plot_uncond --warmup_iters 1000
```

### Evaluation

To evaluate the classifier (on CIFAR10):
```markdown
python eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval test_clf --dataset cifar_test
```
To do OOD detection (on CIFAR100)
```markdown
python eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval OOD --ood_dataset cifar_100
```
To generate a histogram of OOD scores like Table 2
```markdown
python eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval logp_hist --datasets cifar10 svhn --save_dir /YOUR/HIST/FOLDER
```
To calculate energies for a data set and save as ```energies.csv```
```markdown
python eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval pri_energy --dataset cifar_test --save_dir /YOUR/SAVE/DIR 
```
To calculate the binned calibration and save as ```calibration.csv```
```markdown
python eval_wrn_ebm.py --load_path /PATH/TO/YOUR/MODEL.pt --eval calibration --dataset cifar_test --save_dir /YOUR/SAVE/DIR
```

### Attacks

To run Linf attacks on JEM-1
```markdown
python attack_model.py --start_batch 0 --end_batch 6 --load_path /PATH/TO/YOUR/MODEL.pt --exp_name /YOUR/EXP/NAME --n_steps_refine 1 --distance Linf --random_init --n_dup_chains 5 --base_dir /PATH/TO/YOUR/EXPERIMENTS/DIRECTORY
```
To run L2 attacks on JEM-1
```markdown
python attack_model.py --start_batch 0 --end_batch 6 --load_path /cloud_storage/BEST_EBM.pt --exp_name rerun_ebm_1_step_5_dup_l2_no_sigma_REDO --n_steps_refine 1 --distance L2 --random_init --n_dup_chains 5 --sigma 0.0 --base_dir /cloud_storage/adv_results &
 ```
 
