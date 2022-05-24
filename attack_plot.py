import argparse
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# 
parser.add_argument('--distance', type=str, default='L2')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--num_batch', type=int, default=5)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--attack_file', type=str, default="./adversarials_batch_0_cfar10_L2.npy")

args = parser.parse_args()


def find_epsilons(adversaries_rl_name, args_distance, args_batch_size, num_batch=1):

    adv_size = args_batch_size
    eps_mean = np.zeros(adv_size + 1)
    eps_max = np.zeros(adv_size + 1)

    acc_mean = np.zeros(adv_size + 1)

    for j in range(num_batch):
        adversaries_rl = np.load(adversaries_rl_name.replace('_batch_0', '_batch_' + str(j)), allow_pickle=True)

        actual_size = adversaries_rl.shape[0]
        
        eps = np.zeros(adv_size + 1)
        acc = np.zeros(adv_size + 1)
        eps[0] = 0.
        acc[0] = 1.

        for i in range(adv_size):
            initial_offs = adv_size - actual_size
            #print(initial_offs)
            if i > initial_offs:
                # get difference between original image and advesarial version
                perturbation = adversaries_rl[i - initial_offs,2] - adversaries_rl[i - initial_offs,0]

                # find norm of perturbation
                if args_distance == 'L2':
                    eps[i+1] = min(np.linalg.norm(perturbation.flatten()*255, 2), 255.)
                else:
                    eps[i+1] = np.linalg.norm(perturbation.flatten()*255, np.inf)
            else:
                eps[i+1] = 0

            # calculate list of accuracies
            acc[i+1] = 1. - (i+1.)/adv_size

        eps = np.sort(eps)
        
        eps_mean += eps

        acc_mean = acc
            
    return eps_mean/num_batch, acc_mean
    
# plot results
eps_ed, acc_ed = find_epsilons(args.attack_file, args.distance, args.batch_size, args.num_batch)

plt.plot(eps_ed, acc_ed * 100, label="NG-EBM")

plt.legend(loc="upper right")

if args.distance == 'L2':
    plt.xlim([0,250])
else:
    plt.xlim([0,20])

plt.xlabel("epsilon")
plt.ylabel("accuracy")
plt.title("Adversarial attack " + args.dataset + ", norm " + args.distance + ".")

plt.savefig("adversarials_" + args.dataset + "_" + args.distance + ".png")


