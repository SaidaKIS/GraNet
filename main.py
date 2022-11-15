from cProfile import label
import dataset
import losses
import model_GraNet
import train
import utils
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import pickle
import pandas as pd
from glob import glob
from torchsummary import summary

#change this in general 
#/Users/smdiazcas/miniconda/envs/pyUnet/lib/python3.9/site-packages/torch/storage.py
#class Unpickler(pickle.Unpickler):
#    def find_class(self, module, name):
#        if module == 'torch.storage' and name == '_load_from_bytes':
#            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#        else: return super().find_class(module, name)

def flatten(t):
    return [item for sublist in t for item in sublist]

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    bin_classes = ['Intergranular lane', 'Uniform-shaped granules', 'Granules with a dot', 'Granules with a lane',
                   'Complex-shaped granules']

    #Parameters
    root = 'data/Masks_C/' # Raw full IMaX maps (6 for training and 1 for validate)
    l = 30000 # Submaps dataset size 
    size_box = 128 # size of each submap
    channels = 1
    N_EPOCHS = 200 
    BACH_SIZE = 32  
    loss = 'FocalLoss' # 'CrossEntropy', 'FocalLoss', 'mIoU'
    save_model = False
    bilinear = False # Unet upsampling mechanisim is Traspose convolution
    model_summary = True
    lr = 1e-3
    dropout = False

    data=dataset.segDataset(root,l=2000, s=size_box)
    for i in range(2):
        img, mask, ind, c = data[i]



    #prop=pd.DataFrame(columns=[0, 1, 2, 3, 4], index=np.arange(0,2000))
    #data=dataset.segDataset(root,l=2000, s=size_box)
    #centre = []
    #for i in range(2000):
    #    img, mask, ind, c = data[i]
    #    values, counts = np.unique(mask, return_counts=True)
    #    prop.loc[i, values] = np.array(counts/sum(counts))
    #    if ind == 0:
    #        centre.append(c)
    #    if i % 200 == 0:
    #        print(i)
    #        ax1 = plt.subplot(121)
    #        ax1.imshow(img[0])
    #        ax1.set_xticks([])
    #        ax1.set_yticks([])
    #        ax2 = plt.subplot(122)
    #        ax2.imshow(mask)
    #        ax2.set_xticks([])
    #        ax2.set_yticks([])
    #        plt.tight_layout()
    #        plt.show()
    #print(prop.mean())
    #
    file_list = sorted(glob(root+'*.npz'))
    file = np.load(file_list[0])
    #mask = file['cmask_map'].astype(np.float32)
    #c = np.array(centre)
    #utils.test_centers(mask, c[:,0], c[:,1])

    ##Train a model
    #train.run(root, l, size_box, channels, N_EPOCHS, BACH_SIZE, loss, lr = lr, scale=1,
    #    save_model=True, bilinear=False, model_summary=False, dropout=dropout)

    #Test model
    #Initial summary
    #model_unet = model.UNet(n_channels=1, n_classes=5, scale=8, bilinear=bilinear, dropout=dropout).to(device)
    #summary(model_unet, (channels, 128, 128))

    # Generate a prediction 
    #model_test1 = torch.load('../New_results/NewGT_Jan2022/Augmentation/unet_epoch_12_0.52334_IoU_non_Dropout.pt', map_location=torch.device(device))
    #file = 'data/Masks_S_v3/Train/Mask_data_Frame_0.npz'
    #model_test1 = torch.load('../New_results/NewGT_Jan2022/Augmentation/unet_epoch_199_13.23084_FL_nonDp_g10.pt', map_location=torch.device(device))   
#
    ####smap_f0, cmask_map_f0, total, total0, ls=utils.model_eval(file, model_test1, device, size_box)
    #smap_f0, cmask_map_f0, total, total0, ls=utils.model_eval_full(file, model_test1, device, size=761)
    #print(ls)
###
    #utils.probability_maps(smap_f0[0], total[0], bin_classes)
    #utils.comparative_maps_wz(smap_f0[0], cmask_map_f0[0], total0[0], bin_classes, save=True) 

    #tmap = cmask_map_f0[0]
    #pmap = total0[0]
    #xi, jy = cmask_map_f0[0].shape
    #conf_mx = np.zeros((5,5))
    #total_clss = np.zeros(5)
#
    #for i in range(xi):
    #    for j in range(jy):
    #        pixt = int(tmap[j,i])
    #        total_clss[pixt] = total_clss[pixt] + 1 
    #        pixp = int(pmap[j,i])
    #        conf_mx[pixp,pixt] = conf_mx[pixp,pixt] + 1 

    #plt.rcParams.update({
    #"font.family": "sans-serif",
    #"font.sans-serif": ["Helvetica"]})
    #plt.rcParams.update({'font.size': 20})
    #plt.rcParams['figure.figsize'] = 15,9
#
    #import seaborn as sns
    #ax = sns.heatmap(np.divide(conf_mx, total_clss), annot=True, fmt='.2', cmap='Blues')
#
    #bin_classes_cm = ['Intergranular\nlane', 'Uniform-shaped\ngranules', 'Granules\nwith a dot', 'Granules\nwith a lane','Complex-shaped\ngranules']
    #
    #ax.set_xlabel('Predicted Classes', fontsize=25)
    #ax.set_ylabel('Ground-Truth Classes', fontsize=25)
#
    #for i in ax.get_yticklabels():
    #    i.set_ha('right')
    #    i.set_rotation(0)
#
    #for i in ax.get_xticklabels():
    #    i.set_ha('center')
    #
    ### Ticket labels - List must be in alphabetical order
    #ax.xaxis.set_ticklabels(bin_classes_cm)
    #ax.yaxis.set_ticklabels(bin_classes_cm)
    #
    ### Display the visualization of the Confusion Matrix.
    #plt.tight_layout()
    #plt.show()


    #imax_save = '/Users/smdiazcas/Documents/Phd/Research/NN_granulation/contmaps.sav'
    #utils.test_Imax(imax_save, model_test1, bin_classes)
   
    #Training information
    #with open ('../New_results/NewGT_Jan2022/Augmentation/Train_params_2022_02_15_11_57_49_FL_fullModel_nonDp_highaug_lr6e-3.npy', 'rb') as f:
    ##with open ('../New_results/NewGT_Jan2022/Augmentation/Train_params_2022-02-05_03_00_00_IoU_non_Dropout.npy', 'rb') as f:    
    #    training_info = np.load(f, allow_pickle=True)
    #    metrics = np.load(f, allow_pickle=True)
    #    h_train_metrics = np.load(f, allow_pickle=True)
    #    h_val_metrics = np.load(f, allow_pickle=True)
####
    #print(training_info)
    #utils.metrics_plots(metrics, Title='Test 5: Focal Loss $\gamma = 10$ lr - 0.006')
    #utils.metrics_plots(metrics, Title='Test 2: Mean Intersection-over-Union (mIoU)')
##
    #h_lt=[]
    #h_lv=[]
    #h_at=[]
    #h_av=[]
    #for i in range(5):
    #    h_lt.append(h_train_metrics[i,0,:])
    #    h_at.append(h_train_metrics[i,1,:])
    #    h_lv.append(h_val_metrics[i,0,:])
    #    h_av.append(h_val_metrics[i,1,:])
#
    #fig, ax =plt.subplots(nrows=2,ncols=2)
    #ax[0][0].hist(h_lt, bins=10)
    #ax[0][1].hist(h_at, bins=10)
    #ax[1][0].hist(h_lv, bins=10)
    #ax[1][1].hist(h_av, bins=10)
    #ax[0][0].set_title('Loss Training')
    ##ax[0][0].legend(prop={'size': 13})
    #ax[0][1].set_title('Acc Training')
    #ax[1][0].set_title('Loss Validation')
    #ax[1][1].set_title('Acc Validation')
    #ax[1][0].set_xlabel('Values')
    #ax[1][1].set_xlabel('Values')
    #ax[0][0].set_ylabel('Counts/Dataset elements')
    #ax[1][0].set_ylabel('Counts/Dataset elements')
#
    #plt.show()
##