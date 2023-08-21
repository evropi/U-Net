#This repository contains code for the brain segmentation of micro-CT images of ants described in the manuscript https://www.biorxiv.org/content/10.1101/2021.05.29.446283v1. Please, contact the authors for the data used to run the notebooks and for more specific instructions.

The training procedure might take long if performed on a CPU. We advise using a GPU powered workstation. 

More analytically, the following instructions can be of use for anyone who wants to apply the code on ant (or other insect) micro-CT scans.

The unet file in the repository contains 4 scripts that should be saved in the environment where the training will be performed. The training set should be in the form of .tiff images and the masks should have the same name as the raw images followed by _mask. 

#data loading
data_provider = image_util.ImageDataProvider("…/*.tif")

After loading the data, the user can split them into training and validation sets. As a next step, the network will be loaded.

#choose hyperparameters 
net = unet.Unet(channels = data_provider.channels, n_class = data_provider.n_class, layers = , features_root = )

The user should choose the batch size, number of epochs, optimizer, and dropout probability. Default hyperparameters will be used if not assigned by the user, i.e., batch size =4, epochs = 10 optimizer = momentum and dropout = 0.

#setup & training
trainer = unet.Trainer(net, batch_size = , verification_batch_size = , optimizer = "momentum", opt_kwargs = dict(momentum = 0.8))
path = trainer.train(data_provider, "./unet_trained_trial_one", training_iters = , epochs = , prediction_path="./prediction_trial_one")

The training process can be tracked using Tensorboard. It is highly recommended to use Jupyter notebook. 

After training, the tuned network can be re-called and used by “predict (model, x_test)”. 

#verification
prediction = net.predict("./unet_trained/model.ckpt", my_prediction_np_re)

#predictions
fig, ax = plt.subplots(1, 3, sharex = True, sharey = True,figsize=(12,5))
ax0 = fig.add_subplot(131)
ax0.imshow(my_prediction_np, aspect="auto",cmap = plt.cm.Greys_r)
ax1 = fig.add_subplot(132)
ax1.imshow(mymask, aspect = "auto",cmap = plt.cm.Greys_r)
mask = prediction[0,...,1] > threshold 
ax2 = fig.add_subplot(133)
ax2.imshow(mask, aspect="auto",cmap = plt.cm.Greys_r)

ax0.set_title("Input")
ax1.set_title("Ground truth")
ax2.set_title("Prediction")
fig.tight_layout()
fig.savefig("./results.png")

An exemplar training process can be found in “brain_segmentation_example.ipynb”. 

![image](https://github.com/evropi/U-Net/assets/25073395/ef9f0a2e-8b51-439d-b0fc-90290e677002)
