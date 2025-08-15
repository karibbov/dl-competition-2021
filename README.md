# Old Lost and Found Projects

This is a deep learning competition project from Fall 2021.
It includes:
- Code for supervised and contrastive-loss based self-supervised CNN training
- Multiple architectures: ResNet, U-Net, and custom CNNs
- A variety of data augmentation techniques
- Hyperparameter optimization for:
    - Optimizer choice
    - Learning rate schedule
    - CNN architecture
    - Data augmentation
    - Loss function

The model was **only** trained on the data present in this repository and performed on par with transfer learning from pre-trained resnet models on the held-out dataset. 
The rest of this repository has not been modified in the last 3 years. 

# README
Samir Garibov.

The main models used during training are: 

- `SkipModel` (main model architecture trained - A similar model to resnet with skip connections)
- `USkipModel` (for self-supervised learning attempt), 
- `TransferSkipModel` (for adapting `USkipModel` to `SkipModel`)

The main training is done in the `pipeline.py` file which has three different steps:

- Self Supervised pretraining
- Hyper-parameter search with BOHB and hpster /or/ custom HPO 
- Final Training with all the data for the best hyperparameters

Unfortunately, I was unable to run BOHB to termination with 22 iterations, on tfpool computers since they restart every night at 03:10. This knowledge costed me 2 nights of interrupted runs, therefore I didn't have enough time to try BOHB with less iterations and still be effective. However, I extracted (manually copied from terminal:\\ ) some of the best parameters from my last BOHB run, and included sensible modifications of some in my custom HPO, which is just running 3 configurations with the highest budget. (In a way BOHB was used for insight for best schedulers). 

Since `src.pipeline` was interrupted after pretraining was completed, in my last run I skipped the pretraining and just used the defaults, in case you want to train it from scratch, You'll need to set `--ssl_run` parameter to `SSL` 

The reason `main.py` is so complicated is because I used it for everything during the whole process and wrote both pipeline and hpster workers as a wrapper around it. Hope this explanation would be helpful to understand what is going on. Again I'm sorry that I don't have enough time to refactor the code. 

