* New architecture: Dinov2 ViT-S/14 as an image encoder.


| Filename | Description |
|---|---|
| stanford_hydra-17419904.out | Run using DinoV2 ViT-S/14. 30 epochs shows a decrease in loss (training and validation) with increase in accuracy (train and val). |
| stanford_hydra-17422554.out | Training is improving, generalization is not. Training improves because the head now gest more weight updates and there is a higher learning rate (1e-3 with cosine decay instead of 1e-4). The global batch went from 2048 to 256. More weight updates are done because the global batch size is reduced, therefore, there are more batchs, then, more steps as the optimizer takes one step per batch. |
| stanford_hydra-17489495.out | Run same model as before but with 150 episodes instead of 40. No increase in accuracy is observed in validation data.|
| stanford_hydra-17492315.out | Grid pooling is implemented |
| stanford_hydra-17515265.out | Implement fine tuning of the last two blocks of the DinoV2 ViT-S/14 model. No improvement in performance in validation data. |
| stanford_hydra-17550111.out | Give the same head the features of frames t and t+1 from the same episode and predict the action at t. The model sees what the action did. Should performance not be improved: the labels and pipeline are the problem. I should Check that frames and actions line up within each episode. Look at what lands in the middle bins, and remove no-op actions. If single-step deltas are smaller than the camera can show, predict the sum of the next k deltas instead. with next_frame=True, each sample's image is the slice of frames t and t+1. No fine tuning is enabled, train_blocks = 0. Taking out the fine-tuning of the last two blocks sped-up the time per epoch.| 
| stanford_hydra-17557209.out | Implement in train.py the cosine similarity between consecutive frames to see whether features change across steps. Your labels are fine, but one step of motion barely changes the frozen features: about 3% of the distance between unrelated frames (1 − 0.991 vs. 1 − 0.66), against about 13% over 10 steps, so single-step actions are nearly invisible to the model.| 
| ... | Adapt preprocessing, models and train scripts to incorporate this change: Rerun the probe on frames t and t+10, with the feature difference f(t+10) − f(t) as input, predicting the summed 10-step action.  |
| ... | ... |
         


