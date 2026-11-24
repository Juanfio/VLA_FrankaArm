* New architecture: Dinov2 ViT-S/14 as an image encoder.


| Filename | Description |
|---|---|
| stanford_hydra-17419904.out | Run using DinoV2 ViT-S/14. 30 epochs shows a decrease in loss (training and validation) with increase in accuracy (train and val). |
| stanford_hydra-17422554.out | Training is improving, generalization is not. Training improves because the head now gest more weight updates and there is a higher learning rate (1e-3 with cosine decay instead of 1e-4). The global batch went from 2048 to 256. More weight updates are done because the global batch size is reduced, therefore, there are more batchs, then, more steps as the optimizer takes one step per batch. |
| stanford_hydra-17489495.out | Run same model as before but with 150 episodes instead of 40. No increase in accuracy is observed in validation data.|
| stanford_hydra-17492315.out | Grid pooling is implemented |



| ... | ... |
         


