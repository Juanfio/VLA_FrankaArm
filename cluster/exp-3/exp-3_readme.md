* Improve model performance.

| Filename | Description |
|---|---|
|stanford_hydra-17296957.out |  Run with class_weights in cross entropy loss. Weights taken as sqrt(frequencies.mean() frequencies). Weights are clamped at 5.0.|
|stanford_hydra-17298061.out |  Run with class_weights in cross entropy loss. Weights taken as (frequencies.mean() frequencies).pow(0.25). Weights are clamped at 2.0.|
|stanford_hydra-17404025.out |  Goal: Test whether your model can overfit a small dataset. Label smoothing = 0, Weight decay = 0, Dropout = 0, Class weight = None.|
