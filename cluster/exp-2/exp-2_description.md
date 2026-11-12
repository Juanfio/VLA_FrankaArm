Experiment	DDP processes	  cpus-per-task      Threads/process   Batch/process	    Global batch            Filename
    A           1	               48                   48              128	                128            stanford_hydra-17217122.out
    B	        2       	       48                   24	            64	                128            stanford_hydra-17226327.out
    C	        4	               48                   12	            32	                128            stanford_hydra-17227995.out
    D	        8	               48                    6	            16	                128            stanford_hydra-17237257.out
    E           4                  48                   12              64                  256            stanford_hydra-17239164.out
    F           8                  48                    6              64                  512            stanford_hydra-17240149.out
    G           12                 48                    4              64                  768            stanford_hydra-17257311.out
    H           32                128                    4              64                 2048            stanford_hydra-17258026.out


    Conclusion:
        * Increasing DDP processes while decreasing batch size (i.e., remaining global batch constant. Exp A-D) decreases training time.
        * Preprocessing remained almost constant.
        * The global batch represents the number of sample contributions used to calculate the gradients for one synchronized optimizer update. It is computed as the product between the individual batch size (the one entering _make_dataloader()) and the total number of processes.
        * Exp E-F:
            * The key question is whether larger batches allow your CPU resources to process more samples per second without degrading model convergence. That is the next optimization worth investigating.
