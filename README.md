# ICLRec
Pytorch implementation for paper: Intent Contrastive Learning for Sequential Recommendation.

The recommendation loss is changed from BPR to cross-entropy loss for better performance.

You can also refer to the offical implementation: [https://github.com/salesforce/ICLRec](https://github.com/salesforce/ICLRec).
## Datasets
Toys dataset is contained in this repository.
## Quick Start
You can run the model with the following code:
```
python runICLRec.py --dataset toys --embed_size 128 --num_intent_cluster 256 
```


