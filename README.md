# COVID_transformer
a vision transofmer for classifying xray lung images, to be classified into "covid", "normal", "non-covid"

##Notes on training
-small learning rate with warmup cosin anealing is a key to get reasonable accuracy
-using `Adam with decoupled weight decay` (shortly AdamW) to stabilize training.
