
# DMFL - Deep Multimodal Federated Learning for Enhanced Multimodal Sentiment Analysis Using Images and Text

This repository includes the original implementation of "**Deep Multimodal Federated Learning for Enhanced Multimodal Sentiment Analysis Using Images and Text**"  in TensorFlow 2.x with TFX as the backend. 

## Acknowledgement:
- This work has been conducted under the supervision of `Prof JX Wang` and School of Computer Science and Engineering, `Central South University`, China.


## Proposed Architecture

-  The proposed Stacked-GAP (SG) network, which extracts characteristics from each network unit. The GAP layers (G) reduce the overall network parameters, thereby making the network more resistant to overfitting.

![MG network architecture](https://raw.githubusercontent.com/zohairahmed007/DMFL/main/architecture/Stackedgap.jpg)



-  The MobileNet has been trained on ImageNet to represent emotion information in networks. 

![MG network architecture](https://raw.githubusercontent.com/zohairahmed007/DMFL/main/architecture/MobileNet.jpg)



-  Transfer learning using Universal Sentence Encoder for Sentiment Classfication 

![MG network architecture](https://raw.githubusercontent.com/zohairahmed007/DMFL/main/architecture/use.png)

## Description


- The `models/textdata_builder.py` comprises the implementation of text and image pair separation and the construction of frames for subsequent network input.


- The `models/visual_net_SG.py` contains the implementaion of visual network.The goal is to extract several levels of features from various divisions, including low-level features (such as tone and boundary), middle-level features (such as texturing and appearance), and high-level features (such as object size and dimension).

- `models/multihead_attention_transformer.py` Multi-head Attention is a module for attention mechanisms that runs multiple times in parallel through an attention mechanism. Multiple attention heads make it possible to pay different amounts of attention to distinct parts of a sequence (e.g., longer-term dependencies versus shorter-term dependencies).

- `models/finetune_text_classification.py` Fine-tuning sentiment classification adapting to freezing the parameters to USE training cycles from losing any information they contain On top of the fixed layers, add some new trainable layers and again training the new layers with an our dataset.

- `models/multimodel_sentiment.py` Combine two networks of mortality and fuse for a conclusive predection. 

- `models/gradcam_regions.py` The Grad-CAM technique utilizes the gradients of the classification score with respect to the final convolutional feature map, to identify the parts of an input image that most impact the classification score. 

## Visual Network Results
![MG network architecture](https://raw.githubusercontent.com/zohairahmed007/DMFL/main/results/regions.png)


## Multi Model Sentiment Results

| S.NO | ID |textinput |Ground Label|Visual|Visual|Text|Text|Fusion|Fusion|
| ---| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|1|768|Help find my friend. 5 month old black lab pup lost today on Woodbury common, Devon. His mums is worried sick.|0|0.259307|0.740693|0.999341|0.999341|0.000658914|0.629324|0.370676|0
|2|326|you gotta let that hurt go.|1|0.313483|0.686517|0.00205311|0.00205311|0.997947|0.157768|0.842232|1
|3|783035469424631815|Happy b day lady killa and National Boyfriend Day to the loml|1|0.287737|0.712263|0.00000612558|0.00000612558|0.999994|0.143872|0.856128|1
|4|799334325032456192|I love obnoxious filters|1|0.21046|0.78954|0.682003|0.682003|0.317997|0.446231|0.553769|1
|5|200|RIP Young Prince, you will be missed young homie! He died of cat cancer|1|0.340688|0.659312|0.97087|0.97087|0.0291301|0.655779|0.344221|0
|6|65|Had to wait a million years to get into Hermès protest|0|0.141103|0.858897|0.953687|0.953687|0.0463129|0.547395|0.452605|0
|7|329|Be mine maybe puppy December wish|1|0.151342|0.848658|0.0000000698548|0.0000000698548|1|0.0756712|0.924329|1



## Requirements

- Python 3.7.11
- TensorFlow: 2.1.0
- Keras: 2.2.4
- OpenCV: 4.5.3
- Numpy: 1.19.1
- Matplotlib: 3.4.3




## Authors

- Zohair (邹海 Zōu hǎi)
- Prof Jianxin Wang
