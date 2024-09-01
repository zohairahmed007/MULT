
# Sentimentality Beyond Words: A Multi-Modal Transformer Approach for Text and Image Sentiment Analysis

This repository includes the original implementation of "**Sentimentality Beyond Words: A Multi-Modal Transformer Approach for Text and Image Sentiment Analysis**"  in TensorFlow 2.x with TFX as the backend and Pytorch. 

## Acknowledgement:
- This work has been conducted under the supervision of `Prof JX Wang` and School of Computer Science and Engineering, `Central South University`, China.


## Proposed Architecture

-  The novel sentiment  model that leverages the power of Multi-modal Transformers (MULT) to seamlessly integrate and process both textual and visual information. 


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

![alt text](https://github.com/zohairahmed007/MULT/blob/main/results/regions.png?raw=true)


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
