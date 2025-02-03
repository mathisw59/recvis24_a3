# Results of the experiments

## First one: MobileNetV3

MobileNetV3 was fast to train and featured a good accuracy with a nice task transfer. It reached <b>0.41</b> accuracy using linear evaluation. 

## Second: ResNet18

I had to use a reduced batch size for unsupervised learning, as my GPU has a limited RAM. However, the SSL accuracy was great, reaching <b>0.9</b>. The transfer was however limited, reaching only <b>0.14</b> accuracy. Using only the weights obtained only via learning on ImageNet provided a poor accuracy of <b>0.043</b>. 

## Third model: ViT

I had to use a Vit-B-16 model since my GPU has not enough RAM. I had to use a batch size of 32 for unsupervised learning, which is ridiculously small compared to the batch size >> 512 advised by the SIMple Contrastive LeaRning (SimCLR) framework. Self-supervised learning using this framework was considered inpractical and thus not used. Linear evaluation of a model trained on ImageNet yielded an accuracy of <b>0.23</b>. 