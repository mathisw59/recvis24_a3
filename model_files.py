to_do_list = [
    'unsupervised_basic_cnn.yaml',
    'unsupervised_squeezenet.yaml',
    'unsupervised_mobilenetv3yaml',
    'unsupervised_resnet.yaml',
    'unsupervised_vit.yaml',

    'supervised_basic_cnn.yaml',
    'supervised_squeezenet.yaml',
    'supervised_mobilenetv3.yaml',
    'supervised_resnet.yaml',
    'supervised_vit.yaml',

    'supervised_basic_cnn_scratch.yaml',
    'supervised_squeezenet_scratch.yaml',
    'supervised_mobilenetv3_scratch.yaml',
    'supervised_resnet_scratch.yaml',
    'supervised_vit_scratch.yaml',
]

import os

to_do_list = [os.path.join('configuration_files', f) for f in to_do_list]