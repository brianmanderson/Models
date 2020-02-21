## Most commonly used for 'my_3D_UNet', note this works for 2D and 3D models
## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!
### Feed layer dictionaries to this model via something like below, this will make a simple 2D UNet with residual connections
    from Keras_3D_Models import my_3D_UNet
    
    num_classes = 2
    layers_dict = {}
    conv_block = lambda x: {'convolution': {'channels': x, 'kernel': (3, 3), 'activation': None, 'strides': (1, 1)}}
    act = lambda x: {'activation': x}
    pooling_downsampling = {'pooling': {'pooling_type': 'Max', 'pool_size': (2, 2), 'direction': 'Down'}}
    pooling_upsampling = {'pooling': {'pool_size': (2, 2), 'direction': 'Up'}}
    residual_block = lambda x: {'residual': x}
    layers_dict['Layer_0'] = {'Encoding': {},
                              'Decoding': {},
                              'Pooling': {'Encoding': {},
                                          'Decoding': {}
                                          }}
    layers_dict['Layer_1'] = {'Encoding': {},
                              'Decoding': {},
                              'Pooling': {'Encoding': {},
                                          'Decoding': {}
                                          }}
    layers_dict['Base'] = {}
    
    enc = dec = lambda x: residual_block([act('relu'), conv_block(x), act('relu'), conv_block(x)])
    layers_dict['Layer_0']['Encoding'] = enc(16)
    layers_dict['Layer_0']['Pooling']['Encoding'] = pooling_downsampling
    layers_dict['Layer_1']['Encoding'] = enc(32)
    layers_dict['Layer_1']['Pooling']['Encoding'] = pooling_downsampling
    
    layers_dict['Base'] = enc(64)
    
    layers_dict['Layer_1']['Pooling']['Decoding'] = pooling_upsampling
    layers_dict['Layer_1']['Decoding'] = dec(64)
    layers_dict['Layer_0']['Pooling']['Decoding'] = pooling_upsampling
    layers_dict['Layer_0']['Decoding'] = dec(32)
    
    layers_dict['Final_Steps'] = [act('relu'),
                                  {'convolution': {'channels': num_classes, 'kernel': (1, 1), 'activation': 'softmax'}}
                                  ]
    model = my_3D_UNet(kernel=(3, 3), layers_dict=layers_dict, pool_size=(2, 2))