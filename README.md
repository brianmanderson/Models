## Most commonly used for 'my_3D_UNet', note this works for 2D and 3D models
## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!
### Feed layer dictionaries to this model via something like below, this will make a simple 2D UNet with residual connections
    from Keras_3D_Models import my_3D_UNet
    
    num_classes = 2
    layers_dict = {}
    conv_block = lambda x: {
        'convolution': {'channels': x, 'kernel': (3, 3), 'activation': 'relu'}}
    residual_block = lambda x: {'residual':x}
    pooling_upsampling = {'Encoding':{'Pool_Size':(2,2),'Pooling_Type':'Max'},
                          'Decoding':{'Pool_Size':(2,2)}}
    encoding = [conv_block(32),conv_block(32)]
    decoding = [conv_block(64), conv_block(64)]
    layers_dict['Layer_0'] = {'Encoding':residual_block(encoding),'Decoding':residual_block(decoding),
                              'Pooling':pooling_upsampling}
    encoding = [conv_block(64),conv_block(64)]
    decoding = [conv_block(64), conv_block(64)]
    layers_dict['Layer_1'] = {'Encoding':residual_block(encoding),'Decoding':residual_block(decoding),
                              'Pooling':pooling_upsampling}
    base = [conv_block(64),conv_block(64)]
    layers_dict['Base'] = [residual_block(base)]
    layers_dict['Final_Steps'] = [{'convolution': {'channels': num_classes, 'kernel': (1, 1), 'activation': 'softmax'}}]

    model = my_3D_UNet(kernel=(3, 3), layers_dict=layers_dict, pool_size=(2, 2), custom_loss=None,
                       batch_norm=False, pool_type='Max')
