__author__ = 'Brian M Anderson'
# Created on 4/13/2020

from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
from functools import partial, update_wrapper
# SGD = tf.train.experimental.enable_mixed_precision_graph_rewrite(SGD())
ExpandDimension = lambda axis: Lambda(lambda x: K.expand_dims(x, axis))
SqueezeDimension = lambda axis: Lambda(lambda x: K.squeeze(x, axis))
Subtract_new = lambda y: Lambda(lambda x: Subtract()([x, y]))
Multipy_new = lambda y: Lambda(lambda x: Multiply()([x, y]))


def return_hollow_layers_dict(layers=3):
    '''
    :param layers: Number of layers
    :return: A 'hollow' dictionary of what you should feed your network, populate each list
    with dictionaries from Return_Layer_Functions
    '''
    layers_dict = {}
    for layer in range(layers - 1):
        layers_dict['Layer_' + str(layer)] = {'Encoding': [],
                                              'Pooling':{'Encoding': [],
                                                         'Decoding': []},
                                              'Decoding': []}
    layers_dict['Base'] = []
    layers_dict['Final_Steps'] = []
    return layers_dict

class Return_Layer_Functions(object):
    def __init__(self, kernel=None, strides=None, padding=None, batch_norm=None, pool_size=None,
                 pooling_type=None):
        '''
        You can define any defaults here
        :param kernel: (3,3)
        :param strides: (1,1)
        :param padding: 'same' or 'valid'
        :param batch_norm: True or False
        :param pool_size: (2,2)
        '''
        self.set_default_kernel(kernel)
        self.set_default_padding(padding)
        self.set_default_strides(strides)
        self.set_default_batch_norm(batch_norm)
        self.set_default_pool_size(pool_size)
        self.set_default_pool_type(pooling_type)

    def set_default_kernel(self, kernel):
        '''
        :param kernel: default kernel, (3,3)
        :return:
        '''
        self.kernel = kernel

    def set_default_strides(self, strides):
        '''
        :param strides: default strides, (1,1)
        :return:
        '''
        self.strides = strides

    def set_default_padding(self, padding):
        '''
        :param padding: 'same' or 'valid'
        :return:
        '''
        self.padding = padding

    def set_default_batch_norm(self, batch_norm):
        '''
        :param batch_norm: True or False
        :return:
        '''
        self.batch_norm = batch_norm

    def set_default_pool_size(self, pool_size):
        '''
        :param pool_size: (2,2))
        :return:
        '''
        self.pool_size = pool_size

    def set_default_pool_type(self, pooling_type):
        '''
        :param pooling_type: 'Max' or 'Average'
        :return:
        '''
        self.pooling_type = pooling_type

    def convolution_layer(self, channels, type='convolution', kernel=None, activation=None, batch_norm=None, strides=None,
                          dialation_rate=1, padding='same', **kwargs):
        '''
        :param type: 'convolution' or 'tranpose'
        :param channels: # of channels
        :param kernel: kernel size, ex (3,3)
        :param activation: activation, ['relu','elu','linear','exponential','hard_sigmoid','sigmoid','tanh','softmax']
        :param batch_norm: perform batch_norm after convolution?
        :param strides: strides, (1,1), (2,2) for strided
        :param dialation_rate: rate for dialated convolution (atrous convolutions)
        :param padding: 'same' or 'valid'
        :return:
        '''
        if kernel is None:
            kernel = self.kernel
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        if batch_norm is None:
            batch_norm = self.batch_norm
        assert channels is not None, 'Need to provide a number of channels'
        assert kernel is not None, 'Need to provide a kernel, or set a default'
        assert strides is not None, 'Need to provide strides, or set a default'
        assert padding is not None, 'Need to provide padding, or set a default'
        assert batch_norm is not None, 'Need to provide batch_norm, or set a default'
        block = {type: {'channels':channels, 'kernel':kernel, 'activation':activation,
                        'batch_norm':batch_norm, 'strides':strides, 'dialation_rate':dialation_rate,
                        'padding':padding}}
        return block

    def residual_layer(self, submodules, batch_norm=False, activation=None, **kwargs):
        '''
        :param submodules: dictionary or list collection you want a residual connection across
        :param batch_norm: True/False for BN after convolution
        :param activation: activation, ['relu','elu','linear','exponential','hard_sigmoid','sigmoid','tanh','softmax']
        :return:
        '''
        return {'residual':{'submodules':submodules,'batch_norm':batch_norm, 'activation':activation}}

    def activation_layer(self, activation):
        '''
        :param activation: activation, ['relu','elu','linear','exponential','hard_sigmoid','sigmoid','tanh','softmax']
        :return:
        '''
        return {'activation':activation}

    def pooling_layer(self, pool_size=None, pooling_type=None):
        '''
        :param pooling_type: 'Max' or 'Average
        :param pool_size: tuple of pool size
        :return:
        '''
        if pool_size is None:
            pool_size = self.pool_size
        if pooling_type is None:
            pooling_type = self.pooling_type
        assert pooling_type is not None, "Need to pass a type to pooling layer ('Max', or 'Average')"
        assert pool_size is not None, 'Need to pass a pool_size (2,2), etc.'
        pooling = {'pooling': {'pool_size': pool_size, 'pooling_type': pooling_type}}
        return pooling

    def upsampling_layer(self, pool_size=None):
        '''
        :param pool_size: size of pooling (2,2), etc.
        :return:
        '''
        if pool_size is None:
            pool_size = self.pool_size
        assert pool_size is not None, 'Need to provide a pool size for upsampling!'
        return {'upsampling':{'pool_size':pool_size}}


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


class Unet(object):

    def __init__(self, save_memory=False, concat_not_add=True):
        self.previous_conv = None
        self.concat_not_add = concat_not_add
        self.save_memory = save_memory
        self.universe_index = 0

    def define_res_block(self, do_res_block=False):
        self.do_res_block = do_res_block

    def define_unet_dict(self, layers_dict):
        self.layers_dict = layers_dict
        self.layers_names = []
        layers = 0
        for name in layers_dict:
            if name.find('Layer') == 0:
                layers += 1
        for i in range(layers):
            self.layers_names.append('Layer_' + str(i))
        if 'Base' in layers_dict:
            self.layers_names.append('Base')
        return None

    def define_fc_dict(self, layers_dict_FC):
        self.layers_dict_FC = layers_dict_FC
        self.layer_names_fc = []
        layers = 0
        for name in layers_dict_FC:
            if name.find('Final') != 0:
                layers += 1
        for i in range(layers):
            self.layer_names_fc.append('Layer_' + str(i))
        if 'Final' in layers_dict_FC:
            self.layer_names_fc.append('Final')
        return None

    def FC_Block(self, output_size, x, sparse=False, dropout=0.0, name=''):
        for i in range(1):
            if sparse:
                x = Dense(output_size, activation='linear', activity_regularizer=sparse_reg, name=name)(x)
                x = Activation(self.activation)(
                    x)  # If using a regularization, you want to perform it before the activation, not after https://machinelearningmastery.com/how-to-reduce-generalization-error-in-deep-neural-networks-with-activity-regularization-in-keras/
            else:
                x = Dense(output_size, activation=self.activation,
                          name=name)(x)
            if dropout != 0.0:
                x = Dropout(dropout)(x)
        return x

    def run_FC_block(self, x, all_connections_list, name=''):
        variable_dropout = None
        if 'Connections' in all_connections_list:
            all_connections = all_connections_list['Connections']
            if 'Dropout' in all_connections_list:
                variable_dropout = all_connections_list['Dropout']
        else:
            all_connections = all_connections_list
        for i in range(len(all_connections)):
            if variable_dropout:
                self.drop_out = variable_dropout[i]
            x = self.FC_Block(all_connections[i], x, dropout=self.drop_out,
                              name=name + '_' + str(i))
        return x

    def do_FC_block(self, x):
        self.layer = 0
        self.desc = 'Encoder_FC'
        layer_order = []
        for layer in self.layer_names_fc:
            print(layer)
            if layer == 'Final':
                continue
            layer_order.append(layer)
            all_connections_list = self.layers_dict_FC[layer]['Encoding']
            x = self.run_FC_block(x, all_connections_list, name=self.desc + '_' + layer)
        layer_order.reverse()

        self.desc = 'Decoding_FC'
        for layer in layer_order:
            all_connections = self.layers_dict_FC[layer]['Decoding']
            x = self.run_FC_block(x, all_connections, name=self.desc + '_' + layer)
        if 'Final' in self.layers_dict_FC:
            self.desc = 'Final'
            all_connections = self.layers_dict_FC['Final']['Encoding']
            x = self.run_FC_block(x, all_connections, name=self.desc)
        return x

    def define_2D_or_3D(self, is_2D=False):
        self.is_2D = is_2D
        if is_2D:
            self.conv = Conv2D
            self.pool = MaxPooling2D
            self.up_sample = UpSampling2D
            self.tranpose_conv = Conv2DTranspose
        else:
            self.conv = Conv3D
            self.pool = MaxPooling3D
            self.up_sample = UpSampling3D
            self.tranpose_conv = Conv3DTranspose

    def define_batch_norm(self, batch_norm=False):
        self.batch_norm = batch_norm

    def define_kernel(self, kernel):
        self.kernel = kernel
        if len(kernel) == 2:
            self.define_2D_or_3D(True)
        else:
            self.define_2D_or_3D()

    def define_activation(self, activation):
        normal_activations = ['relu', 'elu', 'linear', 'exponential', 'hard_sigmoid', 'sigmoid', 'tanh']
        if type(activation) is str and activation.lower() in normal_activations:
            self.activation = partial(Activation, activation)
        elif type(activation) is dict:
            if 'kwargs' in activation:
                self.activation = partial(activation['activation'], **activation['kwargs'])
            else:
                self.activation = partial(activation['activation'])
        else:
            self.activation = activation

    def return_activation(self, activation):
        normal_activations = ['relu', 'elu', 'linear', 'exponential', 'hard_sigmoid', 'sigmoid', 'tanh', 'softmax']
        if type(activation) is str and activation.lower() in normal_activations:
            activation = partial(Activation, activation)
        elif type(activation) is dict:
            if 'kwargs' in activation:
                activation = partial(activation['activation'], **activation['kwargs'])
            else:
                activation = partial(activation['activation'])
        else:
            activation = activation
        return activation

    def define_pool_size(self, pool_size):
        self.pool_size = pool_size

    def define_padding(self, padding='same'):
        self.padding = padding

    def define_pooling_type(self, name='Max'):
        self.pooling_type = name

    def upsampling_block(self, x, name, pool_size=None):
        assert pool_size is not None and len(pool_size) < 4, 'Need to provide a pool_size tuple: ex. (2,2,2), (2,2)'
        if len(pool_size) == 3:
            x = UpSampling3D(size=pool_size, name='{}_3DUpSampling'.format(name))(x)
        elif len(pool_size) == 2:
            x = UpSampling2D(size=pool_size, name='{}_2DUpSampling'.format(name))(x)
        return x

    def pooling_block(self, x, name, pooling_type=None, pool_size=None):
        assert pool_size is not None and len(pool_size) < 4, 'Need to provide a pool_size tuple: ex. (2,2,2), (2,2)'
        assert pooling_type is not None, 'Need to provide a pool_type, "Max" or "Average"'
        if len(pool_size) == 3:
            if pooling_type == 'Max':
                x = MaxPooling3D(pool_size=pool_size, name='{}_3DMaxPooling'.format(name))(x)
            elif pooling_type == 'Average':
                x = AveragePooling3D(pool_size=pool_size, name='{}_3DAvgPooling'.format(name))(x)
        else:
            if pooling_type == 'Max':
                x = MaxPooling2D(pool_size=pool_size, name='{}_2DMaxPooling'.format(name))(x)
            elif pooling_type == 'Average':
                x = AveragePooling2D(pool_size=pool_size, name='{}_2DAvgPooling'.format(name))(x)
        return x

    def atrous_block(self, x, name, channels=None, kernel=None, atrous_rate=5, activations=None,
                     **kwargs):  # https://arxiv.org/pdf/1901.09203.pdf, follow formula of k^(n-1)
        # where n is the convolution layer number, this is for k = 3, 5 gives a field of 243x243
        if kernel is None:
            kernel = self.kernel
        rates = [[kernel[i] ** rate_block for i in range(len(kernel))] for rate_block in range(atrous_rate)]
        for i, rate in enumerate(rates):
            temp_name = name + 'Atrous_' + str(rate[-1])
            x = self.conv_block(channels=channels, x=x, name=temp_name, dialation_rate=rate, activate=False,
                                kernel=kernel)
            # x = self.conv(output_size,self.filters, activation=None,padding=self.padding, name=temp_name, dilation_rate=rate)(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            if i != len(rates) - 1:  # Don't activate last one
                if activations is not None:
                    if type(activations) is list:
                        if activations[i] is not 'linear':
                            x = self.return_activation(activations[i])(name=name + '_activation_{}'.format(i))(x)
                    elif activations is not 'linear':
                        x = self.return_activation(activations)(name=name + '_activation_{}'.format(i))(x)
                else:
                    x = self.activation(name=name + '_activation_{}'.format(i))(x)
        return x

    def residual_block(self, x, name, submodules, batch_norm=False, activation=None):
        input_val = x
        x = self.run_filter_dict(x, submodules, name, 'Residual')  # Loop through everything within
        if x.shape[-1] != input_val.shape[-1]:
            if self.is_2D:
                ones_kernel = tuple([1 for _ in range(2)])
            else:
                ones_kernel = tuple([1 for _ in range(3)])
            input_val = self.conv_block(channels=x.shape[-1], x=input_val, name='{}_Residual_Reshape'.format(name),
                                        activation=activation, kernel=ones_kernel, padding='same', batch_norm=batch_norm)
        x = Add(name=name + '_add')([x, input_val])
        return x

    def dict_block(self, x, name=None, **kwargs):
        conv_func = self.conv
        if 'residual' in kwargs:
            x = self.residual_block(x, name, **kwargs['residual'])
        elif 'transpose' in kwargs:
            conv_func = self.tranpose_conv
            x = self.conv_block(x, conv_func=conv_func, name=name, **kwargs['transpose'])
        elif 'convolution' in kwargs:
            x = self.conv_block(x, conv_func=conv_func, name=name, **kwargs['convolution'])
        elif 'atrous' in kwargs:
            x = self.atrous_block(x, name=name, **kwargs['atrous'])
        elif 'pooling' in kwargs:
            x = self.pooling_block(x, name=name, **kwargs['pooling'])
        elif 'upsampling' in kwargs:
            x = self.upsampling_block(x, name=name, **kwargs['upsampling'])
        elif 'activation' in kwargs:
            x = self.return_activation(kwargs['activation'])(name=name + '_activation_{}'.format(kwargs['activation']))(
                x)
        else:
            x = self.conv_block(x, conv_func=conv_func, name=name, **kwargs)
        return x

    def conv_block(self, x, channels=None, kernel=None, name=None, strides=None, dialation_rate=1, conv_func=None,
                   activation=None, batch_norm=False, padding=None, **kwargs):
        if strides is None:
            strides = 1
        if conv_func is None:
            conv_func = self.conv
        assert channels is not None, 'Need to provide "channels"'
        if kernel is None:
            kernel = self.kernel
        x = conv_func(int(channels), kernel_size=kernel, activation=None, padding=padding,
                      name=name, strides=strides, dilation_rate=dialation_rate)(x)

        if batch_norm:
            x = BatchNormalization()(x)
        if activation is not None:
            x = self.return_activation(activation)(name=name + '_activation')(x)
        return x

    def run_filter_dict(self, x, layer_dict, layer, desc):
        if type(layer_dict) is list:
            for dictionary in layer_dict:
                if type(dictionary) is dict:
                    x = self.dict_block(x, name='{}_{}_{}'.format(self.universe_index, layer, desc), **dictionary)
                    self.universe_index += 1
                else:
                    x = self.run_filter_dict(x, dictionary, layer=layer, desc=desc)
        else:
            x = self.dict_block(x, name='{}_{}_{}'.format(self.universe_index, layer, desc), **layer_dict)
            self.universe_index += 1
        return x

    def run_unet(self, x):
        self.layer = 0
        self.layer_vals = {}
        desc = 'Encoder'
        layer_index = 0
        layer_order = []
        for layer in self.layers_names:
            print(layer)
            if layer.find('Layer') == -1:
                continue
            layer_order.append(layer)
            all_filters = self.layers_dict[layer]['Encoding']
            x = self.run_filter_dict(x, all_filters, layer, desc)
            self.layer_vals[layer_index] = x
            if 'Pooling' in self.layers_dict[layer]:
                if 'Encoding' in self.layers_dict[layer]['Pooling']:
                    x = self.run_filter_dict(x, self.layers_dict[layer]['Pooling']['Encoding'],
                                             '{}_Encoding_Pooling'.format(layer), '')
            layer_index += 1
        concat = False
        if 'Base' in self.layers_dict:
            concat = True
            all_filters = self.layers_dict['Base']
            x = self.run_filter_dict(x, all_filters, 'Base_', '')
        desc = 'Decoder'
        self.layer = 0
        layer_order.reverse()
        for layer in layer_order:
            if layer.find('Layer') == -1 or 'Decoding' not in self.layers_dict[layer]:
                continue
            print(layer)
            layer_index -= 1
            if 'Pooling' in self.layers_dict[layer]:
                if 'Decoding' in self.layers_dict[layer]['Pooling']:
                    x = self.run_filter_dict(x, self.layers_dict[layer]['Pooling']['Decoding'],
                                             '{}_Decoding_Pooling'.format(layer), '')
            if concat:
                x = Concatenate(name='concat' + str(self.layer) + '_Unet')([x, self.layer_vals[layer_index]])
            all_filters = self.layers_dict[layer]['Decoding']
            x = self.run_filter_dict(x, all_filters, layer, desc)
            self.layer += 1
        if 'Final_Steps' in self.layers_dict:
            print('Final_Steps')
            x = self.run_filter_dict(x, self.layers_dict['Final_Steps'], 'Final_Steps', '')
        return x


class base_UNet(Unet):
    def __init__(self, layers_dict=None, is_2D=False, save_memory=False, concat_not_add=True):
        super().__init__(save_memory=save_memory, concat_not_add=concat_not_add)
        self.layer_vals = {}
        self.define_2D_or_3D(is_2D)
        self.define_unet_dict(layers_dict)

    def get_unet(self, layers_dict):
        pass


class my_UNet(base_UNet):
    def __init__(self, layers_dict=None, create_model=True, z_images=None, tensor_input=None,
                 out_classes=2, is_2D=False, save_memory=False, mask_output=False, image_size=None,
                 custom_loss=None, mask_loss=False, concat_not_add=True):
        self.mask_loss = mask_loss
        self.custom_loss = custom_loss
        self.tensor_input = tensor_input
        self.image_size = image_size
        self.z_images = z_images
        self.previous_conv = None
        assert layers_dict is not None, 'Need to pass a layers dictionary'
        self.is_2D = is_2D
        self.create_model = create_model
        super().__init__(layers_dict=layers_dict, is_2D=is_2D, save_memory=save_memory, concat_not_add=concat_not_add)
        self.out_classes = out_classes
        self.mask_output = mask_output
        self.get_unet(layers_dict)

    def get_unet(self, layers_dict):
        if self.tensor_input is None:
            image_input_primary = x = Input(self.image_size, name='UNet_Input')
        else:
            image_input_primary = x = self.tensor_input
        x = self.run_unet(x)
        if self.mask_loss or self.mask_output:
            self.mask = Input(shape=(None, None, None, self.out_classes), name='mask')
            self.sum_vals = Input(shape=(None, None, None, self.out_classes), name='sum_vals')
            inputs = [image_input_primary, self.mask, self.sum_vals]
            if self.mask_loss:
                assert self.custom_loss is not None, 'Need to specify a custom loss when using masked input'
                partial_func = partial(self.custom_loss, mask=self.mask)
                self.custom_loss = update_wrapper(partial_func, self.custom_loss)
            if self.mask_output:
                x = Multiply()([self.mask, x])
                x = Add()([self.sum_vals, x])
        else:
            inputs = image_input_primary
        if self.create_model:
            model = Model(inputs=inputs, outputs=x)
            self.created_model = model
        self.output = x


if __name__ == '__main__':
    pass
