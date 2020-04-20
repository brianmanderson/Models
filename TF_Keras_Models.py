__author__ = 'Brian M Anderson'
# Created on 4/13/2020

from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
from functools import partial, update_wrapper
# SGD = tf.train.experimental.enable_mixed_precision_graph_rewrite(SGD)
ExpandDimension = lambda axis: Lambda(lambda x: K.expand_dims(x, axis))
SqueezeDimension = lambda axis: Lambda(lambda x: K.squeeze(x, axis))
Subtract_new = lambda y: Lambda(lambda x: Subtract()([x, y]))
Multipy_new = lambda y: Lambda(lambda x: Multiply()([x, y]))


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def convolution_layer_dict():
    '''
    Things you can pass to the convolution layer dictionary
    {'convolution':{'channels':a,'kernel':b,'activation':c,'padding':d,'strides':e'}}
    :param channels: number of filters
    :param kernel: kernel size
    :param activation: what activation, defaults to None
    :param padding: valid/same
    :param strides: stride size
    '''
    return lambda a, b, c, d, e: {
        'convolution': {'channels': a, 'kernel': b, 'activation': c, 'padding': d, 'strides': e}}


def pooling_layer_dict():
    '''
    Things you can pass to the pooling layer dictionary
    {'pooling':{'pooling_type':'Max','pool_size':(2,2),'direction':'Down'}}
    {'pooling':{'pool_size':(2,2),'direction':'Up'}}
    :param pooling_type: for down sampling, can be 'Max' or 'Average
    :param pool_size: side of kernel for pooling
    :param direction: 'Up' for up sampling, 'Down' for downsampling
    '''
    return lambda a, b, c: {'pooling': {'pooling_type': 'Max', 'pool_size': b, 'direction': c}}


def residual_layer():
    '''
    Things you can pass to residual layer
    {'residual':[]}
    pass a list of other layers and a residual connection will be performed between them
    '''
    return lambda a: {'residual': a}


def activation_layer():
    '''
    Things you can pass to an activation layer
    :param activations: list of ['relu','elu','linear','exponential','hard_sigmoid','sigmoid','tanh','softmax'], or
    you can pass a dictionary for kwargs
    {'activation':'relu'}
    '''
    return lambda x: {'activation': x}


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

    def strided_conv_block(self, output_size, x, name, strides=(2, 2, 2)):
        x = Conv3DTranspose(output_size, self.filters, strides=strides, padding=self.padding,
                            name=name)(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
        x = Activation(self.activation, name=name + '_activation')(x)
        return x

    def define_pooling_type(self, name='Max'):
        self.pooling_type = name

    def pooling_block(self, x, name, pooling_type='Max', pool_size=None, direction=None):
        if pool_size is None:
            pool_size = self.pool_size
        if pooling_type is None:
            pooling_type = self.pooling_type
        assert direction is 'Up' or direction is 'Down', 'Need to provide direction in pooling:{' \
                                                         '"direction":("Up" or "Down")}, {} was given'.format(direction)
        if direction is 'Down':
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
        else:
            if len(pool_size) == 3:
                x = UpSampling3D(size=pool_size, name='{}_3DUpSampling'.format(name))(x)
            elif len(pool_size) == 2:
                x = UpSampling2D(size=pool_size, name='{}_2DUpSampling'.format(name))(x)
        return x

    def shared_conv_block(self, x, y, output_size, name, strides=1):
        layer = Conv3D(output_size, self.filters, activation=None, padding=self.padding, name=name, strides=strides)
        x = layer(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
        x = Activation(self.activation, name=name + '_activation')(x)
        y = layer(y)
        if self.batch_norm:
            x = BatchNormalization()(x)
        y = Activation(self.activation, name=name + '_activation')(y)
        return x, y

    def do_conv_block_enc(self, x):
        self.layer = 0
        layer_vals = {}
        desc = 'Encoder'
        self.layer_index = 0
        self.layer_order = []
        for layer in self.layers_names:
            print(layer)
            if layer == 'Base':
                continue
            self.layer_order.append(layer)
            all_filters = self.layers_dict[layer]['Encoding']
            x = self.run_filter_dict(x, all_filters, layer, desc)
            layer_vals[self.layer_index] = x
            if 'Pooling' in self.layers_dict[layer]:
                self.define_pool_size(self.layers_dict[layer]['Pooling'])
            if len(self.layers_names) > 1:
                x = self.pooling_down_block(x, layer + '_Pooling')
            self.layer_index += 1
        self.concat = False
        if 'Base' in self.layers_dict:
            self.concat = True
            all_filters = self.layers_dict['Base']['Encoding']
            x = self.run_filter_dict(x, all_filters, 'Base_', '')
        return x, layer_vals

    def do_conv_block_decode(self, x, layer_vals=None):
        desc = 'Decoder'
        self.layer = 0
        self.layer_order.reverse()
        for layer in self.layer_order:
            if 'Decoding' not in self.layers_dict[layer]:
                continue
            print(layer)
            self.layer_index -= 1
            if 'Pooling' in self.layers_dict[layer]:
                self.define_pool_size(self.layers_dict[layer]['Pooling'])
            if self.concat:
                x = self.up_sample(size=self.pool_size, name='Upsampling' + str(self.layer) + '_UNet')(x)
                x = Concatenate(name='concat' + str(self.layer) + '_Unet')([x, layer_vals[self.layer_index]])
            all_filters = self.layers_dict[layer]['Decoding']
            x = self.run_filter_dict(x, all_filters, layer, desc)
            self.layer += 1
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

    def residual_block(self, x, name, submodules):
        input_val = x
        x = self.run_filter_dict(x, submodules, name, 'Residual')  # Loop through everything within
        if x.shape[-1] != input_val.shape[-1]:
            ones_kernel = tuple([1 for _ in range(len(self.kernel))])
            input_val = self.conv_block(channels=x.shape[-1], x=input_val, name='{}_Residual_Reshape'.format(name),
                                        activate=False,
                                        kernel=ones_kernel)
        x = Add(name=name + '_add')([x, input_val])
        return x

    def dict_block(self, x, name=None, **kwargs):
        conv_func = self.conv
        if 'residual' in kwargs:
            x = self.residual_block(x, name, submodules=kwargs['residual'])
        elif 'transpose' in kwargs:
            conv_func = self.tranpose_conv
            x = self.conv_block(x, conv_func=conv_func, name=name, **kwargs['transpose'])
        elif 'convolution' in kwargs:
            x = self.conv_block(x, conv_func=conv_func, name=name, **kwargs['convolution'])
        elif 'atrous' in kwargs:
            x = self.atrous_block(x, name=name, **kwargs['atrous'])
        elif 'pooling' in kwargs:
            x = self.pooling_block(x, name=name, **kwargs['pooling'])
        elif 'activation' in kwargs:
            x = self.return_activation(kwargs['activation'])(name=name + '_activation_{}'.format(kwargs['activation']))(
                x)
        else:
            x = self.conv_block(x, conv_func=conv_func, name=name, **kwargs)
        return x

    def conv_block(self, x, channels=None, kernel=None, name=None, strides=None, dialation_rate=1, conv_func=None,
                   activation=None, **kwargs):
        if strides is None:
            strides = 1
        if conv_func is None:
            conv_func = self.conv
        assert channels is not None, 'Need to provide "channels"'
        if kernel is None:
            kernel = self.kernel
        x = conv_func(int(channels), kernel_size=kernel, activation=None, padding=self.padding,
                      name=name, strides=strides, dilation_rate=dialation_rate)(x)

        if self.batch_norm:
            x = BatchNormalization()(x)
        if activation is not None:
            x = self.return_activation(activation)(name=name + '_activation')(x)
        return x

    def dict_conv_block(self, x, desc, kernels=None, res_blocks=None, atrous_blocks=None, up_sample_blocks=None,
                        down_sample_blocks=None, activations=None, strides=None, channels=None, type='Conv', **kwargs):
        conv_func = self.conv
        if type == 'Transpose':
            conv_func = self.tranpose_conv
        elif type == 'Upsample':
            up_sample_blocks = 1
        rescale = False
        if type != 'Upsample' and type != 'Downsample':
            for i in range(len(channels)):
                stride = None
                if strides is not None:
                    stride = strides[i]
                # if activations:
                #     self.define_activation(activations[i])
                if kernels:
                    self.define_filters(kernels[i])
                    if len(kernels[i]) + 1 == len(x.shape):
                        self.define_2D_or_3D(is_2D=False)
                        x = ExpandDimension(0)(x)
                        rescale = True
                    elif len(kernels[i]) + 1 > len(x.shape):
                        self.define_2D_or_3D(True)
                        x = SqueezeDimension(0)(x)
                if rescale:
                    self.desc = desc + '3D_' + str(i)
                else:
                    self.desc = desc + str(i)

                if res_blocks:
                    rate = res_blocks[i] if res_blocks else 0
                    x = self.residual_block(channels[i], x=x, name=self.desc, blocks=rate)
                elif atrous_blocks:
                    x = self.atrous_block(channels[i], x=x, name=self.desc, rate_blocks=atrous_blocks[i],
                                          activations=activations)
                elif up_sample_blocks is not None:
                    if stride is not None:
                        self.define_pool_size(stride)
                    x = self.up_sample(self.pool_size)(x)
                else:
                    activation = None
                    if activations is not None:
                        activation = activations[i]
                    x = self.conv_block(channels[i], x=x, strides=stride, name=self.desc, conv_func=conv_func,
                                        activation=activation)
        elif type == 'Upsample':
            if strides is not None:
                for i in range(len(strides)):
                    self.define_pool_size(strides[i])
                    x = self.up_sample(self.pool_size)(x)
            else:
                x = self.up_sample(self.pool_size)(x)
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
    def __init__(self, kernel=(3, 3, 3), layers_dict=None, pool_size=(2, 2, 2), activation='elu', pool_type='Max',
                 batch_norm=False, is_2D=False, save_memory=False, concat_not_add=True):
        super().__init__(save_memory=save_memory, concat_not_add=concat_not_add)
        self.layer_vals = {}
        self.define_2D_or_3D(is_2D)
        self.define_unet_dict(layers_dict)
        self.define_pool_size(pool_size)
        self.define_batch_norm(batch_norm)
        self.define_kernel(kernel)
        self.define_activation(activation)
        self.define_padding('same')
        self.define_pooling_type(pool_type)

    def get_unet(self, layers_dict):
        pass


class my_UNet(base_UNet):
    def __init__(self, kernel=(3, 3, 3), layers_dict=None, pool_size=(2, 2, 2), create_model=True, activation='relu',
                 pool_type='Max', z_images=None, complete_input=None,
                 batch_norm=False, striding_not_pooling=False, out_classes=2, is_2D=False,
                 input_size=1, save_memory=False, mask_output=False, image_size=None,
                 custom_loss=None, mask_loss=False, concat_not_add=True):

        self.mask_loss = mask_loss
        self.custom_loss = custom_loss
        self.complete_input = complete_input
        self.image_size = image_size
        self.z_images = z_images
        self.previous_conv = None
        if not layers_dict:
            print('Need to pass in a dictionary')
        self.is_2D = is_2D
        self.input_size = input_size
        self.create_model = create_model
        super().__init__(kernel=kernel, layers_dict=layers_dict, pool_size=pool_size, activation=activation,
                         pool_type=pool_type, batch_norm=batch_norm, is_2D=is_2D, save_memory=save_memory,
                         concat_not_add=concat_not_add)
        self.striding_not_pooling = striding_not_pooling
        self.out_classes = out_classes
        self.mask_output = mask_output
        self.get_unet(layers_dict)

    def get_unet(self, layers_dict):
        if self.complete_input is None:
            if self.is_2D:
                image_input_primary = x = Input(shape=(self.image_size, self.image_size, self.input_size),
                                                name='UNet_Input')
            else:
                image_input_primary = x = Input(
                    shape=(self.z_images, self.image_size, self.image_size, self.input_size), name='UNet_Input')
        else:
            image_input_primary = x = self.complete_input
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
                # KeyError('Do not use mask_output, does not seem to work')
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
