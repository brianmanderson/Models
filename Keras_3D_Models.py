from tensorflow.python.keras.models import Model
import tensorflow.python.keras.backend as K
import tensorflow as tf
from tensorflow.python.keras.backend import variable
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.models import load_model
from functools import partial, update_wrapper
import numpy as np


ExpandDimension = lambda axis: Lambda(lambda x: K.expand_dims(x, axis))
SqueezeDimension = lambda axis: Lambda(lambda x: K.squeeze(x, axis))
Subtract_new = lambda y: Lambda(lambda x: Subtract()([x,y]))
Multipy_new = lambda y: Lambda(lambda x: Multiply()([x,y]))

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


class Unet(object):

    def __init__(self,save_memory=False, concat_not_add=True):
        self.previous_conv = None
        self.concat_not_add = concat_not_add
        self.save_memory = save_memory

    def define_res_block(self,do_res_block=False):
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

    def FC_Block(self,output_size,x, sparse=False, dropout=0.0,name=''):
        for i in range(1):
            if sparse:
                x = Dense(output_size, activation='linear', activity_regularizer=sparse_reg,name=name)(x)
                x = Activation(self.activation)(x) # If using a regularization, you want to perform it before the activation, not after https://machinelearningmastery.com/how-to-reduce-generalization-error-in-deep-neural-networks-with-activity-regularization-in-keras/
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
        normal_activations = ['relu','elu','linear','exponential','hard_sigmoid','sigmoid','tanh']
        if type(activation) is str and activation.lower() in normal_activations:
            self.activation = partial(Activation,activation)
        elif type(activation) is dict:
            if 'kwargs' in activation:
                self.activation = partial(activation['activation'],**activation['kwargs'])
            else:
                self.activation = partial(activation['activation'])
        else:
            self.activation = activation

    def return_activation(self, activation):
        normal_activations = ['relu','elu','linear','exponential','hard_sigmoid','sigmoid','tanh','softmax']
        if type(activation) is str and activation.lower() in normal_activations:
            activation = partial(Activation,activation)
        elif type(activation) is dict:
            if 'kwargs' in activation:
                activation = partial(activation['activation'],**activation['kwargs'])
            else:
                activation = partial(activation['activation'])
        else:
            activation = activation
        return activation

    def define_pool_size(self, pool_size):
        self.pool_size = pool_size

    def define_padding(self, padding='same'):
        self.padding = padding

    def residual_block(self, output_size,x,name,blocks=0, activation=None):
        if x.shape[-1] != output_size:
            x = self.conv_block(output_size,x=x,name=name + '_' + 'rescale_input',activate=False,filters=self.filters)
            if activation is not None:
                x = input_val = self.return_activation(activation)(name=name + '_activation')(x)
            else:
                x = input_val = self.activation(name=name + '_activation')(x)
        else:
            input_val = x

        for i in range(blocks):
            x = self.conv_block(output_size,x,name=name+'_'+str(i))
        x = self.conv(output_size, self.filters, activation=None, padding=self.padding,name=name)(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
        x = Add(name=name+'_add')([x,input_val])
        if activation is not None:
            x = self.return_activation(activation)(name=name + '_activation')(x)
        else:
            x = self.activation(name=name + '_activation')(x)
        return x

    def strided_conv_block(self, output_size, x, name, strides=(2,2,2)):
        x = Conv3DTranspose(output_size, self.filters, strides=strides, padding=self.padding,
                            name=name)(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
        x = Activation(self.activation, name=name+'_activation')(x)
        return x

    def define_pooling_type(self,name='Max'):
        self.pooling_name = name

    def pooling_down_block(self, x, desc):
        if not self.is_2D:
            if self.pooling_name == 'Max':
                x = MaxPooling3D(pool_size=self.pool_size, name=desc)(x)
            elif self.pooling_name == 'Average':
                x = AveragePooling3D(pool_size=self.pool_size, name=desc)(x)
        else:
            if self.pooling_name == 'Max':
                x = MaxPooling2D(pool_size=self.pool_size, name=desc)(x)
            elif self.pooling_name == 'Average':
                x = AveragePooling2D(pool_size=self.pool_size, name=desc)(x)
        return x

    def shared_conv_block(self, x, y, output_size, name, strides=1):
        layer = Conv3D(output_size, self.filters, activation=None, padding=self.padding, name=name, strides=strides)
        x = layer(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
        x = Activation(self.activation, name=name+'_activation')(x)
        y = layer(y)
        if self.batch_norm:
            x = BatchNormalization()(x)
        y = Activation(self.activation, name=name+'_activation')(y)
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

    def do_conv_block_decode(self, x, layer_vals = None):
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

    def atrous_block(self, x, name, channels=None, kernel=None,atrous_rate=5, activations=None, **kwargs): # https://arxiv.org/pdf/1901.09203.pdf, follow formula of k^(n-1)
        # where n is the convolution layer number, this is for k = 3, 5 gives a field of 243x243
        if kernel is None:
            kernel = self.kernel
        rates = [[kernel[i]**rate_block for i in range(len(kernel))] for rate_block in range(atrous_rate)]
        x = self.activation(name='{}_pre_activation'.format(name))(x)
        for i, rate in enumerate(rates):
            temp_name = name + 'Atrous_' + str(rate[-1])
            x = self.conv_block(channels=channels,x=x,name=temp_name,dialation_rate=rate,activate=False, kernel=kernel)
            # x = self.conv(output_size,self.filters, activation=None,padding=self.padding, name=temp_name, dilation_rate=rate)(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            if i != len(rates) - 1: # Don't activate last one
                if activations is not None:
                    if type(activations) is list:
                        if activations[i] is not 'linear':
                            x = self.return_activation(activations[i])(name=name+'_activation_{}'.format(i))(x)
                    elif activations is not 'linear':
                        x = self.return_activation(activations)(name=name + '_activation_{}'.format(i))(x)
                else:
                    x = self.activation(name=name+'_activation_{}'.format(i))(x)
        return x

    def dict_block(self, x, name=None, **kwargs):
        conv_func = self.conv
        if 'residual' in kwargs:
            input_val = x
            sub_modules = kwargs['residual']
            x = self.run_filter_dict(x, sub_modules, name, 'Residual')
            if x.shape[-1] != input_val.shape[-1]:
                ones_kernel = tuple([1 for _ in range(len(self.kernel))])
                input_val = self.conv_block(channels=x.shape[-1], x=input_val, name='{}_Residual_Reshape'.format(name), activate=False,
                                            kernel=ones_kernel)
            x = Add(name=name + '_add')([x, input_val])
        elif 'transpose' in kwargs:
            conv_func = self.tranpose_conv
            x = self.conv_block(x, conv_func=conv_func, name=name, **kwargs['transpose'])
        elif 'convolution' in kwargs:
            x = self.conv_block(x,conv_func=conv_func,name=name,**kwargs['convolution'])
        elif 'atrous' in kwargs:
            x = self.atrous_block(x, name=name,**kwargs['atrous'])
        else:
            x = self.conv_block(x, conv_func=conv_func, name=name, **kwargs)
        return x

    def conv_block(self,x,channels=None,kernel=None,name=None, strides=None, dialation_rate=1,conv_func=None,
                   activation=None,activate=True, **kwargs):
        if strides is None:
            strides = 1
        if conv_func is None:
            conv_func = self.conv
        assert channels is not None, 'Need to provide "channels"'
        if kernel is None:
            kernel = self.kernel
        x = conv_func(int(channels), kernel_size=kernel,activation=None, padding=self.padding,
                      name=name, strides=strides, dilation_rate=dialation_rate)(x)

        if self.batch_norm:
            x = BatchNormalization()(x)
        if activate:
            if activation is not None and activation is not 'linear':
                x = self.return_activation(activation)(name=name + '_activation')(x)
            elif activation is not 'linear':
                x = self.activation(name=name + '_activation')(x)
        return x
    def dict_conv_block(self, x, desc, kernels=None,res_blocks=None,atrous_blocks=None,up_sample_blocks=None,
                        down_sample_blocks=None,activations=None,strides=None,channels=None,type='Conv',**kwargs):
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
                    x = self.residual_block(channels[i], x=x, name=self.desc,blocks=rate)
                elif atrous_blocks:
                    x = self.atrous_block(channels[i],x=x,name=self.desc,rate_blocks=atrous_blocks[i],activations=activations)
                elif up_sample_blocks is not None:
                    if stride is not None:
                        self.define_pool_size(stride)
                    x = self.up_sample(self.pool_size)(x)
                else:
                    activation = None
                    if activations is not None:
                        activation = activations[i]
                    x = self.conv_block(channels[i], x=x, strides=stride, name=self.desc, conv_func=conv_func, activation=activation)
        elif type == 'Upsample':
            if strides is not None:
                for i in range(len(strides)):
                    self.define_pool_size(strides[i])
                    x = self.up_sample(self.pool_size)(x)
            else:
                x = self.up_sample(self.pool_size)(x)
        return x

    def run_filter_dict(self, x, layer_dict, layer, desc):
        if type(layer_dict) == list:
            for i, dictionary in enumerate(layer_dict):
                x = self.dict_block(x, name=layer + '_' + desc + '_' + str(i), **dictionary)
        else:
            x = self.dict_block(x, name=layer + '_' + desc + '_', **layer_dict)
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
            if 'Pooling' in self.layers_dict[layer] or ('Pooling' in self.layers_dict[layer] and self.layers_dict[layer]['Pooling'] is not None):
                if 'Encoding' in self.layers_dict[layer]['Pooling']:
                    x = self.run_filter_dict(x, self.layers_dict[layer]['Pooling']['Encoding'], layer, 'strided_conv')
                else:
                    if 'Pool_Size' in self.layers_dict[layer]['Pooling']:
                        self.define_pool_size(self.layers_dict[layer]['Pooling']['Pool_Size'])
                    if 'Pooling_Type' in self.layers_dict[layer]:
                        self.define_pooling_type(self.layers_dict[layer]['Pooling_Type'])
                    if len(self.layers_names) > 1:
                        x = self.pooling_down_block(x, layer + '_Pooling')
            layer_index += 1
        concat = False
        if 'Base' in self.layers_dict:
            concat = True
            all_filters = self.layers_dict['Base']['Encoding']
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
                    x = self.run_filter_dict(x, self.layers_dict[layer]['Pooling']['Decoding'], layer, 'transpose_conv')
                else:
                    if 'Pool_Size' in self.layers_dict[layer]['Pooling']:
                        self.define_pool_size(self.layers_dict[layer]['Pooling']['Pool_Size'])
                    x = self.up_sample(size=self.pool_size, name='Upsampling' + str(self.layer) + '_UNet')(x)
                    if x.shape[-1] != self.layer_vals[layer_index].shape[-1] and not self.concat_not_add:
                        x = self.conv_block(x,channels=self.layer_vals[layer_index].shape[-1],name='Conv_' + desc + str(layer))
            if x.shape[-1] == self.layer_vals[layer_index].shape[-1] and not self.concat_not_add:
                x = Add(name='Add_' + desc + str(layer))([x,self.layer_vals[layer_index]])
                x = self.activation(name='Activate_' + desc + str(layer))(x)
            elif concat:
                x = Concatenate(name='concat' + str(self.layer) + '_Unet')([x, self.layer_vals[layer_index]])
            all_filters = self.layers_dict[layer]['Decoding']
            x = self.run_filter_dict(x, all_filters, layer, desc)
            self.layer += 1
        if 'Final_Steps' in self.layers_dict:
            x = self.run_filter_dict(x, self.layers_dict['Final_Steps'], 'Final_Steps', '')
        return x


class base_UNet(Unet):
    def __init__(self, kernel=(3,3,3),layers_dict=None, pool_size=(2,2,2),activation='elu', pool_type='Max',
                 batch_norm=False,is_2D=False,save_memory=False,concat_not_add=True):
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

    def get_unet(self,layers_dict):
        pass


class my_3D_UNet(base_UNet):
    def __init__(self, kernel=(3,3,3),layers_dict=None, pool_size=(2,2,2),create_model=True, activation='relu',
                 pool_type='Max',final_activation='softmax',z_images=None,complete_input=None,
                 batch_norm=False, striding_not_pooling=False, out_classes=2,is_2D=False,semantic_segmentation=True,
                 input_size=1,save_memory=False, mask_output=False, image_size=None,
                 custom_loss=None, mask_loss=False, concat_not_add=True):
        self.mask_loss = mask_loss
        self.custom_loss = custom_loss
        self.semantic_segmentation = semantic_segmentation
        self.complete_input = complete_input
        self.image_size = image_size
        self.z_images = z_images
        self.previous_conv = None
        self.final_activation = final_activation
        if not layers_dict:
            print('Need to pass in a dictionary')
        self.is_2D = is_2D
        self.input_size = input_size
        self.create_model = create_model
        super().__init__(kernel=kernel, layers_dict=layers_dict, pool_size=pool_size, activation=activation,
                         pool_type=pool_type, batch_norm=batch_norm, is_2D=is_2D,save_memory=save_memory,concat_not_add=concat_not_add)
        self.striding_not_pooling = striding_not_pooling
        self.out_classes = out_classes
        self.mask_output = mask_output
        self.get_unet(layers_dict)

    def get_unet(self, layers_dict):
        if self.complete_input is None:
            if self.is_2D:
                image_input_primary = x = Input(shape=(self.image_size, self.image_size, self.input_size), name='UNet_Input')
            else:
                image_input_primary = x = Input(shape=(self.z_images, self.image_size, self.image_size, self.input_size), name='UNet_Input')
        else:
            image_input_primary = x = self.complete_input
        x = self.run_unet(x)
        if self.mask_loss or self.mask_output:
            self.mask = Input(shape=(None,None,None,self.out_classes),name='mask')
            self.sum_vals = Input(shape=(None, None, None, self.out_classes), name='sum_vals')
            inputs = [image_input_primary,self.mask,self.sum_vals]
            if self.mask_loss:
                assert self.custom_loss is not None, 'Need to specify a custom loss when using masked input'
                partial_func = partial(self.custom_loss, mask=self.mask)
                self.custom_loss = update_wrapper(partial_func, self.custom_loss)
            if self.mask_output:
                # KeyError('Do not use mask_output, does not seem to work')
                x = Multiply()([self.mask,x])
                x = Add()([self.sum_vals,x])
        else:
            inputs = image_input_primary
        if self.create_model:
            model = Model(inputs=inputs, outputs=x)
            self.created_model = model
        else:
            x = BilinearUpsampling(output_size=(512,512), name='BMA_Upsampling')(x)
            x = Activation('softmax',name='Output_Activation')(x)
            model = Model(inputs=inputs, outputs=x)
            self.created_model = model
        self.output = x


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class identify_potential_slices(Unet):

    def __init__(self, start_block=32, num_layers=2, filters=(3,3,3), reduce_size=0):
        super().__init__()
        self.pool_size = (1,2,2)
        self.start_block = start_block
        self.num_layers = num_layers
        self.filters = filters
        self.activation = 'elu'
        self.reduce_size = reduce_size
        self.get_unet()

    def get_unet(self):
        image_input = x = Input(shape=(None, 512, 512, 1), name='UNet_Input')
        self.layer = 0
        total_reduction = 9
        for i in range(self.reduce_size):
            total_reduction -= 1
            x = AveragePooling3D(self.pool_size)(x)
        block = self.start_block
        layer_vals = {}
        self.desc = 'Encoder'
        for i in range(total_reduction):
            x = layer_vals[i] = self.conv_block(int(block), x, name=self.desc + str(self.layer))
            x = MaxPooling3D(pool_size=self.pool_size,name='max_pool' + str(i) + '_UNet')(x)
            self.layer += 1
        x = Conv3D(2, (1,1,1), activation=None, padding='same',name='output')(x) #output_Unet, new_output
        x = Activation('softmax',name='Output_Activation')(x)
        model = Model(inputs=image_input, outputs=x)
        self.created_model = model

    def get_unet_old(self):
        image_input = x = Input(shape=(None, None, None, 1), name='UNet_Input')
        self.layer = 0
        block = self.start_block
        self.desc = 'size_reduction'
        self.filters = (3,3,3)
        for i in range(4):
            x = MaxPooling3D(pool_size=(1, 2, 2), name='max_pool_in' + str(i))(x)
        filter_vals = {}
        layer_vals = {}
        self.desc = 'Encoder'
        for i in range(8):
            x = layer_vals[i] = self.conv_block(int(block), x, name=self.desc + str(self.layer))
            x = MaxPooling3D(pool_size=self.pool_size,name='max_pool' + str(i) + '_UNet')(x)
            filter_vals[i] = block
            self.layer += 1
        self.desc = 'Features'
        x = self.conv_block(int(block),x, name=self.desc + str(self.layer))
        self.desc = 'Decoder'
        self.layer = 0
        for i in range(self.num_layers - 1, -1, -1):
            x = UpSampling3D(size=(2,1,1), name='Upsampling' + str(i) + '_UNet')(x)
            block = filter_vals[i] # Mirror one side to the next
            x = Concatenate(name='concat' + str(i) + '_Unet')([x, layer_vals[i]])
            x = self.conv_block(int(block), x, name=self.desc + str(self.layer))
            self.layer += 1
        for i in range(4):
            x = MaxPooling3D(pool_size=(1, 2, 2), name='max_pool_out' + str(i))(x)
        x = Conv3D(2, (1,1,1), activation=None, padding='same',name='output')(x) #output_Unet, new_output
        x = Activation('softmax',name='Output_Activation')(x)
        model = Model(inputs=image_input, outputs=x)
        self.created_model = model


class identify_potential_slices_unet(Unet):

    def __init__(self, start_block=32, num_layers=2, filters=(3,3,3), reduce_size=0):
        self.pool_size = (1,2,2)
        self.start_block = start_block
        self.num_layers = num_layers
        self.filters = filters
        self.activation = 'elu'
        self.reduce_size = reduce_size
        self.get_unet()

    def get_unet(self):
        image_input = x = Input(shape=(None, 512, 512, 1), name='UNet_Input')
        self.layer = 0
        total_reduction = 9
        for i in range(self.reduce_size):
            total_reduction -= 1
            x = AveragePooling3D(self.pool_size)(x)
        block = self.start_block
        layer_vals = {}
        self.desc = 'Encoder'
        filter_vals = {}
        for i in range(self.num_layers):
            x = layer_vals[i] = self.conv_block(int(block), x, name=self.desc + str(self.layer))
            x = MaxPooling3D(pool_size=(2,2,2),name='max_pool' + self.desc + str(i) + '_UNet')(x)
            filter_vals[i] = block
            block *= 2
            self.layer += 1
        self.desc = 'Features'
        x = self.conv_block(int(block),x, name = 'Base')
        self.desc = 'Decoder'
        self.layer = 0
        for i in range(self.num_layers - 1, -1, -1):
            x = UpSampling3D(size=(2,2,2), name='Upsampling' + str(i) + '_UNet')(x)
            block = filter_vals[i] # Mirror one side to the next
            x = Concatenate(name='concat' + str(i) + '_Unet')([x, layer_vals[i]])
            x = self.conv_block(int(block), x, name=self.desc + str(self.layer))
            self.layer += 1
        self.desc = 'Final'
        for i in range(total_reduction):
            x = self.conv_block(int(block), x, name=self.desc + str(self.layer))
            x = MaxPooling3D(pool_size=self.pool_size,name='max_pool' + self.desc + str(i) + '_UNet')(x)
            self.layer += 1
        x = Conv3D(2, (1,1,1), activation=None, padding='same',name='output')(x) #output_Unet, new_output
        x = Activation('softmax',name='Output_Activation')(x)
        model = Model(inputs=image_input, outputs=x)
        self.created_model = model


class my_3D_UNet_total_skip_modular(object):

    def __init__(self, image_size=256, image_num=16, start_block=32, channels=32, filter_vals=None,num_of_classes=2,
                 num_layers=2, expand_after_concat=False, use_vgg=False, max_filters=999):
        self.max_filters = max_filters
        if filter_vals is None:
            filter_vals = [3,3,3]
        self.pool_size = (2,2,2)
        self.use_vgg = use_vgg
        self.expand_after_concat = expand_after_concat
        self.filter_vals = filter_vals
        self.channels = channels
        self.image_size = image_size
        self.num_of_classes = num_of_classes
        self.start_block = start_block
        self.num_layers = num_layers
        self.image_num = image_num
        self.activation = 'elu'
        self.get_unetx2D_short()

    def conv_block(self,output_size,x, drop=0.0):
        for i in range(2):
            x = Conv3D(output_size, self.filters, activation=None, padding='same',
                       name='conv' + self.desc + str(self.layer) + '_' + str(i) +'UNet')(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation, name=self.desc+'_activation')(x)
            if drop != 0.0:
                x = SpatialDropout3D(drop)(x)
        return x

    def up_sample_out(self,x, drop):
        pool_size = int(self.image_num/int(x.shape[1]))
        x = UpSampling3D(size=(pool_size,pool_size,pool_size))(x)
        filters = int(self.start_block*2)
        for i in range(2):
            x = Conv3D(int(filters), self.filters, activation=None, padding='same',
                       name='conv' + str(self.layer) + '_' + str(i) +'UNet')(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation, name=self.layer + '_' + str(i) + '_activation')(x)
            if drop != 0.0:
                # x = Dropout(drop)(x)
                x = SpatialDropout3D(drop)(x)
            filters /= 2
        self.layer += 1
        return x

    def residual_block(self,output_size,x, short_cut=False, prefix='', is_first_res_block=1):
        print(is_first_res_block)
        if is_first_res_block != 0:
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
        x = Conv3D(int(output_size), activation=None, padding='same',
                   name='Conv_start_' + self.desc + str(self.layer), strides=[1, 1, 1], kernel_size=(3, 3, 3))(x)
        inputs = x
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv3D(output_size, activation=None, padding='same',
                   name='Conv_' + self.desc + str(self.layer) + '_0_UNet' + prefix, strides=[1,1,1],kernel_size=(3,3,3))(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv3D(output_size, activation=None, padding='same',
                   name='Conv_' + self.desc + str(self.layer) + '_1_UNet' + prefix, strides=[1,1,1],kernel_size=(3,3,3))(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv3D(output_size, kernel_size=(1, 1, 1), strides=(1,1,1), padding='same',
                           name='Conv_' + self.desc + str(self.layer) + '_2_UNet' + prefix)(x)
        if short_cut:
            x = Add(name='Conv_' + self.desc + str(self.layer) + '_3_short_cut_UNet' + prefix)([x,inputs])
        return x

    def get_unetx2D_short(self):
        image_inputs = x = Input(shape=(None, None, None, 2), name='UNet_Input_image')
        # inputs = K.expand_dims(inputs,axis=1)
        self.filters = (self.filter_vals[0], self.filter_vals[1], self.filter_vals[2])
        self.layer = 0
        block = self.start_block
        layer_vals = {}
        filter_vals = {}
        self.desc = 'Encoder'
        for i in range(self.num_layers):
            x = layer_vals[i] = self.residual_block(int(block),x, short_cut=True, is_first_res_block=i)
            x = MaxPooling3D(pool_size=self.pool_size,name='max_pool' + str(i) + '_UNet')(x)
            filter_vals[i] = block
            block = 2*block if block*2 < self.max_filters else block
            self.layer += 1

        x = self.residual_block(int(block),x,short_cut=False)
        self.desc = 'Decoder'
        self.layer = 0
        for i in range(self.num_layers - 1, -1, -1):
            x = self.residual_block(int(block),x, short_cut=True)
            x = UpSampling3D(size=(2,2,2), name='Upsampling' + str(i) + '_UNet')(x)
            x = Concatenate(name='concat' + str(i) + '_Unet')([x, layer_vals[i]])
            block = filter_vals[i] # Mirror one side to the next
            x = Conv3D(block, self.filter_vals, activation=None, padding='same', name='reduce_' + str(i))(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)

            # block /= 2
            self.layer += 1

        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv3D(self.num_of_classes, (1,1,1), activation=None, padding='same',name='output_Unet')(x)
        x = Activation('softmax')(x)
        model = Model(inputs=image_inputs, outputs=x)
        self.created_model = model


class my_3D_UNet_total_skip(object):

    def __init__(self, image_size=256, batch_size=32, start_block=64, channels=3, filter_vals=None,num_of_classes=2,num_layers=4):
        if filter_vals is None:
            filter_vals = [3,3,3]
        self.pool_size = (2,2,2)
        self.filter_vals = filter_vals
        self.channels = channels
        self.image_size = image_size
        self.num_of_classes = num_of_classes
        self.start_block = start_block
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.activation = 'elu'
        self.get_unetx2D_short()

    def conv_single(self, output_size, x, drop=0.0):
        x = Conv3D(output_size, self.filters, activation=None, padding='same',
                   name='conv_single' + str(self.layer) + '_' + 'UNet')(x)
        # x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        if drop != 0.0:
            # x = Dropout(drop)(x)
            x = SpatialDropout3D(drop)(x)

        return x
    def conv_block(self,output_size,x, drop=0.0):
        for i in range(2):
            x = Conv3D(output_size, self.filters, activation=None, padding='same',
                       name='conv' + str(self.layer) + '_' + str(i) +'UNet')(x)
            # x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            if drop != 0.0:
                # x = Dropout(drop)(x)
                x = SpatialDropout3D(drop)(x)

        self.layer += 1
        return x

    def up_sample_out(self,x, drop):
        pool_size = int(self.image_size/int(x.shape[2]))
        x = UpSampling3D(size=(pool_size,pool_size,pool_size))(x)
        i = -1
        while int(x.shape[-1]) > self.start_block:
            i += 1
            x = Conv3D(int(int(x.shape[-1])/2), self.filters, activation=None, padding='same',
                       name='conv' + str(self.layer) + '_' + str(i) +'UNet')(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            if drop != 0.0:
                # x = Dropout(drop)(x)
                x = SpatialDropout3D(drop)(x)
        self.layer += 1
        return x

    def get_unetx2D_short(self):
        x = inputs = Input((self.batch_size, self.image_size, self.image_size, self.channels))
        # inputs = K.expand_dims(inputs,axis=1)
        self.filters = (self.filter_vals[0], self.filter_vals[1], self.filter_vals[2])
        self.layer = 1
        block = self.start_block
        layer_vals = {}
        output_net = {}
        drop_out = 0.1
        for i in range(self.num_layers):
            x = layer_vals[i] = self.conv_block(int(block),x, drop=drop_out)
            x = MaxPooling3D(pool_size=self.pool_size,name='max_pool' + str(i) + '_UNet')(x)
            block *= 2
            drop_out = 0.2

        x = self.conv_block(int(block), x, drop=drop_out)
        for i in range(self.num_layers - 1, -1, -1):
            block /= 2
            output_net[i] = self.up_sample_out(x, drop_out)
            x = UpSampling3D(size=(2,2,2), name='conv_transpose' + str(i) + '_UNet')(x)
            x = self.conv_single(int(block),x,drop=drop_out)
            x = Concatenate(name='concat' + str(i) + '_Unet')([x, layer_vals[i]])
            x = self.conv_block(int(block), x, drop=drop_out)
        keys = list(output_net.keys())
        for key in keys:
            x = Concatenate()([x,output_net[key]])
        x = Conv3D(self.num_of_classes, self.filters, activation=None, padding='same',name='output_Unet')(x)
        x = Activation('softmax')(x)
        # x = Flatten()(x)

        self.model = Model(inputs=[inputs], outputs=[x],name='3D_Pyramidal_UNet')


class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return tf.compat.v1.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return tf.compat.v1.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BilinearUpsampling3D(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling3D, self).__init__(**kwargs)

        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=5)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * input_shape[2] if input_shape[2] is not None else None
            depth = self.upsampling[2] * input_shape[3] if input_shape[3] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
            depth = self.output_size[2]
        return (input_shape[0],
                height,
                width,
                depth,
                input_shape[4])

    def call(self, inputs):
        if self.upsampling:
            return tf.compat.v1.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                               inputs.shape[2] * self.upsampling[1],
                                                               inputs.shape[3] * self.upsampling[2]),
                                                      align_corners=True)
        else:
            return tf.compat.v1.image.resize_bilinear(inputs, (self.output_size[0],self.output_size[1],
                                                               self.output_size[2]),
                                                      align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class my_3D_UNet_dvf(base_UNet):

    def __init__(self, filter_vals=(3,3,3),layers_dict=None, pool_size=(2,2,2), activation='elu',pool_type='Max',
                 batch_norm=False, striding_not_pooling=False, do_res_block=False,distance_map=False):
        super().__init__(filter_vals=filter_vals, layers_dict=layers_dict, pool_size=pool_size, activation=activation,
                         pool_type=pool_type, batch_norm=batch_norm)
        if distance_map:
            self.input_channels = 3
        else:
            self.input_channels = 1
        self.striding_not_pooling = striding_not_pooling
        self.define_res_block(do_res_block)
        self.get_unet(layers_dict)

    def get_unet(self, layers_dict):
        image_input_primary = Input(shape=(None, None, None, self.input_channels), name='UNet_Input_image_primary')
        image_input_secondary = Input(shape=(None, None, None, self.input_channels), name='UNet_Input_image_secondary')
        x = Concatenate(name='Input_concat')([image_input_primary, image_input_secondary])
        x = self.run_unet(x)
        # flow = Conv3D(3, kernel_size=self.filters, padding='same', name='flow',
        #            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
        flow = Conv3D(3, kernel_size=self.filters, padding='same', name='flow')(x)
        # flow = Multiply()([flow, image_input_primary])  # Makes a mask around our field for the primary image
        # y = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([image_input_primary, flow])
        model = Model(inputs=[image_input_primary,image_input_secondary], outputs=flow)
        self.created_model = model


class my_3D_UNet_dvf_w_images(base_UNet):

    def __init__(self, filter_vals=(3,3,3),layers_dict=None, pool_size=(2,2,2), activation='elu',pool_type='Max',
                 batch_norm=False, striding_not_pooling=False, do_res_block=False,image_size=(None,None,None,1)):
        super().__init__(filter_vals=filter_vals, layers_dict=layers_dict, pool_size=pool_size, activation=activation,
                         pool_type=pool_type, batch_norm=batch_norm)
        self.striding_not_pooling = striding_not_pooling
        self.image_size = image_size
        self.define_res_block(do_res_block)
        self.get_unet(layers_dict)

    def get_unet(self, layers_dict):
        image_input_primary = Input(shape=self.image_size, name='UNet_Input_image_primary')
        image_input_secondary = Input(shape=self.image_size, name='UNet_Input_image_secondary')
        weights_tensor = Input(shape=self.image_size,name='weights')
        x = Concatenate(name='Input_concat')([image_input_primary, image_input_secondary])
        x = self.get_unet(x)
        flow = Conv3D(3, kernel_size=self.filters, padding='same', name='flow',
                      kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
        # kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5)
        # y = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([image_input_primary, flow])
        # model = Model(inputs=[image_input_primary,image_input_secondary,weights_tensor], outputs=y)
        # self.created_model = model
        # partial(weighted_mse, weights=weights_tensor)
        # self.custom_loss = wrapped_partial(weighted_mse,weights=weights_tensor))
        # partial_func = partial(weighted_mse, weights=weights_tensor)
        # self.custom_loss = update_wrapper(partial_func, weighted_mse)


class my_3D_UNet_dvf_multi_pooling(base_UNet):

    def __init__(self, filter_vals=(3,3,3),layers_dict=None, pool_size=(2,2,2), do_res_block = True, activation='elu',
                 pool_type='Max',batch_norm=False):
        super().__init__(filter_vals=filter_vals, layers_dict=layers_dict, pool_size=pool_size, activation=activation,
                         pool_type=pool_type, batch_norm=batch_norm)
        self.define_res_block(do_res_block)
        self.get_unet(layers_dict)

    def get_unet(self, layers_dict):
        image_input_primary = Input(shape=(None, None, None, 1), name='UNet_Input_image_primary')
        image_input_secondary = Input(shape=(None, None, None, 1), name='UNet_Input_image_secondary')
        x_input = Concatenate(name='Input_concat')([image_input_primary, image_input_secondary])
        self.layer = 0
        layer_vals = {}
        self.desc = 'Encoder'
        layer_index = 0
        for layer in self.layers_names:
            print(layer)
            if layer == 'Base':
                continue
            all_filters = layers_dict[layer]['Encoding']
            if 'Pooling' in layers_dict[layer]:
                self.define_pool_size(layers_dict[layer]['Pooling'])
            x = self.pooling_down_block(x, layer + '_Pooling')
            for i in range(len(all_filters)):
                strides = 1
                self.desc = layer + '_Encoding_Conv' + str(i) if strides == 1 else layer + '_Strided_Conv' + str(i)
                if self.do_res_block and layer_index != 0:
                    x = self.residual_block(all_filters[i],x=x,name=self.desc)
                else:
                    x = self.conv_block(all_filters[i], x=x, strides=strides, name=self.desc)
            x = UpSampling3D(size=self.pool_size)(x)
            layer_vals[layer] = x
            self.layer += 1
        for layer in self.layers_names[:-1]:
            x = Concatenate(name='concat' + str(layer) + '_Unet')([x, layer_vals[layer]])
        all_filters = layers_dict['Layer_0']['Decoding']
        for i in range(len(all_filters)):
            self.desc = 'Final_Decoding_Conv' + str(i)
            x = self.conv_block(all_filters[i], x, self.desc)
            # x = self.residual_block(all_filters[i], x=x, name=self.desc)
        self.layer += 1

        x = Conv3D(3, kernel_size=self.filters, padding='same', name='flow',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
        x = Multiply()([x, image_input_primary])  # Makes a mask around our field for the primary image
        model = Model(inputs=[image_input_primary,image_input_secondary], outputs=x)
        self.created_model = model


class my_3D_UNet_dvf_split(base_UNet):

    def __init__(self, filter_vals=(3,3,3),layers_dict=None, pool_size=(2,2,2), activation='elu', pool_type='Max',batch_norm=False):
        super().__init__(filter_vals=filter_vals, layers_dict=layers_dict, pool_size=pool_size, activation=activation,
                         pool_type=pool_type, batch_norm=batch_norm)
        self.get_unet(layers_dict)

    def get_unet(self, layers_dict):
        image_input_primary = x_primary = Input(shape=(None, None, None, 1), name='UNet_Input_image_primary')
        image_input_secondary = x_secondary = Input(shape=(None, None, None, 1), name='UNet_Input_image_secondary')
        self.layer = 0
        layer_vals = {}
        self.desc = 'Encoder'
        layer_index = 0
        layer_order = []
        for layer in self.layers_names:
            print(layer)
            if layer == 'Base':
                continue
            layer_order.append(layer)
            all_filters = layers_dict[layer]['Encoding']
            for i in range(len(all_filters)):
                strides = 1
                self.desc = layer + '_Encoding_Conv' + str(i) if strides == 1 else layer + '_Strided_Conv' + str(i)
                x_primary, x_secondary = self.shared_conv_block(x_primary,x_secondary,all_filters[i],strides=strides,name=self.desc)
                x = Concatenate(name='concat_Encode_' + layer)([x_primary, x_secondary])
                layer_vals[layer_index] = x
            if 'Pooling' in layers_dict[layer]:
                self.define_pool_size(layers_dict[layer]['Pooling'])
            x_primary, x_secondary = self.pooling_down_block(x_primary,layer + '_Primary_Pooling'), self.pooling_down_block(x_secondary,layer + '_Secondary_Pooling')
            layer_index += 1
        if 'Base' in layers_dict:
            strides = 1
            all_filters = layers_dict['Base']['Encoding']
            for i in range(len(all_filters)):
                self.desc = 'Base_Conv' + str(i)
                x_primary, x_secondary = self.shared_conv_block(x_primary, x_secondary, all_filters[i], strides=strides,
                                                                name=self.desc)
        x = Concatenate(name='concat_Encode_Base')([x_primary, x_secondary])
        self.desc = 'Decoder'
        self.layer = 0
        layer_order.reverse()
        for layer in layer_order:
            print(layer)
            layer_index -= 1
            all_filters = layers_dict[layer]['Decoding']
            if 'Pooling' in layers_dict[layer]:
                self.define_pool_size(layers_dict[layer]['Pooling'])
            x = UpSampling3D(size=self.pool_size, name='Upsampling' + str(self.layer) + '_UNet')(x)
            # x = Conv3DTranspose(all_filters[0],self.filters,strides=self.pool_size,padding=self.padding,name=layer + 'Strided_Conv')(x)
            # x = Conv3DTranspose(all_filters[0],kernel_size=self.filters,strides=2,name='UpConv' + str(self.layer) + '_UNet',padding='same')(x)
            x = Concatenate(name='concat' + str(self.layer) + '_Unet')([x, layer_vals[layer_index]])
            for i in range(len(all_filters)):
                self.desc = layer + '_Decoding_Conv' + str(i)
                x = self.conv_block(all_filters[i], x, self.desc)
                # x = self.residual_block(all_filters[i], x=x, name=self.desc)
            self.layer += 1

        x = Conv3D(3, kernel_size=self.filters, padding='same', name='flow',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
        x = Multiply()([x, image_input_primary])  # Makes a mask around our field for the primary image
        model = Model(inputs=[image_input_primary,image_input_secondary], outputs=x)
        self.created_model = model


class my_3D_UNet_dvf_Inception(object):

    def __init__(self, image_size=256, image_num=16, start_block=32, channels=32, filter_vals=None,flip=False,
                 num_layers=2, expand_after_concat=False, use_vgg=False, max_filters=999, min_filters=16):
        self.max_filters = max_filters
        self.min_filters = min_filters
        self.flip = flip
        if filter_vals is None:
            filter_vals = [3,3,3]
        self.pool_size = (2,2,2)
        self.use_vgg = use_vgg
        self.expand_after_concat = expand_after_concat
        self.filter_vals = filter_vals
        self.channels = channels
        self.image_size = image_size
        self.start_block = start_block
        self.num_layers = num_layers
        self.image_num = image_num
        self.activation = 'elu'
        self.get_unetx2D_short()

    def inception_esc_block(self,output_size,x, drop=0.0):
        input_val = x
        for i in range(2):
            x = Conv3D(output_size, self.filters, activation=None, padding='same',
                       name=self.desc + '_conv' + str(self.layer) + '_' + str(i) +'UNet')(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            if drop != 0.0:
                # x = Dropout(drop)(x)
                x = SpatialDropout3D(drop)(x)
        return x

    def residual_block(self,output_size,x, short_cut=False, prefix='', is_first_res_block=1):
        print(is_first_res_block)
        if is_first_res_block != 0:
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
        x = Conv3D(int(output_size), activation=None, padding='same',
                   name='Conv_start_' + self.desc + str(self.layer), strides=[1, 1, 1], kernel_size=(3, 3, 3))(x)
        inputs = x
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv3D(output_size, activation=None, padding='same',
                   name='Conv_' + self.desc + str(self.layer) + '_0_UNet' + prefix, strides=[1,1,1],kernel_size=(3,3,3))(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv3D(output_size, activation=None, padding='same',
                   name='Conv_' + self.desc + str(self.layer) + '_1_UNet' + prefix, strides=[1,1,1],kernel_size=(3,3,3))(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv3D(output_size, kernel_size=(1, 1, 1), strides=(1,1,1), padding='same',
                           name='Conv_' + self.desc + str(self.layer) + '_2_UNet' + prefix)(x)
        if short_cut:
            x = Add(name='Conv_' + self.desc + str(self.layer) + '_3_short_cut_UNet' + prefix)([x,inputs])
        return x

    def get_unetx2D_short(self):
        image_inputs = x = Input(shape=(None, None, None, 2), name='UNet_Input_image')
        # inputs = K.expand_dims(inputs,axis=1)
        self.filters = (self.filter_vals[0], self.filter_vals[1], self.filter_vals[2])
        self.layer = 0
        block = self.start_block
        layer_vals = {}
        filter_vals = {}
        self.desc = 'Encoder'
        for i in range(self.num_layers):
            # x = layer_vals[i] = self.residual_block(int(block),x, short_cut=True, is_first_res_block=i)
            x = layer_vals[i] = self.conv_block(int(block), x)
            x = MaxPooling3D(pool_size=self.pool_size,name='max_pool' + str(i) + '_UNet')(x)
            filter_vals[i] = block
            if not self.flip:
                block = 2*block if block*2 < self.max_filters else block
            else:
                block = block / 2 if block / 2 > self.min_filters else block
            self.layer += 1

        x = self.conv_block(int(block),x)
        self.desc = 'Decoder'
        self.layer = 0
        for i in range(self.num_layers - 1, -1, -1):
            # x = self.residual_block(int(block),x, short_cut=True)
            x = UpSampling3D(size=(2,2,2), name='Upsampling' + str(i) + '_UNet')(x)
            # x = Deconv3D(filters=(block), kernel_size=(2,2,2),output_padding='same')(x)
            # if layer_vals[i].shape.as_list()[1] != x.shape.as_list()[1]:
            #     x = pad_depth(x,layer_vals[i].shape.as_list()[1])
            x = Concatenate(name='concat' + str(i) + '_Unet')([x, layer_vals[i]])
            x = Conv3D(int(block), kernel_size=self.filters, padding='same',
                       name='Conv_concat' + self.desc + str(self.layer) + '_UNet')(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            block = filter_vals[i] # Mirror one side to the next
            # x = Conv3D(block, self.filter_vals, activation=None, padding='same', name='reduce_' + str(i))(x)
            # x = BatchNormalization()(x)
            # x = Activation(self.activation)(x)

            # block /= 2
            x = self.conv_block(int(block), x)
            self.layer += 1

        x = Conv3D(64, (3, 3, 3), activation=None, padding='same', name='before_output_0')(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv3D(32, (3, 3, 3), activation=None, padding='same', name='before_output_1')(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv3D(3, (1,1,1), activation=None, padding='same',name='output_Unet')(x)
        x = BatchNormalization()(x)
        x = Activation('linear')(x)
        model = Model(inputs=image_inputs, outputs=x)
        self.created_model = model


class my_3D_UNet_Auto_Encoder(Unet):

    def __init__(self, start_block=32, channels=32, filter_vals=(3,3,3),out_classes=1,retrain=False,num_layers=2):
        self.retrain = retrain
        self.out_classes = out_classes
        self.filters = filter_vals
        self.pool_size = (2,2,2)
        self.filter_vals = filter_vals
        self.channels = channels
        self.start_block = start_block
        self.num_layers = num_layers
        self.activation = 'elu'
        self.define_padding('same')
        self.get_unet()

    def get_unet(self):
        image_input = x = Input(shape=(None, None, None, 1), name='UNet_Input')
        self.layer = 0
        block = self.start_block
        network_values = {}
        filter_vals = {}
        layer_vals = {}
        self.desc = 'Encoder'
        for i in range(self.num_layers):
            x = layer_vals[i] = self.conv_block(int(block), x, name=self.desc + '_' + str(self.layer))
            network_values[self.desc + str(self.layer)] = x
            x = AveragePooling3D(pool_size=self.pool_size,name='max_pool' + str(i) + '_UNet')(x)
            filter_vals[i] = block
            self.layer += 1
        self.desc = 'Features'
        self.layer = 0
        x = self.conv_block(int(block),x, name=self.desc + '_' + str(self.layer))
        self.desc = 'Decoder'
        for i in range(self.num_layers - 1, -1, -1):
            x = UpSampling3D(size=(2,2,2), name='Upsampling' + str(i) + '_UNet')(x)
            block = filter_vals[i] # Mirror one side to the next
            x = Concatenate(name='concat' + str(i) + '_Unet')([x, layer_vals[i]])
            x = self.conv_block(int(block), x,  name=self.desc + '_' + str(self.layer))
            network_values[self.desc + str(self.layer)] = x
            self.layer += 1
        output_name = 'output_Unet'
        x = Conv3D(8, (3,3,3), activation='softmax', padding='same',name='soft_max_layer')(x)
        # summary = tf.Summary(value=[tf.Summary.Value(tag='soft_max_layer', image=x)])
        network_values['soft_max_layer'] = x
        x = Conv3D(self.out_classes, (3,3,3), activation='linear', padding='same',name=output_name)(x) #output_Unet, new_output
        network_values['output_Unet'] = x
        model = Model(inputs=image_input, outputs=x)
        self.created_model = model


class my_3D_UNet_Auto_Encoder_Inception(Unet):

    def __init__(self,input_size=(32,32,32), mask_size=(10,10,10),drop_out=0.0, sparse=False, pool_size=(2,2,2), filter_vals=(3,3,3),layers_dict_conv={},
                 layers_dict_FC={},output_size=(10,10,10),concatentate=True,activation='elu'):
        self.mask_size = mask_size
        self.output_size = output_size
        self.layers_dict_conv = layers_dict_conv
        self.layers_names_conv = [i for i in layers_dict_conv]
        self.layers_names_conv.sort()
        if self.layers_names_conv:
            self.layers_names_conv = self.layers_names_conv[1:] + [self.layers_names_conv[0]]

        self.concatentate = concatentate
        self.sparse = sparse
        self.drop_out = drop_out
        self.input_size = input_size
        self.define_pool_size(pool_size)
        self.define_batch_norm(True)
        self.define_filters(filter_vals)
        self.define_activation(activation)
        self.define_padding('same')
        self.get_unet()

    def do_conv_block_dec(self,x):
        self.desc = 'Decoder'
        self.layer = 0
        self.layer_names.reverse()
        previous_layers = ['Base']
        for layer in self.layer_names:
            print(layer)
            all_filters = self.layers_dict_conv[layer]['Decoding']
            x = UpSampling3D(size=self.pool_size, name='Upsampling' + str(self.layer) + '_UNet')(x)
            # x = Conv3DTranspose(all_filters[0],kernel_size=self.filters,strides=2,name='UpConv' + str(self.layer) + '_UNet',padding='same')(x)
            if self.concatentate:
                cropped_previous = self.layer_vals[layer]
                if self.padding == 'valid':
                    cropped = np.sum([len(self.layers_dict_conv[xxx]['Encoding']) * 2 for xxx in previous_layers])
                    cropped = int(cropped + np.sum(
                        [len(self.layers_dict_conv[xxx]['Decoding']) * 2 for xxx in previous_layers[1:]]) + 2 * len(
                        previous_layers[1:]))
                    previous_layers.append(layer)
                    cropped_previous = Cropping3D(cropping=((cropped,cropped),(cropped,cropped),(cropped,cropped)))(cropped_previous)
                x = Concatenate(name='concat' + str(self.layer) + '_Unet')([x, cropped_previous])
            for i in range(len(all_filters)):
                self.desc = layer + '_Decoding_Conv' + str(i)
                x = self.conv_block(all_filters[i], x, self.desc)
            self.layer += 1
        return x

    def get_unet(self):
        image_input = Input(shape=self.input_size, name='UNet_Input') # 16384
        mask_input = Input(shape=self.mask_size, name='mask')
        self.layer_vals = {}
        self.layer = 0
        self.desc = 'Encoder'
        self.layer_names = []
        for xxx, layer in enumerate(self.layers_names_conv):
            print(layer)
            self.layer_names.append(layer)
            all_filters = self.layers_dict_conv[layer]['Encoding']
            filter_vals = None
            if 'Kernel' in all_filters:
                filter_vals = all_filters['Kernel']
            if 'Channels' in all_filters:
                all_filters = all_filters['Channels']
            for i in range(len(all_filters)):
                if filter_vals:
                    self.define_filters(filter_vals[i])
                print(i)
                strides = 1
                self.desc = layer + '_Encoding_Conv' + str(i)
                x = self.conv_block(all_filters[i], x=image_input, strides=strides, name=self.desc)
                if i == 0:
                    out = x
                else:
                    out = Concatenate(name='concat' + str(self.layer) + '_Unet')([out, x])

        self.desc = 'Decoder'
        self.layer_names = []
        layer_order = []
        for xxx, layer in enumerate(self.layers_names_conv):
            layer_order.append(layer)
            print(layer)
            self.layer_names.append(layer)
            all_filters = self.layers_dict_conv[layer]['Encoding']
            filter_vals = None
            if 'Channels' in all_filters:
                filter_vals = all_filters['Kernel']
                all_filters = all_filters['Channels']
            for i in range(len(all_filters)):
                if filter_vals:
                    self.define_filters(filter_vals[i])
                print(i)
                strides = 1
                self.desc = layer + '_Encoding_Conv' + str(i)
                x = self.conv_block(all_filters[i], x=image_input, strides=strides, name=self.desc)
                if i == 0:
                    out = x
                else:
                    out = Concatenate(name='concat' + str(self.layer) + '_' + str(i) + '_Unet')([out, x])
        layer_order.reverse()
        self.define_filters((3,3,3))
        x = out
        for layer in layer_order:
            print(layer)
            all_filters = self.layers_dict_conv[layer]['Decoding']
            for i in range(len(all_filters)):
                self.desc = layer + '_Decoding_Conv' + str(i)
                x = self.conv_block(all_filters[i], x, self.desc)
            self.layer += 1

        self.define_filters((1,1,1))
        self.define_activation('linear')
        self.define_batch_norm(False)
        x = self.conv_block(1, x, name='output')
        # self.define_filters((1,1,1))
        # x = self.conv_block(1,x,name='Output')
        x = Multiply()([x, mask_input])
        model = Model(inputs=[image_input,mask_input], outputs=x)
        self.created_model = model


def sparse_reg(activ_matrix):
    rho = 0.01 # desired average activation of the hidden units
    beta = 3 # weight of sparsity penalty term
    #return 1000000*K.shape(activ_matrix)[0] # usefull to DEBUG
    # axis 0 size is batch_size
    # axis 1 size is layer_size
    rho_bar = K.mean(activ_matrix, axis=0) # average over the batch samples
    KLs = rho*K.log(rho/rho_bar) + (1-rho)*K.log((1-rho)/(1-rho_bar))
    return beta * K.sum(KLs) # sum over the layer units


class my_3D_UNet_Auto_Encoder_FC(Unet):

    def __init__(self,input_size=(32,32,32), mask_size=(10,10,10),drop_out=0.0, sparse=False, pool_size=(2,2,2), filter_vals=(3,3,3),layers_dict_conv={},
                 layers_dict_FC={},save_memory=False,output_size=(10,10,10),concatentate=True,activation='elu',batch_norm=False, output_classes=1, output_activation=None):
        self.save_memory = save_memory
        self.mask_size = mask_size
        self.output_size = output_size
        self.layers_dict_FC = layers_dict_FC
        self.layers_dict_conv = layers_dict_conv
        self.define_unet_dict(layers_dict_conv)

        self.define_fc_dict(layers_dict_FC)

        self.concatentate = concatentate
        self.sparse = sparse
        self.drop_out = drop_out
        self.input_size = input_size
        self.out_classes = output_classes
        self.output_activation = output_activation
        self.define_pool_size(pool_size)
        self.define_res_block(False)
        self.define_batch_norm(batch_norm)
        self.define_filters(filter_vals)
        self.define_activation(activation)
        self.define_padding('same')
        self.define_pooling_type('Max')
        self.run_unet()

    def run_unet(self):
        image_input = x = Input(shape=self.input_size, name='UNet_Input') # 16384
        mask_input = Input(shape=self.mask_size, name='mask')
        x, layer_vals = self.do_conv_block_enc(x)
        if self.layers_dict_FC:
            reshape_size = tuple([i.value for i in x.shape[1:]])
            x = Flatten()(x)
            x = self.do_FC_block(x)
            x = Dense(np.prod(reshape_size), activation=self.activation,name='reshaped')(x)
            x = Reshape(reshape_size)(x)
        # else:
        x = self.do_conv_block_decode(x, layer_vals)
        x = self.conv(self.out_classes,(1,1,1),activation=self.output_activation,name='output')(x)
        x = Multiply()([x, mask_input])
        model = Model(inputs=[image_input,mask_input], outputs=x)
        self.created_model = model

class Pyramid_Pool_3D(object):

    def __init__(self, start_block=32, channels=1, filter_vals=None,num_of_classes=2,num_layers=2, resolutions=[64,128,128]):
        self.start_block = start_block
        self.resolutions = resolutions
        if filter_vals is None:
            filter_vals = [5,5,5]
        self.pool_size = (2,2,2)
        self.filter_vals = filter_vals
        self.channels = channels
        self.num_of_classes = num_of_classes
        self.num_layers = num_layers
        self.activation = 'relu'
        self.unet()

    def residual_block(self,output_size,x, drop=0.0, short_cut=False, prefix=''):
        inputs = x
        x = Conv3D(output_size, self.filters, activation=None, padding='same',
                   name=self.image_resolution + '_conv' + str(self.layer) + '_0_UNet' + prefix)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv3D(output_size, self.filters, activation=None, padding='same',
                   name=self.image_resolution + '_conv' + str(self.layer) + '_1_UNet' + prefix)(x)
        x = BatchNormalization()(x)
        x = SpatialDropout3D(drop)(x)

        x = Conv3D(output_size, kernel_size=(1, 1, 1), strides=(1,1,1), padding='same',
                           name=self.image_resolution + '_conv' + str(self.layer) + '_2_UNet' + prefix)(x)
        x = BatchNormalization()(x)
        if short_cut:
            x = Add(name=self.image_resolution + '_skip' + str(self.layer) + '_UNet' + prefix)([x,inputs])
        x = Activation(self.activation)(x)

        return x

    def unet(self):
        high_image = Input([self.resolutions[0], self.resolutions[1], self.resolutions[2], self.channels])
        medium_image = Input([self.resolutions[0], self.resolutions[1], self.resolutions[2], self.channels])
        low_image = Input([self.resolutions[0], self.resolutions[1], self.resolutions[2], self.channels])
        self.data_dict = {'high':{'Input':high_image},'medium':{'Input':medium_image},'low':{'Input':low_image}}
        self.filters = (self.filter_vals[0], self.filter_vals[1], self.filter_vals[2])
        self.layer = 1

        '''
        First, do the low resolution block
        '''
        self.image_resolution = 'low'
        x = self.data_dict[self.image_resolution]['Input']
        self.res_block(x)
        '''
        Then, concatenate results and do medium block
        '''
        self.image_resolution = 'medium'
        x = Concatenate()([self.data_dict['medium']['Input'],self.data_dict['low']['last_layer']])
        self.res_block(x)
        '''
        Lastly, concatenate results and do high block
        '''
        self.image_resolution = 'high'
        x = Concatenate()([self.data_dict['high']['Input'], self.data_dict['medium']['last_layer']])
        self.res_block(x)
        self.model = Model(inputs=[self.data_dict['high']['Input'], self.data_dict['medium']['Input'],
                                   self.data_dict['low']['Input']],
                           outputs=[self.data_dict['high']['last_layer'],self.data_dict['medium']['last_layer'],
                                    self.data_dict['low']['last_layer']],
                           name='Multi_Scale_BMA')

    def res_block(self, x):
        drop_out = 0.0
        for self.layer in range(self.num_layers):
            short_cut = False
            self.start_block = int(self.start_block * 2)
            x = self.residual_block(int(self.start_block), x, drop=drop_out, short_cut=short_cut, prefix='_Down_0')
            drop_out = 0.2
            short_cut = True
            x = self.residual_block(int(self.start_block), x, drop=drop_out, short_cut=short_cut, prefix='_Down_1')
            self.data_dict[self.image_resolution][str(self.layer)] = x
            x = MaxPooling3D()(x)

        for self.layer in range(self.num_layers-1,-1,-1):
            short_cut = False
            x = UpSampling3D()(x)
            x = Concatenate()([x,self.data_dict[self.image_resolution][str(self.layer)]])
            self.start_block = int(self.start_block / 2)
            x = self.residual_block(int(self.start_block), x, drop=drop_out, short_cut=short_cut, prefix='_Up_0')
            drop_out = 0.2
            short_cut = True
            x = self.residual_block(int(self.start_block), x, drop=drop_out, short_cut=short_cut, prefix='_Up_1')
        x = Conv3D(self.num_of_classes, (1, 1, 1), padding='same', activation='softmax', name=self.image_resolution + '_last_layer')(x)
        self.data_dict[self.image_resolution]['last_layer'] = x


class TDLSTM_Conv(object):
    def __init__(self, input_batch=16,input_image=256,input_channels=33, start_block=64, layers=2):
        self.input_image = input_image
        self.input_batch = input_batch
        self.input_channels = input_channels
        self.layer_start = start_block
        self.num_classes = 2
        self.layers = layers
        self.block = 0
        self.activation = 'elu'
        self.conv_number = 2
        self.unet_network()


    def conv_block(self, x):
        for i in range(self.conv_number):
            x = ConvLSTM2D(filters=int(self.layer_start), kernel_size=(3,3), padding='same', return_sequences=True,
                           activation=None, name='conv_block_' + str(self.desc) + str(self.i) + '_' + str(i))(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            if self.drop_out_spatial > 0.0:
                x = SpatialDropout3D(self.drop_out_spatial)(x)
            if self.drop_out > 0.0:
                x = Dropout(self.drop_out)(x)
            self.i += 1
        return x

    def up_sample_out(self,x):
        pool_size = int(self.input_image/int(x.shape[3]))
        if pool_size != 1:
            x = TimeDistributed(UpSampling2D((pool_size, pool_size)))(x)
        filters = 64
        for i in range(2):
            x = ConvLSTM2D(filters=int(filters), kernel_size=(3,3), padding='same', return_sequences=True,
                           activation=None, name='up_sample_conv_block_' + str(self.desc) + str(self.i) + '_' + str(i))(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            if self.drop_out_spatial > 0.0:
                x = SpatialDropout3D(self.drop_out_spatial)(x)
            if self.drop_out > 0.0:
                x = Dropout(self.drop_out)(x)
            filters /= 2
        self.layer_up += 1
        self.i += 1
        return x
    def unet_network(self):
        input_image = Input([self.input_batch,self.input_image,self.input_image,self.input_channels], name='input')

        x = input_image
        net = {}
        output_net = {}
        self.drop_out = 0.0
        self.drop_out_spatial = 0.2
        self.desc = 'down_'
        self.i = 0
        for self.block in range(self.layers):
            x = self.conv_block(x)
            net['conv ' + str(self.block)] = x
            x = TimeDistributed(MaxPooling2D((2,2), (2,2)))(x)
            self.layer_start *= 2

        self.layer_up = 0
        x = self.conv_block(x)

        self.desc = 'up_'
        for self.block in range(self.layers-1,-1,-1):
            # output_net['up_conv' + str(self.layer_up)] = self.up_sample_out(x)
            self.layer_start /= 2
            x = TimeDistributed(UpSampling2D((2,2)))(x)
            x = Concatenate(name='concat' + str(self.block) + '_Unet')([x, net['conv ' + str(self.block)]])
            x = self.conv_block(x)

        # keys = list(output_net.keys())
        # for key in keys:
        #     x = Concatenate()([x,output_net[key]])

        output = ConvLSTM2D(filters=self.num_classes, kernel_size=(3, 3), padding='same', return_sequences=True,
                       activation='softmax', name='output')(x)
        output = Flatten()(output)

        self.created_model = Model(input_image, outputs=[output])
        return None
    def network(self):
        input_image = Input([self.input_batch,self.input_image,self.input_image,self.input_channels], name='input')

        x = input_image
        self.drop_out = 0.0
        self.drop_out_spatial = 0.2
        self.desc = 'down_'
        self.i = 0
        x = self.conv_block(x)

        output = ConvLSTM2D(filters=self.num_classes, kernel_size=(3, 3), padding='same', return_sequences=True,
                       activation='softmax', name='output')(x)
        output = Flatten()(output)

        self.created_model = Model(input_image, outputs=[output])
        return None


if __name__ == '__main__':
    xxx = 1