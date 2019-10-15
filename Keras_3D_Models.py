from Utilities.Keras_Utils import Model, K, np, weighted_mse, categorical_crossentropy_masked, tf
from keras.backend import variable
from keras.layers import *
from keras.initializers import RandomNormal
from keras.models import load_model
from functools import partial, update_wrapper
from keras.utils import conv_utils


ExpandDimension = lambda axis: Lambda(lambda x: K.expand_dims(x, axis))
SqueezeDimension = lambda axis: Lambda(lambda x: K.squeeze(x, axis))
Subtract_new = lambda y: Lambda(lambda x: Subtract()([x,y]))
Multipy_new = lambda y: Lambda(lambda x: Multiply()([x,y]))

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

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


class DCEC(object):
    def __init__(self,output_shape=(28,28,28,1),
                 ae_model_path='.',
                 n_clusters=4,
                 alpha=1.0):

        super(DCEC, self).__init__()

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []

        self.cae = load_model(ae_model_path)
        hidden = self.cae.get_layer(name='Layer_0_Decoding_Conv1').output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        # Define DCEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(Flatten()(hidden))
        clustering_layer = Reshape(output_shape)(clustering_layer)
        self.model = Model(inputs=self.cae.input,outputs=clustering_layer)


class Unet(object):

    def __init__(self,save_memory=False):
        self.previous_conv = None
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
        else:
            self.conv = Conv3D
            self.pool = MaxPooling3D
            self.up_sample = UpSampling3D

    def define_batch_norm(self, batch_norm=False):
        self.batch_norm = batch_norm

    def define_filters(self, filters):
        self.filters = filters
        if len(filters) == 2:
            self.define_2D_or_3D(True)
        else:
            self.define_2D_or_3D()

    def define_activation(self, activation):
        self.activation = activation

    def define_pool_size(self, pool_size):
        self.pool_size = pool_size

    def define_padding(self, padding='same'):
        self.padding = padding

    def conv_block(self,output_size,x, name, strides=1, dialation_rate=1, activate=True,filters=None):
        if not filters:
            filters = self.filters
        if len(filters) + 1 == len(x.shape):
            self.define_2D_or_3D(is_2D=False)
            x = ExpandDimension(0)(x)
        elif len(filters) + 2 < len(x.shape):
            self.define_2D_or_3D(True)
            x = SqueezeDimension(0)(x)
        if not self.save_memory or max(filters) == 1:
            x = self.conv(output_size, filters, activation=None, padding=self.padding,
                       name=name, strides=strides, dilation_rate=dialation_rate)(x)
        else:
            for i in range(len(filters)):
                filter = np.ones(len(filters)).astype('int')
                filter[i] = filters[i]
                x = self.conv(output_size, filter, activation=None, padding=self.padding,name=name+'_'+str(i), strides=strides, dilation_rate=dialation_rate)(x) # Turn a 3x3 into a 3x1 with a 1x3
        if self.batch_norm:
            x = BatchNormalization()(x)
        if activate:
            x = Activation(self.activation,name=name+'_activation')(x)
        return x

    def residual_block(self, output_size,x,name,blocks=0):
        # This used to be input_val is the convolution
        if x.shape[-1] != output_size:
            x = self.conv_block(output_size,x=x,name=name + '_' + 'rescale_input',activate=False,filters=self.filters)
            x = input_val = Activation(self.activation)(x)
        else:
            input_val = x

        for i in range(blocks):
            x = self.conv_block(output_size,x,name=name+'_'+str(i))
        x = self.conv(output_size, self.filters, activation=None, padding=self.padding,name=name)(x)
        x = Add(name=name+'_add')([x,input_val])
        x = Activation(self.activation)(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
        return x

    def atrous_block(self, output_size, x, name, rate_blocks=5): # https://arxiv.org/pdf/1901.09203.pdf, follow formula of k^(n-1)
        # where n is the convolution layer number, this is for k = 3, 5 gives a field of 243x243
        rates = []
        get_new = True
        if x.shape[-1] == output_size:
            input_val = x
            get_new = False
        #     x = input_val = self.conv_block(output_size, x=x, name=name + 'Atrous_' + 'rescale_input', activate=True,
        #                                     filters=self.filters)
        # else:
        #     input_val = x
        for rate_block in range(rate_blocks):
            rate = []
            for i in range(len(self.filters)):
                rate.append(self.filters[i]**(rate_block)) # not plus 1 minus 1, since we are 0 indexed
            # if len(rate) == 3 and rate[0] > 9:
            #     rate[0] = 9
            rates.append(rate)
        for i, rate in enumerate(rates):
            temp_name = name + 'Atrous_' + str(rate[-1])
            x = self.conv_block(output_size=output_size,x=x,name=temp_name,dialation_rate=rate,activate=False, filters=self.filters)
            # x = self.conv(output_size,self.filters, activation=None,padding=self.padding, name=temp_name, dilation_rate=rate)(x)
            if i == len(rates)-1:
                x = Add(name=name+'_add')([x,input_val])
            x = Activation(self.activation, name=temp_name + '_activation')(x)
            if i == 0 and get_new:
                input_val = x
            if self.batch_norm:
                x = BatchNormalization()(x)
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
        x = Activation(self.activation, name=name+'_activation')(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
        y = layer(y)
        y = Activation(self.activation, name=name+'_activation')(y)
        if self.batch_norm:
            y = BatchNormalization()(y)
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

    def dict_conv_block(self, x, filter_dict, desc):
        filter_vals = None
        res_blocks = None
        atrous_blocks = None
        activations = None
        if 'Res_block' in filter_dict:
            res_blocks = filter_dict['Res_block']
        if 'Kernel' in filter_dict:
            filter_vals = filter_dict['Kernel']
        if 'Atrous_block' in filter_dict:
            atrous_blocks = filter_dict['Atrous_block']
        if 'Activation' in filter_dict:
            activations = filter_dict['Activation']
        if 'Channels' in filter_dict:
            all_filters = filter_dict['Channels']
        elif 'FC' in filter_dict:
            if len(x.shape) != 2:
                x = Flatten()(x)
            for i, rate in enumerate(filter_dict['FC']):
                if activations:
                    self.define_activation(activations[i])
                x = self.FC_Block(rate,x,dropout=filter_dict['Dropout'][i], name=desc + '_FC_'+str(i))
            return x
        else:
            all_filters = filter_dict
        rescale = False
        for i in range(len(all_filters)):
            if activations:
                self.define_activation(activations[i])
            if filter_vals:
                self.define_filters(filter_vals[i])
                if len(filter_vals[i]) + 1 == len(x.shape):
                    self.define_2D_or_3D(is_2D=False)
                    x = ExpandDimension(0)(x)
                    rescale = True
                elif len(filter_vals[i]) + 1 > len(x.shape):
                    self.define_2D_or_3D(True)
                    x = SqueezeDimension(0)(x)
            strides = 1
            if rescale:
                self.desc = desc + '3D_' + str(i)
            else:
                self.desc = desc + str(i)

            if res_blocks:
                rate = res_blocks[i] if res_blocks else 0
                x = self.residual_block(all_filters[i], x=x, name=self.desc,blocks=rate)
            elif atrous_blocks:
                x = self.atrous_block(all_filters[i],x=x,name=self.desc,rate_blocks=atrous_blocks[i])
            else:
                x = self.conv_block(all_filters[i], x=x, strides=strides, name=self.desc)
        return x

    def run_filter_dict(self, x, layer_dict, layer, desc):
        if type(layer_dict) == list:
            for i, filters in enumerate(layer_dict):
                x = self.dict_conv_block(x, filters, layer + '_' + desc + '_' + str(i))
        else:
            x = self.dict_conv_block(x, layer_dict, layer + '_' + desc + '_')
        return x

    def run_unet(self, x):
        self.layer = 0
        self.layer_vals = {}
        desc = 'Encoder'
        layer_index = 0
        layer_order = []
        for layer in self.layers_names:
            print(layer)
            if layer == 'Base':
                continue
            layer_order.append(layer)
            all_filters = self.layers_dict[layer]['Encoding']
            x = self.run_filter_dict(x, all_filters, layer, desc)
            self.layer_vals[layer_index] = x
            if 'Pooling' not in self.layers_dict[layer] or ('Pooling' in self.layers_dict[layer] and self.layers_dict[layer]['Pooling'] is not None):
                if 'Pooling' in self.layers_dict[layer]:
                    self.define_pool_size(self.layers_dict[layer]['Pooling'])
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
            if 'Decoding' not in self.layers_dict[layer]:
                continue
            print(layer)
            layer_index -= 1
            if 'Pooling' in self.layers_dict[layer]:
                self.define_pool_size(self.layers_dict[layer]['Pooling'])
            if concat:
                x = self.up_sample(size=self.pool_size, name='Upsampling' + str(self.layer) + '_UNet')(x)
                x = Concatenate(name='concat' + str(self.layer) + '_Unet')([x, self.layer_vals[layer_index]])
            all_filters = self.layers_dict[layer]['Decoding']
            x = self.run_filter_dict(x, all_filters, layer, desc)
            self.layer += 1
        return x


class base_UNet(Unet):
    def __init__(self, filter_vals=(3,3,3),layers_dict=None, pool_size=(2,2,2),activation='elu', pool_type='Max',
                 batch_norm=False,is_2D=False,save_memory=False):
        super().__init__(save_memory=save_memory)
        self.layer_vals = {}
        self.define_2D_or_3D(is_2D)
        self.define_unet_dict(layers_dict)
        self.define_pool_size(pool_size)
        self.define_batch_norm(batch_norm)
        self.define_filters(filter_vals)
        self.define_activation(activation)
        self.define_padding('same')
        self.define_pooling_type(pool_type)

    def get_unet(self,layers_dict):
        pass


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
            x = Activation(self.activation)(x)
            x = BatchNormalization()(x)
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

class my_3D_UNet(base_UNet):

    def __init__(self, filter_vals=(3,3,3),layers_dict=None, pool_size=(2,2,2),create_model=True, activation='elu',pool_type='Max',final_activation='softmax',z_images=None,complete_input=None,
                 batch_norm=False, striding_not_pooling=False, out_classes=2,is_2D=False,semantic_segmentation=True, input_size=1,save_memory=False, mask_input=False, image_size=None,
                 mean_val=0,std_val=1):
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
        self.mean_val = mean_val
        self.std_val = std_val
        super().__init__(filter_vals=filter_vals, layers_dict=layers_dict, pool_size=pool_size, activation=activation,
                         pool_type=pool_type, batch_norm=batch_norm, is_2D=is_2D,save_memory=save_memory)
        self.striding_not_pooling = striding_not_pooling
        self.out_classes = out_classes
        self.mask_input = mask_input
        self.get_unet(layers_dict)

    def get_unet(self, layers_dict):
        if self.complete_input is None:
            if self.is_2D:
                image_input_primary = x = Input(shape=(self.image_size, self.image_size, self.input_size), name='UNet_Input')
            else:
                image_input_primary = x = Input(shape=(self.z_images, self.image_size, self.image_size, self.input_size), name='UNet_Input')
        else:
            image_input_primary = x = self.complete_input
        if self.is_2D:
            output_kernel = (1,1)
        else:
            output_kernel = (1,1,1)
        # Normalizing image
        mean_val = variable(value=(self.mean_val,))
        std_val = variable(value=(self.std_val,))
        x = Subtract_new(mean_val)(x)
        x = Multipy_new(1/std_val)(x)
        x = self.run_unet(x)
        self.save_memory = False
        self.define_filters(output_kernel)
        if self.semantic_segmentation:
            x = self.conv_block(self.out_classes, x, name='output', activate=False)
        if self.final_activation is not None:
            x = Activation(self.final_activation)(x)
        if self.mask_input:
            self.mask = Input(shape=(None,None,None,1),name='mask')
            inputs = [image_input_primary,self.mask]
            partial_func = partial(categorical_crossentropy_masked, mask=self.mask)
            self.custom_loss = update_wrapper(partial_func, categorical_crossentropy_masked)
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
        y = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([image_input_primary, flow])
        model = Model(inputs=[image_input_primary,image_input_secondary,weights_tensor], outputs=y)
        self.created_model = model
        # partial(weighted_mse, weights=weights_tensor)
        # self.custom_loss = wrapped_partial(weighted_mse,weights=weights_tensor))
        partial_func = partial(weighted_mse, weights=weights_tensor)
        self.custom_loss = update_wrapper(partial_func, weighted_mse)


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
            x = BatchNormalization()(x)
            x = Conv3D(int(block), kernel_size=self.filters, padding='same',
                       name='Conv_concat' + self.desc + str(self.layer) + '_UNet')(x)
            x = Activation(self.activation)(x)
            block = filter_vals[i] # Mirror one side to the next
            # x = Conv3D(block, self.filter_vals, activation=None, padding='same', name='reduce_' + str(i))(x)
            # x = BatchNormalization()(x)
            # x = Activation(self.activation)(x)

            # block /= 2
            x = self.conv_block(int(block), x)
            self.layer += 1

        x = BatchNormalization()(x)
        x = Conv3D(64, (3, 3, 3), activation=None, padding='same', name='before_output_0')(x)
        x = Activation(self.activation)(x)
        x = BatchNormalization()(x)
        x = Conv3D(32, (3, 3, 3), activation=None, padding='same', name='before_output_1')(x)
        x = Activation(self.activation)(x)
        x = BatchNormalization()(x)
        x = Conv3D(3, (1,1,1), activation=None, padding='same',name='output_Unet')(x)
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


if __name__ == '__main__':
    xxx = 1