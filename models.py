import keras
import keras.backend as K
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, AveragePooling2D, Activation, Add, Dropout
from keras.models import Model

def res_block(x, filters, kernel_size=3):
    
    x1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x1)

    if x.shape.as_list()[-1] != filters:
        x2 = Conv2D(filters=filters, kernel_size=1)(x)
    else:
        x2 = x
    
    x = Add()([x1, x2])
    x = Activation('relu')(x)
    
    return x

def resnet_minc(input_shape=(362,362,3), params=[8, 16, 32, 64, 64, 0.2, 0.2]):

    filters = params[:4]
    dense_width = params[4]
    dropout_rates = params[5:]

    inp = Input(shape=input_shape)

    x = res_block(inp, filters=filters[0])
    x = MaxPooling2D(pool_size=2)(x)

    x = res_block(x, filters=filters[1])
    x = MaxPooling2D(pool_size=2)(x)

    x = res_block(x, filters=filters[2])
    x = MaxPooling2D(pool_size=2)(x)

    x = res_block(x, filters=filters[3])
    x = AveragePooling2D(pool_size=5)(x)
    x = Flatten()(x)
    x = Dropout(dropout_rates[0])(x)

    x = Dense(dense_width)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rates[1])(x)

    x = Dense(23)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=inp, outputs=x)
    return model

def resnet_fmd(input_shape=(128,128,3), params=[8, 16, 32, 0.2]):

    filters = params[:3]
    dropout_rate = params[3]

    inp = Input(shape=input_shape)

    x = res_block(inp, filters=filters[0])
    x = MaxPooling2D(pool_size=4)(x)

    x = res_block(x, filters=filters[1])
    x = MaxPooling2D(pool_size=4)(x)

    x = res_block(x, filters=filters[2])
    x = MaxPooling2D(pool_size=2)(x)
    
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(10)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=inp, outputs=x)
    return model
