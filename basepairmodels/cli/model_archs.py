from keras import layers, models
from keras.backend import int_shape
def DilateSumNet():
    inputs = layers.Input(shape=(1024,4))  # Returns a placeholder tensor
    x = layers.Conv1D(filters=32, kernel_size=7,activation=tf.nn.relu, padding='valid',  kernel_initializer='he_normal')(inputs)
    x = layers.Conv1D(filters=64, kernel_size=5,activation=tf.nn.relu, padding='valid', kernel_initializer='he_normal')(x)
    x = layers.Conv1D(filters=128, kernel_size=7, activation=tf.nn.relu, padding='valid', kernel_initializer='he_normal')(x)
    conv = layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x)

    dil1 = layers.Conv1D(filters=128, kernel_size=37, dilation_rate=2, activation=tf.nn.relu, padding='valid', kernel_initializer='he_normal')(conv)
    dil1Avg = layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(dil1)
    dil1Avg = layers.Cropping1D(36)(dil1Avg)

    dil2 = layers.Conv1D(filters=128, kernel_size=37, dilation_rate=4, activation=tf.nn.relu, padding='valid', kernel_initializer='he_normal')(conv)
    dil2Avg = layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(dil2)
    dil2Avg = layers.Cropping1D(18)(dil2Avg)

    dil3 = layers.Conv1D(filters=128, kernel_size=37, dilation_rate=6, activation=tf.nn.relu, padding='valid', kernel_initializer='he_normal')(conv)
    dil3Avg = layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(dil3)

    add = layers.add([dil1Avg, dil2Avg, dil3Avg])
    add = layers.Reshape((144,1,128))(add)

    up1 = layers.Conv2DTranspose(filters=128, kernel_size=(5,1), strides=(2,1))(add)
    up1conv = layers.Conv2D(filters=64, kernel_size=(3,1),activation=tf.nn.relu, padding='valid',  kernel_initializer='he_normal')(up1)
    up1conv = layers.Conv2D(filters=64, kernel_size=(3,1),activation=tf.nn.relu, padding='valid',  kernel_initializer='he_normal')(up1conv)

    up2 = layers.Conv2DTranspose(filters=64, kernel_size=(7,1), strides=(2,1))(up1conv)
    up2conv = layers.Conv2D(filters=32, kernel_size=(3,1),activation=tf.nn.relu, padding='valid',  kernel_initializer='he_normal')(up2)
    up2conv = layers.Conv2D(filters=32, kernel_size=(3,1),activation=tf.nn.relu, padding='valid',  kernel_initializer='he_normal')(up2conv)

    up3 = layers.Conv2DTranspose(filters=32, kernel_size=(5,1), strides=(2,1))(up2conv)
    up3conv = layers.Conv2D(filters=16, kernel_size=(3,1),activation=tf.nn.relu, padding='valid',  kernel_initializer='he_normal')(up3)
    up3conv = layers.Conv2D(filters=16, kernel_size=(3,1),activation=tf.nn.relu, padding='valid',  kernel_initializer='he_normal')(up3conv)

    pre_predictions = layers.Reshape((1149,16))(up3conv)
    pre_predictions = layers.Cropping1D((62,63))(pre_predictions)

    predictions = layers.LocallyConnected1D(kernel_size=1,filters=2)(pre_predictions)
    predictions = layers.Reshape((2048,))(predictions)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    return model


def DilateConcatNet():
    inputs = layers.Input(shape=(1024,4))  # Returns a placeholder tensor
    x = layers.Conv1D(filters=16, kernel_size=7,activation=tf.nn.relu, padding='same',  kernel_initializer='he_normal')(inputs)
    x = layers.Conv1D(filters=32, kernel_size=5,activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')(x)
    x = layers.Conv1D(filters=64, kernel_size=7, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.2)(x)

    dil1 = layers.Conv1D(filters=64, kernel_size=37, dilation_rate=2, activation=tf.nn.relu, padding='valid', kernel_initializer='he_normal')(x)

    dil2 = layers.Conv1D(filters=64, kernel_size=37, dilation_rate=4, activation=tf.nn.relu, padding='valid', kernel_initializer='he_normal')(x)

    dil3 = layers.Conv1D(filters=64, kernel_size=37, dilation_rate=6, activation=tf.nn.relu, padding='valid', kernel_initializer='he_normal')(x)

    x = layers.Cropping1D((108,108))(x)
    dil1 = layers.Cropping1D((72,72))(dil1)
    dil2 = layers.Cropping1D((36,36))(dil2)


    x = layers.Reshape((808, 1, 64))(x)
    dil1 = layers.Reshape((808, 1, 64))(dil1)
    dil2 = layers.Reshape((808, 1, 64))(dil2)
    dil3 = layers.Reshape((808, 1, 64))(dil3)

    concat = layers.concatenate([x, dil1, dil2, dil3], axis=2)

    predictions = layers.LocallyConnected2D(kernel_size=(1,4),filters=2)(concat)
    predictions = layers.Reshape((1616,))(predictions)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    return model

def DilateConcatConvNet():
    inputs = layers.Input(shape=(1024,4))  # Returns a placeholder tensor
    x = layers.Conv1D(filters=16, kernel_size=7,activation=tf.nn.relu, padding='valid',  kernel_initializer='he_normal')(inputs)
    x = layers.Conv1D(filters=32, kernel_size=5,activation=tf.nn.relu, padding='valid', kernel_initializer='he_normal')(x)
    x = layers.Conv1D(filters=64, kernel_size=7, activation=tf.nn.relu, padding='valid', kernel_initializer='he_normal')(x)
    x = layers.AveragePooling1D(pool_size=2,strides=2)(x)
    x = layers.Dropout(0.2)(x)

    dil1 = layers.Conv1D(filters=64, kernel_size=37, dilation_rate=2, activation=tf.nn.relu, padding='valid', kernel_initializer='he_normal')(x)

    dil2 = layers.Conv1D(filters=64, kernel_size=37, dilation_rate=4, activation=tf.nn.relu, padding='valid', kernel_initializer='he_normal')(x)

    dil3 = layers.Conv1D(filters=64, kernel_size=37, dilation_rate=6, activation=tf.nn.relu, padding='valid', kernel_initializer='he_normal')(x)

    x = layers.Cropping1D((108,108))(x)
    dil1 = layers.Cropping1D((72,72))(dil1)
    dil2 = layers.Cropping1D((36,36))(dil2)

    x = layers.Reshape((288, 1, 64))(x)
    dil1 = layers.Reshape((288, 1, 64))(dil1)
    dil2 = layers.Reshape((288, 1, 64))(dil2)
    dil3 = layers.Reshape((288, 1, 64))(dil3)

    concat = layers.concatenate([x, dil1, dil2, dil3], axis=2)

    c1 = layers.Conv2D(filters=96,kernel_size=[5,4],strides=(1,4),activation=tf.nn.relu, padding='valid',  kernel_initializer='he_normal')(concat)
    c1 = layers.AveragePooling2D(pool_size=(2,1),strides=(2,1))(c1)
    c1 = layers.Reshape((142,96))(c1)


    c2 = layers.Conv1D(filters=128,kernel_size=7,activation=tf.nn.relu, padding='valid',  kernel_initializer='he_normal')(c1)
    c2 = layers.AveragePooling1D(pool_size=2,strides=2)(c2)

    c3 = layers.Conv1D(filters=160,kernel_size=5,activation=tf.nn.relu, padding='valid',  kernel_initializer='he_normal')(c2)
    c3 = layers.AveragePooling1D(pool_size=2,strides=2)(c3)

    c4 = layers.Conv1D(filters=192,kernel_size=7,activation=tf.nn.relu, padding='valid',  kernel_initializer='he_normal')(c3)
    c4 = layers.AveragePooling1D(pool_size=2,strides=2)(c4)

    c5 = layers.Conv1D(filters=224,kernel_size=5,activation=tf.nn.relu, padding='valid',  kernel_initializer='he_normal')(c4)
    c5 = layers.AveragePooling1D(pool_size=2,strides=2)(c5)

    f = layers.Flatten()(c5)

    fc1 = layers.Dense(1024)(f)
    fc2 = layers.Dense(2048)(fc1)
    
    model = tf.keras.Model(inputs=inputs, outputs=fc2)
    
    return model

def BPNetSumAll(seq_len, out_pred_len, 
          filters=64, n_dil_layers=6, conv1_kernel_size=21, 
          profile_kernel_size=25, num_tasks=2, control_smoothing=[1, 50]):

    # The three inputs to BPNet
    inp = layers.Input(shape=(seq_len, 4), name='sequence')
    
    bias_counts_input = layers.Input(shape=(1, ), name="control_logcount")
    
    bias_profile_input = layers.Input(
        shape=(out_pred_len, len(control_smoothing)), name="control_profile")
    # end inputs

    # first convolution without dilation
    first_conv = layers.Conv1D(filters, kernel_size=conv1_kernel_size,
                               padding='valid', activation='relu', 
                               name='1st_conv')(inp)
    
    # 6 dilated convolutions with resnet-style additions
    # each layer receives the sum of feature maps 
    # from all previous layers
    # *** on a quest to have meaninful layer names *** 
    res_layers = [(first_conv, '1stconv')] 
                                           
    layer_names = ['2nd', '3rd', '4th', '5th', '6th', '7th']
    for i in range(1, n_dil_layers + 1):
        if i == 1:
            res_layers_sum = first_conv
        else:
            res_layers_sum = layers.add([l for l, _ in res_layers],
                                        name='add_{}'.format(i))

        # dilated convolution
        conv_layer_name = '{}conv'.format(layer_names[i - 1])
        conv_output = layers.Conv1D(filters, kernel_size=3, padding='valid',
                                    activation='relu', dilation_rate=2**i,
                                    name=conv_layer_name)(res_layers_sum)

        # get shape of latest layer and crop 
        # all other previous layers in the list to that size
        conv_output_shape =int_shape(conv_output)
        cropped_layers = []
        for lyr, name in res_layers:
            lyr_shape = int_shape(lyr)
            cropsize = lyr_shape[1] //2 - conv_output_shape[1] // 2
            lyr_name = '{}-crop_{}th_dconv'.format(name.split('-')[0], i)
            cropped_layers.append(
                (layers.Cropping1D(cropsize, name=lyr_name)(lyr), lyr_name))
        
        # append to the list of previous layers
        cropped_layers.append((conv_output, conv_layer_name))
        res_layers = cropped_layers

    # the final output from the 6 dilated convolutions 
    # with resnet-style connections
    combined_conv = layers.add([l for l, _ in res_layers], 
                               name='combined_conv') 

    # Branch 1. Profile prediction
    # Step 1.1 - 1D convolution with a very large kernel
    profile_out_prebias = layers.Conv1D(filters=num_tasks, 
                                        kernel_size=profile_kernel_size, 
                                        padding='valid', 
                                        name='profile_out_prebias')\
                                        (combined_conv)

    # Step 1.2 - Crop to match size of the required output size, a
    #            minimum difference of 346 is required between input
    # .          seq len and ouput len
    profile_out_prebias_shape = int_shape(profile_out_prebias)
    cropsize = profile_out_prebias_shape[1] // 2 - out_pred_len // 2
    profile_out_prebias = layers.Cropping1D(
        cropsize, name='prof_out_crop2match_output')(profile_out_prebias)

    # Step 1.3 - concatenate with the control profile 
    concat_pop_bpi = layers.concatenate(
        [profile_out_prebias, bias_profile_input], 
        name="concat_with_bias_prof", axis=-1)

    # Step 1.4 - Final 1x1 convolution
    profile_out = layers.Conv1D(filters=num_tasks, kernel_size=1, 
                                name="profile_predictions")(concat_pop_bpi)

    # Branch 2. Counts prediction
    # Step 2.1 - Global average pooling along the "length", the result
    #            size is same as "filters" parameter to the BPNet 
    #            function
    # acronym - gapcc
    gap_combined_conv = layers.GlobalAveragePooling1D(
        name='gap')(combined_conv) 
    
    # Step 2.2 Concatenate the output of GAP with bias counts
    concat_gapcc_bci = layers.concatenate(
        [gap_combined_conv, bias_counts_input], name="concat_with_bias_cnts", 
        axis=-1)
    
    # Step 2.3 Dense layer to predict final counts
    count_out = layers.Dense(num_tasks, 
                             name="logcount_predictions")(concat_gapcc_bci)
  
    # instantiate keras Model with inputs and outputs
    model = models.Model(inputs=[inp, bias_counts_input, bias_profile_input],
                         outputs=[profile_out, count_out])
    
    return model

def BPNet(seq_len, out_pred_len, 
          filters=64, n_dil_layers=6, conv1_kernel_size=21, 
          profile_kernel_size=25, num_tasks=2, control_smoothing=[1, 50]):

    # The three inputs to BPNet
    inp = layers.Input(shape=(seq_len, 4), name='sequence')
    
    bias_counts_input = layers.Input(shape=(1, ), name="control_logcount")
    
    bias_profile_input = layers.Input(
        shape=(out_pred_len, len(control_smoothing)), name="control_profile")
    # end inputs

    # first convolution without dilation
    first_conv = layers.Conv1D(filters, kernel_size=conv1_kernel_size,
                               padding='valid', activation='relu', 
                               name='1st_conv')(inp)
    
    # 6 dilated convolutions with resnet-style additions
    # each layer receives the sum of feature maps 
    # from previous two layers
    # *** on a quest to have meaninful layer names *** 
    res_layers = [(first_conv, '1stconv')] 
                                           
    layer_names = ['2nd', '3rd', '4th', '5th', '6th', '7th']
    for i in range(1, n_dil_layers + 1):
        if i == 1:
            res_layers_sum = first_conv
        else:
            res_layers_sum = layers.add([l for l, _ in res_layers],
                                        name='add_{}'.format(i))

        # dilated convolution
        conv_layer_name = '{}conv'.format(layer_names[i - 1])
        conv_output = layers.Conv1D(filters, kernel_size=3, padding='valid',
                                    activation='relu', dilation_rate=2**i,
                                    name=conv_layer_name)(res_layers_sum)

        # get shape of latest layer and crop the previous
        # layer (index == -1) in the res_layers list to that size
        conv_output_shape = int_shape(conv_output)
        cropped_layers = []
        
        lyr, name = res_layers[-1]
        lyr_shape = int_shape(lyr)
        cropsize = lyr_shape[1] // 2 - conv_output_shape[1] // 2
        lyr_name = '{}-crop_{}th_dconv'.format(name.split('-')[0], i)
        cropped_layers.append(
            (layers.Cropping1D(cropsize, name=lyr_name)(lyr), lyr_name)) 
        
        # now append the current conv_output
        cropped_layers.append((conv_output, conv_layer_name))
        
        res_layers = cropped_layers

    # the final output from the 6 dilated convolutions 
    # with resnet-style connections
    combined_conv = layers.add([l for l, _ in res_layers], 
                               name='combined_conv') 

    # Branch 1. Profile prediction
    # Step 1.1 - 1D convolution with a very large kernel
    profile_out_prebias = layers.Conv1D(filters=num_tasks, 
                                        kernel_size=profile_kernel_size, 
                                        padding='valid', 
                                        name='profile_out_prebias')\
                                        (combined_conv)

    # Step 1.2 - Crop to match size of the required output size, a
    #            minimum difference of 346 is required between input
    # .          seq len and ouput len
    profile_out_prebias_shape = int_shape(profile_out_prebias)
    cropsize = profile_out_prebias_shape[1] // 2 - out_pred_len // 2
    profile_out_prebias = layers.Cropping1D(
        cropsize, name='prof_out_crop2match_output')(profile_out_prebias)

    # Step 1.3 - concatenate with the control profile 
    concat_pop_bpi = layers.concatenate(
        [profile_out_prebias, bias_profile_input], 
        name="concat_with_bias_prof", axis=-1)

    # Step 1.4 - Final 1x1 convolution
    profile_out = layers.Conv1D(filters=num_tasks, kernel_size=1, 
                                name="profile_predictions")(concat_pop_bpi)

    # Branch 2. Counts prediction
    # Step 2.1 - Global average pooling along the "length", the result
    #            size is same as "filters" parameter to the BPNet 
    #            function
    # acronym - gapcc
    gap_combined_conv = layers.GlobalAveragePooling1D(
        name='gap')(combined_conv) 
    
    # Step 2.2 Concatenate the output of GAP with bias counts
    concat_gapcc_bci = layers.concatenate(
        [gap_combined_conv, bias_counts_input], name="concat_with_bias_cnts", 
        axis=-1)
    
    # Step 2.3 Dense layer to predict final counts
    count_out = layers.Dense(num_tasks, 
                             name="logcount_predictions")(concat_gapcc_bci)
  
    # instantiate keras Model with inputs and outputs
    model = models.Model(inputs=[inp, bias_counts_input, bias_profile_input],
                         outputs=[profile_out, count_out])
    
    return model
