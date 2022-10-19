from imports import *
import metrics
import data_management
import interactive_plot

def load_model_unet(file, use_dice_loss=False, use_jaccard_loss=False, use_focal_loss=False):
    from keras.models import load_model
    cobj = {'keras_precision': metrics.keras_precision, 'keras_recall': metrics.keras_recall, 'keras_jaccard_coef': metrics.keras_jaccard_coef, 'keras_dice_coef': metrics.keras_dice_coef}
    if use_dice_loss:
        cobj['keras_dice_coef_loss'] = metrics.keras_dice_coef_loss
    if use_jaccard_loss:
        cobj['keras_jaccard_distance_loss'] = metrics.keras_jaccard_distance_loss
    if use_focal_loss:
        cobj['keras_focal_loss'] = metrics.keras_focal_loss
    model = keras.models.load_model(file, custom_objects=cobj)
    return model

def predict_net(model, img, verbose=1):
    imgs_mask_test = model.predict(img, batch_size=1, verbose=verbose)
    return imgs_mask_test

def train(model, imgs_train, imgs_mask_train, imgs_test, imgs_mask_test, model_name, bt_size, train_epochs, iter_per_epoch=100, val_steps=40, finetune_path=None, perform_centering=False, perform_flipping=True, perform_rotation=True,
          perform_standardization=False, plot_graph=True, verbosity=1, to_dir=False, train_on_borders=False):
      
    print('-------------------------------')
    print('data details:')

    print('imgs_train.shape', imgs_train.shape)
    print('imgs_mask_train.shape', imgs_mask_train.shape)
    print('imgs_test.shape', imgs_test.shape)
    print('imgs_mask_test.shape', imgs_mask_test.shape)
    print('imgs_train.dtype', imgs_train.dtype)
    print('imgs_mask_train.dtype', imgs_mask_train.dtype)
    print('imgs_test.dtype', imgs_test.dtype)
    print('imgs_mask_test.dtype', imgs_mask_test.dtype)
    
    print('balance:')
    totalpx = (imgs_mask_train.shape[0]*imgs_mask_train.shape[1]*imgs_mask_train.shape[2])
    marked = np.sum(imgs_mask_train>0.5)
    print('train_total_px', totalpx)
    print('train_labeled_positive', marked)
    resag = marked/float(totalpx)
    print('train_fraction_positive', resag)
    print('min, max train', imgs_train.min(), imgs_train.max())
    print('train_total_px', totalpx, 'number 0/1 in mask', np.sum(imgs_mask_train==0)+np.sum(imgs_mask_train==1))

    totalpx_test = (imgs_mask_test.shape[0]*imgs_mask_test.shape[1]*imgs_mask_test.shape[2])
    marked_test = np.sum(imgs_mask_test>0.5)
    print('test_total_px', totalpx_test)
    print('test_labeled_positive', marked_test)
    resag_test = marked_test/float(totalpx_test)
    print('test_fraction_positive', resag_test)        
    print('min, max test', imgs_test.min(), imgs_test.max())
    
    print("loading data done")
        
    if not os.path.isdir(model_name):
        os.mkdir(model_name)

    sname = model_name + '/weights.{epoch:02d}-{loss:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(sname, monitor='val_loss',verbose=1, save_best_only=False, save_freq=5)
    print('saving model checkpoints in', model_name + '/')
    
    logfile_path = model_name + '/log.txt'
    lcallbacks = [interactive_plot.InteractivePlot(logfile_path, plot_graph, iter_per_epoch), model_checkpoint]
    print('saving logfile in ', str(logfile_path))
    
    
    data_gen_args = dict()
    data_val_gen_args = dict()
    print('augmenting on the fly...')
    print('computing statistics...')
    train_mean =None
    train_sd = None
    
    if perform_standardization:
        print('will standardize!')
        train_mean = np.mean(imgs_train, axis=0)
        train_sd = np.std(imgs_train, axis=0)
    elif perform_centering:
        print('will center!')
        train_mean = np.mean(imgs_train, axis=0)

    def cus_raw(img):
        return data_management.custom_preproc(img, 'raw', flip_z=perform_flipping, rotate=perform_rotation, mean=train_mean, sd=train_sd, train_on_borders=train_on_borders)

    def cus_mask(img):
        return data_management.custom_preproc(img, 'mask', flip_z=perform_flipping, rotate=perform_rotation, mean=train_mean, sd=train_sd, train_on_borders=train_on_borders)

    seed = 1
    
    #setting up train generator
    data_gen_args = dict(num_channels = MEMORY,
                         featurewise_center=False,
                         featurewise_std_normalization=False,               
                         samplewise_center=False, 
                         samplewise_std_normalization=False,
                         #rotation_range=180.,
                         #width_shift_range=0.05, 
                         #height_shift_range=0.05, 
                         #channel_shift_range=0.05,
                         #fill_mode='constant', cval=0,
                         preprocessing_function = cus_raw,               
                         horizontal_flip=perform_flipping,
                         vertical_flip=perform_flipping)

    datagen = extension.ExtImageDataGenerator(**data_gen_args) 
    
    data_gen_args_mask = data_gen_args.copy()
    data_gen_args_mask['num_channels'] = 1 
    data_gen_args_mask['preprocessing_function'] = cus_mask 
    data_gen_args_mask['featurewise_center'] = False
    data_gen_args_mask['featurewise_std_normalization'] = False
    data_gen_args_mask['samplewise_center'] = False
    data_gen_args_mask['samplewise_std_normalization'] = False
    datagen_mask = extension.ExtImageDataGenerator(**data_gen_args_mask)

    datagen.fit(imgs_train, augment=False, seed=seed)
    datagen_mask.fit(imgs_mask_train, augment=False, seed=seed)

    
    train_dir = None
    test_dir = None
    train_pref = None
    test_pref = None
    if to_dir:
        train_dir = 'a'
        test_dir = 'b'
        train_pref = 'img'
        test_pref = 'img'
        print('saving augmented images!')
    datagen = datagen.flow(imgs_train, batch_size=bt_size,shuffle=True, seed=seed, save_to_dir=train_dir, save_prefix=train_pref)
    datagen_mask = datagen_mask.flow(imgs_mask_train, batch_size=bt_size, shuffle=True,seed=seed, save_to_dir=test_dir, save_prefix=test_pref)

    def combine_generator(gen1, gen2):
        while True:
            yield(next(gen1), next(gen2))
    
    train_generator = combine_generator(datagen, datagen_mask)#zip(datagen, datagen_mask)

    #setting up validation generator
    data_val_gen_args = dict(num_channels = MEMORY,
                             featurewise_center=False,
                             featurewise_std_normalization=False,
                             preprocessing_function = cus_raw,
                             samplewise_center=False,
                             samplewise_std_normalization=False)

    datagen_val = extension.ExtImageDataGenerator(**data_val_gen_args)
   
    data_val_gen_args_mask = data_val_gen_args.copy()
    data_val_gen_args_mask['num_channels'] = 1 
    data_val_gen_args_mask['preprocessing_function'] = cus_mask 
    data_val_gen_args_mask['featurewise_center'] = False
    data_val_gen_args_mask['featurewise_std_normalization'] = False
    data_val_gen_args_mask['samplewise_center'] = False
    data_val_gen_args_mask['samplewise_std_normalization'] = False
    datagen_val_mask = extension.ExtImageDataGenerator(**data_val_gen_args_mask)
    
    datagen_val.fit(imgs_train, augment=False, seed=seed) #fit to train, to substract training and not test mean
    datagen_val_mask.fit(imgs_mask_train, augment=False, seed=seed)

    datagen_val = datagen_val.flow(imgs_test, batch_size=bt_size, shuffle=True, seed=seed)
    datagen_val_mask = datagen_val_mask.flow(imgs_mask_test, batch_size=bt_size, shuffle=True, seed=seed)

    test_generator = combine_generator(datagen_val, datagen_val_mask)#zip(datagen_val, datagen_val_mask)

    print(train_generator)
    model.fit_generator(train_generator,
                             steps_per_epoch=iter_per_epoch, epochs=train_epochs, 
                             validation_data=test_generator,
                             validation_steps=val_steps, callbacks=lcallbacks, verbose=verbosity)

def get_unet(lrate, img_rows=512, img_cols=512, dr_rate=0.5, diceloss=False, jaccardloss=False, focalloss=False, customloss=False, start_filters=8):
    inputs = Input((img_rows, img_cols, MEMORY))
    print('start_filters', start_filters)

    conv1 = Conv2D(start_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    print("conv1 shape:",conv1.shape)
    conv1 = Conv2D(start_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    print("conv1 shape:",conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print("pool1 shape:",pool1.shape)

    conv2 = Conv2D(start_filters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    print("conv2 shape:",conv2.shape)
    conv2 = Conv2D(start_filters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    print("conv2 shape:",conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print("pool2 shape:",pool2.shape)

    conv3 = Conv2D(start_filters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    print("conv3 shape:",conv3.shape)
    conv3 = Conv2D(start_filters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    print("conv3 shape:",conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print("pool3 shape:",pool3.shape)

    conv4 = Conv2D(start_filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(start_filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(dr_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(start_filters*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(start_filters*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(dr_rate)(conv5)

    up6 = Conv2D(start_filters*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
#     merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    merge6 = merge.concatenate([drop4,up6], axis=3)
    conv6 = Conv2D(start_filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(start_filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(start_filters*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
#     merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    merge7 = merge.concatenate([conv3,up7], axis=3)
    conv7 = Conv2D(start_filters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(start_filters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(start_filters*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
#     merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    merge8 = merge.concatenate([conv2,up8], axis=3)
    conv8 = Conv2D(start_filters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(start_filters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(start_filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
#     merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    merge9 = merge.concatenate([conv1,up9], axis=3)
    conv9 = Conv2D(start_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(start_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    
    lss = 'binary_crossentropy' 
    if diceloss:
        print('using dice loss!')
        lss = metrics.keras_dice_coef_loss
    elif jaccardloss:
        print('using jaccard loss!')
        lss = metrics.keras_jaccard_distance_loss
    elif focalloss:
        print('using focal loss!')
        lss = metrics.keras_focal_loss
    elif customloss:
        lss = metrics.keras_binary_crossentropy_mod

    model = Model(inputs, conv10)

    model.compile(optimizer = Adam(lr = lrate), loss = lss, metrics = ['accuracy', metrics.keras_precision, metrics.keras_recall, metrics.keras_jaccard_coef, metrics.keras_dice_coef])

    return model


#normal execute_predict
def execute_predict(model, img_in, stepsize=512, resize_shortest=True, extensive=True):
    if img_in.shape[1]==512 and img_in.shape[2]==512: #no need for resize
        return model.predict(img_in) 
    
    original_shape = img_in.shape[1:3]
    img_in = img_in.copy()
    downsampled=False
    if resize_shortest: #resize s.t. shorter side is 512
        downsampled=True
        new_img_in = []
        for i in tqdm(range(img_in.shape[0])):
            ssize = min(img_in[i].shape[0], img_in[i].shape[1])
            lsize = max(img_in[i].shape[0], img_in[i].shape[1])
            newssize = 512
            newlsize = int(lsize/float(ssize)*newssize)
            
            if img_in[i].shape[1]>img_in[i].shape[0]: #wide
                new_img_in.append(cv2.resize(img_in[i], (newlsize, newssize)))

            else: #high
                new_img_in.append(cv2.resize(img_in[i], (newssize, newlsize)))
            
        img_in = np.array(new_img_in)
    else:
        img_in = img_in[:,:,:,0]
        
    s_to_feed = []
    startx=0
    starty=0
    num_newsamples=0
    index_stack=0
    num_imgs, ih,iw = img_in.shape
    
    big_res = np.zeros(shape=img_in.shape)
    num_eval_field = np.zeros(shape=img_in.shape)
    
    transforms = ['no']
    if extensive:
        transforms.extend(['flip-x', 'flip-y', 'flip-xy'])
    
    total = len(list(range(0, iw, stepsize)))*len(list(range(0, ih, stepsize)))*len(transforms)
    gct = 0
    for startx in range(0, iw, stepsize):
        for starty in range(0, ih, stepsize):
            if startx+512>iw: startx=iw-512
            if starty+512>ih: starty=ih-512
            crop = img_in[:,starty:(starty+512),startx:(startx+512)]
            crop = crop.reshape(crop.shape[0], crop.shape[1], crop.shape[2], 1)
            
            for t in transforms:
                currimg = crop.copy()
                if t in ['flip-x', 'flip-xy']: #flip vertically
                    currimg = np.flip(currimg, 0)
                if t in ['flip-y', 'flip-xy']: #flip horizontally
                    currimg = np.flip(currimg, 1)
                pred = predict_net(model, currimg, verbose=1)
                pred = pred[:,:,:,0]
                
                if t in ['flip-x', 'flip-xy']: #flip vertically
                    pred = np.flip(pred, 0)
                if t in ['flip-y', 'flip-xy']: #flip horizontally
                    pred = np.flip(pred, 1)
                
                big_res[:,starty:(starty+512),startx:(startx+512)]+=pred
                num_eval_field[:,starty:(starty+512),startx:(startx+512)]+=np.ones_like(pred)
                gct+=1
                print(gct, 'of', total)

                
                
    big_res/=num_eval_field
    
    if downsampled: #upsample result
        big_res_new = []
        for i in range(big_res.shape[0]):
            big_res_new.append(cv2.resize(big_res[i], (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST))
        big_res = np.array(big_res_new)
                
    big_res = big_res.reshape(big_res.shape[0],big_res.shape[1], big_res.shape[2], 1) 
    return big_res 

def improve_components(test_pred, depth=9):
    return median_filter(test_pred, size=(depth,1,1,1)) #run z-smoothing (median filter)
   
def predict(model, img, groundtruth=None, overlay=True, threshold=0.1, stepsize=512, resize_shortest=True, verbose=False):
    img_in = img.astype(float)
    img_in/=255.

    newimg = np.zeros((img.shape[1], img.shape[2], 3))
    newimg[:,:,0] = img[0,:,:,0]
    newimg[:,:,1] = img[0,:,:,0]
    newimg[:,:,2] = img[0,:,:,0]   
    
    res = execute_predict(model, img_in, stepsize, resize_shortest)
    if verbose:
        print('res', res.shape)
        print('res_min', res.min())
        print('res_max', res.max())
        print('part pos.', np.sum(res>threshold))
    
    tres = res[0,:,:,0]
    tres = (1-tres)
    tres_embedded = np.zeros((tres.shape[0], tres.shape[1], 4))
    tres_embedded[:,:,0] = 255
    tres_embedded[:,:,1] = tres*0
    tres_embedded[:,:,2] = tres*0
    tres_embedded[:,:,3] = tres>threshold
    
    blended = tres_embedded
    only_gt = tres_embedded
    only_pred = tres_embedded
    if overlay:
        blended = newimg.copy() 
        only_gt = newimg.copy()
        only_pred = newimg.copy()
        if groundtruth is not None:
            print('gt', groundtruth.shape)
            only_gt[groundtruth[:,:,0]>0] = np.minimum(only_gt[groundtruth[:,:,0]>0]+(0,100,0), 255) #(0,255,0) green for groundtruth
            blended[groundtruth[:,:,0]>0] = np.minimum(blended[groundtruth[:,:,0]>0]+(0,100,0), 255) #(0,255,0) green for groundtruth
        
        only_pred[tres<threshold] = np.minimum(only_pred[tres<threshold]+(100,0,0), 255) #(255,0,0) red for detections
        blended[tres<threshold] = np.minimum(blended[tres<threshold]+(100,0,0), 255) #(255,0,0) red for detections
        
        blended = (blended-blended.min())/(blended.max()-blended.min())
        only_pred = (only_pred-only_pred.min())/(only_pred.max()-only_pred.min())
        only_gt = (only_gt-only_gt.min())/(only_gt.max()-only_gt.min())
        
    return blended, tres_embedded, res, only_pred, only_gt

def eval_image(input_img, gt, stepsize=512, resize_shortest=True, verbose=False):
    if not gt is None:
        gt = gt.copy()
        gt[gt==255]=1
        
    if verbose:
        print('input_img.shape', input_img.shape)
        print(input_img.min(), input_img.max())
    plt.figure(figsize=(10,10))
    plt.imshow(input_img, cmap='gray')

    testinp = input_img.reshape(1,input_img.shape[0],input_img.shape[1],1)

    if verbose:
        print(testinp.shape)
        print('min', testinp.min(), 'max', testinp.max())
    blended, tres_embedded, res, only_pred, only_gt = predict(testinp, gt, threshold=0.5, stepsize=stepsize, resize_shortest=resize_shortest, verbose=verbose)
    plt.figure(figsize=(10,10))
    plt.imshow(blended)
    
    #color overlay
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('prediction')
    plt.imshow(only_pred)

    plt.subplot(1,2,2)
    plt.title('groundtruth')
    plt.imshow(only_gt) 
    
    #binary
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('prediction')
    plt.imshow(tres_embedded, cmap='gray')

    if not gt is None:
        plt.subplot(1,2,2)
        plt.title('groundtruth')
        plt.imshow(gt[:,:,1], cmap='gray')
    
    res[res>0.5] = 1
    res[res<=0.5] = 0
    res = res.astype(np.uint8)
    if not gt is None:
        print('jaccard', jaccard_index_single(res[0,:,:,0], gt[:,:,0], verbose=verbose)[0])