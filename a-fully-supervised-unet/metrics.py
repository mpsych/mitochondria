from imports import *
import unet
from unet import execute_predict
from unet import improve_components

def keras_dice_coef(y_true, y_pred, smooth=10): #0.01
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)

def keras_jaccard_coef(y_true, y_pred, smooth=0.01):
    '''Average jaccard coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)

def keras_precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def keras_recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def keras_jaccard_distance_loss(y_true, y_pred, smooth=0.01): #avg jaccard index on batch
    '''Average jaccard coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return 1-K.mean( (intersection + smooth) / (union + smooth), axis=0)

def keras_focal_loss(target, output, gamma=2):
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output), axis=-1)

def keras_dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)

def keras_binary_crossentropy_mod(y_true, y_pred):
    raw_prediction=K.reshape(y_pred,[-1])
    gt = K.reshape(y_true, [-1])
    #supposed 2 is the ignored label
    
    indices = K.squeeze(K.tf.where(K.not_equal(gt,2)),1)
    
    gt = K.cast(K.gather(gt,indices), 'float32')
    prediction = K.gather(raw_prediction,indices)
    return K.mean(K.binary_crossentropy(gt, prediction), axis=-1)

def confusion_matrix(pred, gt, thres=0.5):
    TP = np.sum((gt==1) & (pred>thres))
    FP = np.sum((gt==0) & (pred>thres))
    TN = np.sum((gt==0) & (pred<=thres))
    FN = np.sum((gt==1) & (pred<=thres))
    return (TP, FP, TN, FN)

def statistics(cmatrix):
    TP,FP,TN,FN = cmatrix
    jaccard_foreground = TP/(TP+FP+FN)
    jaccard_background = TN/(TN+FP+FN)
    VOC_score = (jaccard_foreground+jaccard_background)/2.
    accuracy = (TP+TN)/(TP+FP+TN+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return {'jaccard_foreground': jaccard_foreground, 'jaccard_background': jaccard_background, 
            'voc_score': VOC_score, 'accuracy': accuracy, 'precision': precision, 'recall': recall}
    
def PR_curve(pred, gt):
    test_mask_in = gt.copy().astype(np.byte)
    precs, recs, thres = precision_recall_curve(test_mask_in.flatten(), pred.flatten(), pos_label=1)

    res = plt.figure(figsize=(7,7))
    res = plt.title('AUC = ' + str(auc(recs,precs)))
    res = plt.plot(recs, precs, color='orange')
    res = plt.xlabel('recall')
    res = plt.ylabel('precision')
    res = plt.xlim(0,1)
    res = plt.ylim(0,1)    
    
def error_distribution(model, test_in, test_mask_in, stepsize=512, resize=True, extensive=False, figsize=(10,8)):
    pred = execute_predict(model, test_in, stepsize, resize)
    pred = improve_components(pred)
    pred[pred>0.5] = 1
    pred[pred<=0.5] = 0
        
    fp = np.sum((test_mask_in==0) & (pred==1), axis=0)[:,:,0]
    fn = np.sum((test_mask_in==1) & (pred==0), axis=0)[:,:,0]
    
    plt.figure(figsize=figsize)
    plt.imshow(fp)
    plt.title('distribution of errors, fp')
    plt.colorbar()
   
    plt.figure(figsize=figsize)
    plt.imshow(fn)
    plt.title('distribution of errors, fn')
    plt.colorbar()

    plt.figure(figsize=figsize, dpi=300)
    plt.imshow(fn+fp)
    plt.title('distribution of errors, overall')
    plt.colorbar()
    plt.savefig('output.png', format='png', bbox_inches='tight')
    
    if extensive:
        print('extensive analysis...')
        desc = ['width', 'height', 'stack']
        axis_analyze = [(0,1,3),(0,2,3),(1,2,3)]
        
        for i in range(len(desc)):
            axis = axis_analyze[i]
            tp_col = np.sum((test_mask_in==1) & (pred==1), axis=axis)
            fp_col = np.sum((test_mask_in==0) & (pred==1), axis=axis)
            fn_col = np.sum((test_mask_in==1) & (pred==0), axis=axis)
            normalizer = (tp_col+fp_col+fn_col)
            normalizer[normalizer==0] = 1e7
            vals = tp_col/normalizer.astype(float)
            plt.figure(figsize=figsize)
            plt.title('jaccard index across ' + desc[i] + ', mean=' + str(np.mean(vals))) 
            plt.plot(range(len(vals)), vals)
            plt.show()
                     
def error_borders(model, test_in, test_mask_in, stepsize=512, resize=True, extensive=False, figsize=(10,8)):
    pred = execute_predict(model, test_in, stepsize, resize)
    pred = improve_components(pred)
    pred[pred>0.5] = 1
    pred[pred<=0.5] = 0
    distances = []
    
    for i in range(test_in.shape[0]):
        this_img = test_mask_in[i,:,:,0]
        this_pred = test_in[i,:,:,0]
        
        gt_mitochondria = np.where(this_img>0)
        gt_mitochondria = np.array(gt_mitochondria)
        
        errs = np.where(this_img!=this_pred)
        #measure error distance
        errs = np.array(errs) #(2, N)
      
        for j in range(errs.shape[1]):
            x = errs[0,j]
            y = errs[1,j]
            #find distance to nearest GT pixel
            dist = (x-gt_mitochondria[0])**2+(y-gt_mitochondria[1])**2
            min_dist = np.sqrt(np.min(dist))
            distances.append(min_dist)
            
        plt.figure()
        plt.hist(distances)
        plt.show()
            
    return distances