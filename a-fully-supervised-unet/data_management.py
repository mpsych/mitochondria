from imports import *

def load_data(raw_path, mask_path):
    numimages = 0
    for j in range(len(raw_path)):
        input_dir = raw_path[j]
        files_raw = sorted(glob.glob(input_dir + '/*.png'))
        print('found', len(files_raw), 'images in', input_dir)
        numimages+=len(files_raw)
    print('found', numimages, 'in total.')

    imgdatas = np.ndarray((numimages,768,1024,1), dtype=np.float32)
    imglabels = np.ndarray((numimages,768,1024,1), dtype=np.float32)
    global_ct = 0
    for j in tqdm(range(len(raw_path))):
        input_dir = raw_path[j]
        input_dir_mask = mask_path[j]
        files_raw = sorted(glob.glob(input_dir + '/*.png'))
        files_mask = sorted(glob.glob(input_dir_mask + '/*.png'))

        for i in range(len(files_raw)):
            file_raw = files_raw[i]
            file_mask = files_mask[i]
            img = load_img(file_raw,grayscale = True)
            label = load_img(file_mask,grayscale = True)
            img = img_to_array(img)
            label = img_to_array(label)
            imgdatas[global_ct] = img/255.
            imglabels[global_ct] = label/255.
            global_ct+=1
    print(imgdatas.shape)
    print(imglabels.shape)
    return imgdatas, imglabels

def rotate_bound(image, angle, mode, fill_black=True):
    big_stack = []
    for img_index in range(image.shape[2]):
        # grab the dimensions of the image and then determine the center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
 
        if fill_black:
            bt = (0,0,255)
        else:
            bt = 2
        big_stack.append(cv2.warpAffine(image[:,:,img_index], M, (nW, nH), flags=mode, borderMode=cv2.BORDER_CONSTANT, borderValue=bt)) 
    big_stack = np.array(big_stack)
    big_stack = np.swapaxes(big_stack,0,1)
    big_stack = np.swapaxes(big_stack,1,2)
    return big_stack
        
def custom_preproc(img_in, mode, flip_z=False, rotate=False, mean=None, sd=None, fill_black=True, train_on_borders=False):
    #input is (768, 1024, d)
    if flip_z:
        if np.random.rand()<0.5:
            img_in = np.flip(img_in, 2)

    if not mean is None:
        img_in = img_in-mean
    if not sd is None:
        sd[sd==0]=1e-4
        img_in = img_in/sd

    m = None
    if mode=='raw':
        m = cv2.INTER_LINEAR
    elif mode=='mask':
        m = cv2.INTER_NEAREST 
   
    if rotate:
        img_in = rotate_bound(img_in, np.random.rand()*360, m, fill_black = fill_black)
        img_in = img_in.reshape(img_in.shape[0], img_in.shape[1], img_in.shape[2])

    MIN_SIZE = 512
    MAX_SIZE = max(MIN_SIZE+1,min(img_in.shape[0], img_in.shape[1]))
    size = np.random.randint(MIN_SIZE, MAX_SIZE)
    width = img_in.shape[0]
    height = img_in.shape[1]

    xposs = max(1,width-size)
    yposs = max(1,height-size)

    xcrop = np.random.randint(0,xposs)
    ycrop = np.random.randint(0,yposs)
    crop = img_in[xcrop:(xcrop+size), ycrop:(ycrop+size), :]
    
    #upsampling if necessary
    if crop.shape[0]==512 and crop.shape[1]==512:
        if mode=='mask':
            crop[(crop>0) & (crop<1.5)] = 1 
        
        if train_on_borders and mode=='mask':
            crop = mh.borders(crop).astype(int)
        return crop
    else: #upsampling
        big_stack = np.zeros((512,512,crop.shape[2]))
        for imgindex in range(crop.shape[2]):
            ref = cv2.resize(crop[:,:,imgindex], (512,512), interpolation=m)
            big_stack[:,:,imgindex] = ref
        
        if mode=='mask':
            big_stack[(big_stack>0) & (big_stack<1.5)]=1
            if train_on_borders:
                big_stack = mh.borders(big_stack).astype(int)
        return big_stack

def load_stack(imgdir, alphanum=False):
    files = list(sorted(glob.glob(os.path.join(imgdir, '*.png'))))
    st = []
    for i in tqdm(range(len(files))):
        ff = imgdir + '/' + str(i) + '.png' 
        if not alphanum: ff=files[i]
        input_img = misc.imread(ff)
        if len(input_img.shape)>2: input_img = input_img[:,:,0]
        input_img = input_img.astype(float)/255.
        testinp = input_img.reshape(input_img.shape[0],input_img.shape[1], 1)
        st.append(testinp)
    return np.array(st)