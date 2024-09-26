import concurrent.futures
import glob
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import math




err_not_np_img= "not a numpy array or list of numpy array" 
err_img_arr_empty="Image array is empty"
err_row_zero="No. of rows can't be <=0"
err_column_zero="No. of columns can't be <=0"
err_invalid_size="Not a valid size tuple (x,y)"
err_caption_array_count="Caption array length doesn't matches the image array length"

def is_numpy_array(x):
   
    return isinstance(x, np.ndarray)
def is_tuple(x):
    return type(x) is tuple
def is_list(x):
    return type(x) is list
def is_numeric(x):
    return type(x) is int
def is_numeric_list_or_tuple(x):
    for i in x:
        if not is_numeric(i):
            return False
    return True

def visualize(image_array, column=1, v_gap=0.1, h_gap=0.1, fig_size=(20,20), color_map=None, caption_array=-1):
    if not (is_tuple(fig_size) and len(fig_size)==2):
         raise Exception(err_invalid_size)
    if column<=0:
        raise Exception(err_column_zero)
    if(is_list(image_array)):
        for img in image_array:
            if not is_numpy_array(img):
                raise Exception(err_not_np_img)

        if caption_array!=-1:
            if len(caption_array)!=len(image_array):
                raise Exception(err_caption_array_count)
            
        column= math.ceil(column)
        row= math.ceil(len(image_array)/column)
        column= min(column, len(image_array))
        f,axes= plt.subplots(row, column, figsize=fig_size)
        f.subplots_adjust(hspace=h_gap, wspace=v_gap)

        n_row=0
        n_col=0
        index=0
        for img in image_array:
            if column==1:
                axes[n_row].imshow(img, cmap=color_map)
                if(caption_array!=-1):
                    axes[n_row].set_title(caption_array[index])
            elif row==1:
                axes[n_col].imshow(img, cmap=color_map)
                if(caption_array!=-1):
                    axes[n_col].set_title(caption_array[index])
            else:
                axes[n_row,n_col].imshow(img, cmap=color_map)
                if(caption_array!=-1):
                    axes[n_row,n_col].set_title(caption_array[index])
            n_col+=1
            if (n_col)%column==0:
                n_row+=1
                n_col=0
            index+=1
            if(n_row==row):
                break
#         plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if is_numpy_array(image_array):
        plt.figure(figsize = fig_size)
        plt.imshow(image_array, cmap=color_map)
        if(caption_array!=-1):
            plt.title(caption_array)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#         pad=0.4, w_pad=0.5, h_pad=1.0


def load_image(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (1280, 720))

def batch_images(image_paths, batch_size=100):
    """Yield successive batches from image paths."""
    for i in range(0, len(image_paths), batch_size):
        yield image_paths[i:i + batch_size]

def load_images(path, batch_size=10):
    """Load and process images from a given path in batches."""
    images = glob.glob(path)
    image_list = []

    for batch in batch_images(images, batch_size):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            processed_images = list(executor.map(load_image, batch))
            image_list.extend(filter(None, processed_images))

    return image_list


