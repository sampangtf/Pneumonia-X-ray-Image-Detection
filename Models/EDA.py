import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
import os
from sklearn.decomposition import PCA
from math import ceil
import seaborn as sns

# directory
normal_train_dir = 'Project/chest_xray/train/NORMAL'
pneu_train_dir = 'Project/chest_xray/train/PNEUMONIA'
normal_test_dir = 'Project/chest_xray/test/NORMAL'
pneu_test_dir = 'Project/chest_xray/test/PNEUMONIA'

# get the list of jpegs from sub image class folders
normal_train_list = [file_name for file_name in os.listdir(f'{normal_train_dir}') if file_name.endswith('.jpeg')]
pneu_train_list = [file_name for file_name in os.listdir(f'{pneu_train_dir}') if file_name.endswith('.jpeg')]
normal_test_list = [file_name for file_name in os.listdir(f'{normal_test_dir}') if file_name.endswith('.jpeg')]
pneu_test_list = [file_name for file_name in os.listdir(f'{pneu_test_dir}') if file_name.endswith('.jpeg')]


# Sample Images
def sample_images(num_imgs, normal_list, pneu_list, resize=False, size=None):
    np.random.seed(0)
    class_img = int(num_imgs / 2)
    select_norm = np.random.choice(normal_list, class_img, replace=False)
    select_pneu = np.random.choice(pneu_list, class_img, replace=False)
    fig = plt.figure()
    for i in range(num_imgs):
        if i < class_img:
            fp = f'{normal_train_dir}/{select_norm[i]}'
            label = 'NORMAL'
        else:
            fp = f'{pneu_train_dir}/{select_pneu[i - class_img]}'
            label = 'PNEUMONIA'
        fig.add_subplot(2, 3, i + 1)
        if resize:
            fn = image.load_img(fp, target_size=(size, size), color_mode='grayscale')
        else:
            fn = image.load_img(fp, color_mode='grayscale')
        plt.imshow(fn, cmap='Greys_r')
        plt.title(label)
        plt.axis('off')
        plt.savefig('Project/Plot/Sample_Img_Original')
    plt.show()


sample_images(6, normal_train_list, pneu_train_list)
sample_images(6, normal_train_list, pneu_train_list, resize=True, size=100)


# Load Images into matrix
def load_images(path, filenames, size=(64, 64)):
    images = np.zeros([0, size[0] * size[1]])
    for filename in filenames:
        filepath = path + '/' + filename
        img = image.load_img(filepath, target_size=size, color_mode='grayscale')
        img_flat = image.img_to_array(img)
        img_flat = [img_flat.ravel()]
        images = np.concatenate((images, img_flat))
    return images


def load_data():
    normal_train_images = load_images(normal_train_dir, normal_train_list)
    pneu_train_images = load_images(pneu_train_dir, pneu_train_list)
    normal_test_images = load_images(normal_test_dir, normal_test_list)
    pneu_test_images = load_images(pneu_test_dir, pneu_test_list)
    return normal_train_images, pneu_train_images, normal_test_images, pneu_test_images

def mean_image(img_normal, img_pneu, title_normal='Normal', title_pneu='Pneumonia', size=(64, 64)):
    temp_img = (img_normal, img_pneu)
    temp_title = (title_normal, title_pneu)
    fig = plt.figure()
    for i in range(2):
        fig.add_subplot(1, 2, i + 1)
        mean_img = np.mean(temp_img[i], axis=0).reshape(size)
        plt.imshow(mean_img, vmin=0, vmax=255, cmap='Greys_r')
        plt.title(f'Mean {temp_title[i]}')
        plt.axis('off')
    plt.savefig(f'Project/Plot/Mean')
    plt.show()
    return mean_img

load_data()
mean_img = mean_image(normal_train_images, pneu_train_images)

corr = np.corrcoef(np.concatenate([normal_train_images, pneu_train_images], axis=0).T)
sns.heatmap(corr)
plt.title('Correlation Matrix')
plt.savefig(f'Project/Plot/Correlation_Matrix')
plt.show()
