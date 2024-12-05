from abc import ABC, abstractmethod
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class BaseExecuter(ABC):
    @abstractmethod
    def read_all_files(self):
        pass
    def read_a_file(self):
        pass
    def display_a_file(self):
        pass

class PointCloud(BaseExecuter):
    def __init__(self):
        self.pcs = []
        self.single_pc = None

    def read_a_file(self, pc_path):
        self.single_pc  = np.load(pc_path)

    def read_all_files(self, source_folder):
        for subfolder in sorted(os.listdir(source_folder)):
            subfolder_path = os.path.join(source_folder, subfolder)

            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.lower().endswith('pc.npy'):
                        pc_path = os.path.join(subfolder_path, file)
                        self.pcs.append(np.load(pc_path))

    def display_a_file(self, pc):
        depth_map = pc[2]  

        plt.imshow(depth_map, cmap='viridis')  
        plt.colorbar(label='Depth')
        plt.title("Depth Map")
        plt.axis('off')
        plt.show()


class MaskData(BaseExecuter):
    def __init__(self):
        self.masks = []
        self.single_mask = None
        self.resized_masks = []
        self.combined_resized_masks = []

    def read_a_file(self, mask_path):
        self.single_image = np.load(mask_path)
    
    def read_all_files(self, source_folder):
        for subfolder in sorted(os.listdir(source_folder)):
            subfolder_path = os.path.join(source_folder, subfolder)

            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.lower().endswith('mask.npy'):
                        mask_path = os.path.join(subfolder_path, file)
                        self.masks.append(np.load(mask_path))

    def display_a_file(self, msk):
        num_objects = msk.shape[0]
        fig, axes = plt.subplots(1, num_objects, figsize=(15, 5))

        for i in range(num_objects):
            axes[i].imshow(msk[i], cmap='gray')
            axes[i].set_title(f"Mask {i+1}")
            axes[i].axis('off')

        plt.show()

class ImageData(BaseExecuter):
    def __init__(self):
        self.images = []
        self.single_image = None
        self.resized_images = []

    def read_a_file(self, image_path):
        self.single_image = cv2.imread(image_path)
    
    def read_all_files(self, source_folder):
        for subfolder in sorted(os.listdir(source_folder)):
            subfolder_path = os.path.join(source_folder, subfolder)

            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.lower().endswith(('rgb.png', 'rgb.jpg', 'rgb.jpeg')):
                        image_path = os.path.join(subfolder_path, file)
                        self.images.append(cv2.imread(image_path))

    def display_a_file(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')  
        plt.show()


class YOLOSegmentation:
    def __init__(self, class_id=0):
        self.contours = []
        self.class_id = class_id       

    def process_masks(self, masks):
        for mask in masks:
            contours_list = []
            for i in range(mask.shape[0]):
                msk = mask[i]
                msk = msk.astype(np.uint8) * 255
                contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_list.append(contours)

            self.contours.append(contours_list)

    def display_a_file(self, img, cntrs):
        for cntr in cntrs:
            cv2.drawContours(img, cntr, -1, (0, 255, 0), 2)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.show()

class BBox:
    def __init__(self):
        pass

    def _get_bounding_box(self, ground_truth_map):
        try:
            y_indices, x_indices = np.where(ground_truth_map > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            H, W = ground_truth_map.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bbox = [x_min, y_min, x_max, y_max]
        except:
            bbox = [0,0,1,1]

        return bbox
    def get_bounding_box_list(self, masks):
        bbox_list = []
        for mask in masks:
            bbox_image = []
            if mask.ndim == 3:
                for i in range(mask.shape[0]):
                    bbox_image.append(self._get_bounding_box(mask[i]))
                else:
                    bbox_image.append(self._get_bounding_box(mask))

            bbox_list.append(bbox_image)

        return bbox_list
    

class Reshape:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def _resize_boolean_mask(self, mask):
        mask_uint8 = mask.astype(np.uint8)

        resized_mask_uint8 = cv2.resize(mask_uint8, (self.height, self.width), interpolation=cv2.INTER_NEAREST)

        resized_mask = resized_mask_uint8.astype(bool)

        return resized_mask
    
    def resize_image(self, source_folder):
        resized_image_list = []
        for subfolder in os.listdir(source_folder):
            subfolder_path = os.path.join(source_folder, subfolder)

            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.lower().endswith(('rgb.png', 'rgb.jpg', 'rgb.jpeg')):
                        image_path = os.path.join(subfolder_path, file)
                        image = cv2.imread(image_path)

                        if image is not None:
                            resized_image_list.append(cv2.resize(image, (self.height, self.width)))

        return resized_image_list

    def resize_mask(self, source_folder):
        """Resize masks to 256x256 pixels and save them."""
        resized_mask_list = []
        for subfolder in os.listdir(source_folder):
            subfolder_path = os.path.join(source_folder, subfolder)

            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.lower().endswith('mask.npy'):
                        mask_path = os.path.join(subfolder_path, file)
                        mask = np.load(mask_path)

                        if mask.ndim == 3:
                            num_channels, _, _ = mask.shape

                            resized_mask_zero = np.zeros((num_channels, self.height, self.width), dtype=bool)
                            for channel in range(num_channels):
                                resized_mask_zero[channel] = self._resize_boolean_mask(mask[channel])

                            resized_mask_list.append(resized_mask_zero)

        return resized_mask_list
    
    def combine_masks(self, resized_mask):
        combined_mask_list = []
        for i in resized_mask:
            combined_mask = np.any(i, axis=0)
            combined_mask_list.append(combined_mask)
        return combined_mask_list
            


def save_txt(seg, img, lbl_dir, cnt, type):
    for idx, masks in enumerate(seg):
        dest_txt_path = os.path.join(lbl_dir, type, f'{cnt}.txt')
        if os.path.exists(dest_txt_path):
            os.remove(dest_txt_path)

        height, width = img[idx].shape[:2]
        for contours in masks:
            with open(dest_txt_path, 'a') as f:
                for contour in contours:
                    contour = contour.flatten()
                    
                    f.write(f"{0} ")
                    for i in range(0, len(contour), 2):
                        x = contour[i] / width  
                        y = contour[i + 1] / height  
                        f.write(f"{x:.6f} {y:.6f} ")  
                    f.write("\n")  
        cnt += 1

    return cnt

def save_bb_txt(bbs, imgs, lbl_dir, cnt, type):
    for img, bb in zip(imgs, bbs):
        dest_txt_path = os.path.join(lbl_dir, type, f'{cnt}.txt')
        if os.path.exists(dest_txt_path):
            os.remove(dest_txt_path)

        height, width = img.shape[:2]
        txt = ""
        for box in bb:
            x1, y1, x2, y2 = box
            x_center = (x1 + (x2 - x1) / 2) / width
            y_center = (y1 + (y2 - y1) / 2) / height
            norm_width = (x2 - x1) / width
            norm_height = (y2 - y1) / height

            txt += f"0 {x_center} {y_center} {norm_width} {norm_height}\n"

        with open(dest_txt_path, "w") as f:
            f.write(txt)

        cnt += 1
    return cnt

class DatasetSaver:
    def __init__(self, base_dir, images_train, images_val, images_test, seg_train, seg_val, seg_test):
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, 'images')
        self.labels_dir = os.path.join(base_dir, 'labels')

        os.makedirs(os.path.join(self.images_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.images_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.images_dir, 'test'), exist_ok=True)
        
        os.makedirs(os.path.join(self.labels_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.labels_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.labels_dir, 'test'), exist_ok=True)  

        self.images_train = images_train
        self.images_val = images_val
        self.images_test = images_test
        self.seg_train = seg_train
        self.seg_val = seg_val
        self.seg_test = seg_test

    def save_images(self):
        counter = 0
        for img in self.images_train:
            train_img_path = os.path.join(self.images_dir, 'train', f'{counter}.jpg')
            cv2.imwrite(train_img_path, img)
            counter += 1

        for img in self.images_val:
            val_img_path = os.path.join(self.images_dir, 'val', f'{counter}.jpg')
            cv2.imwrite(val_img_path, img)
            counter += 1

        for img in self.images_test:
            test_img_path = os.path.join(self.images_dir, 'test', f'{counter}.jpg')
            cv2.imwrite(test_img_path, img)
            counter += 1

        print(f"Saved {len(self.images_train)} training images, "
              f"{len(self.images_val)} validation images, "
              f"{len(self.images_test)} testing images.")
    
    def generate_txt_from_mask(self):
        counter = 0
        counter = save_txt(self.seg_train, self.images_train, self.labels_dir, counter, 'train')
        counter = save_txt(self.seg_val, self.images_val, self.labels_dir, counter, 'val')
        counter = save_txt(self.seg_test, self.images_test, self.labels_dir, counter, 'test')

        print(f"Saved {len(self.seg_train)} training txt, "
              f"{len(self.seg_test)} validation txt, "
              f"{len(self.seg_val)} testing txt.")
        
    def generate_bb_txt_from_mask(self):
        counter = 0
        counter = save_bb_txt(self.seg_train, self.images_train, self.labels_dir, counter, 'train')
        counter = save_bb_txt(self.seg_val, self.images_val, self.labels_dir, counter, 'val')
        counter = save_bb_txt(self.seg_test, self.images_test, self.labels_dir, counter, 'test')

        print(f"Saved {len(self.seg_train)} training txt, "
              f"{len(self.seg_test)} validation txt, "
              f"{len(self.seg_val)} testing txt.")

class SAMDataset(Dataset):
    def __init__(self, dataset, processor, bboxes):
        self.dataset = dataset
        self.processor = processor
        self.bboxes = bboxes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        prompt = [self.bboxes[idx][0]]

        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs




        



                            



