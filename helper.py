import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

class Helper:
    def __init__(self):
        pass

    def plot_yolo_boundaries_from_txt(self, image_path, txt_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        height, width, _ = image.shape

        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line_parts = line.strip().split()
            points = np.array(line_parts[1:], dtype=float).reshape(-1, 2)

            points[:, 0] *= width
            points[:, 1] *= height
            points = points.astype(np.int32)

            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def generate_mask(self, test_image_rgb, results):
        mask_list = []

        H, W, _ = test_image_rgb.shape

        for i, result in enumerate(results):
            for j, mask in enumerate(result.masks.data):
                mask = (mask.cpu().numpy() * 255).astype(np.uint8)
                mask = cv2.resize(mask, (W, H))

                mask_list.append(np.expand_dims(mask, axis=0))

        return np.stack(mask_list, axis=0).squeeze(1)
    
    def combine_masks(self, mask):
        combined_mask = np.zeros((mask.shape[1], mask.shape[2]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            combined_mask = np.logical_or(combined_mask, mask[i]).astype(np.uint8)
        return combined_mask
    
    
    def calculate_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0  
        return intersection / union
    
    def overlay_masks_on_image(self, image, masks):
        num_masks = masks.shape[0]
        colors = np.random.rand(num_masks, 3)  

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        overlay = np.zeros_like(image_rgb)

        for i in range(num_masks):
            mask = masks[i]  

            mask = (mask > 0).astype(np.uint8)  

            color = (colors[i] * 255).astype(np.uint8)  
            colored_mask = np.zeros_like(image_rgb)
            colored_mask[mask > 0] = color  
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)

        combined_image = cv2.addWeighted(image_rgb, 1, overlay, 1, 0)
        return combined_image
    
    def draw_yolo_bboxes(self, box, image, mask=None):
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        if mask:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(image_rgb)
            for b in box:
                x1, y1, x2, y2 = b
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
                axes[0].add_patch(rect)
            axes[0].set_title("Image with Bounding Boxes")

            axes[1].imshow(np.array(mask), cmap='gray')
            axes[1].set_title("Mask")

            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

            plt.show()

        else:
            plt.imshow(image_rgb)
            for b in box:
                x1, y1, x2, y2 = b
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
                plt.gca().add_patch(rect)  

            plt.title("Image with Bounding Boxes")  
            plt.axis('off') 
            plt.show()

