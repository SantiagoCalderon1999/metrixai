import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from sklearn import metrics

class Insertion:
    def __init__(self, saliency_map: np.ndarray, image: np.ndarray, class_index: np.ndarray) -> None:
        r"""
        Args:
            saliency_map_tensor
        """
        self.saliency_map = saliency_map
        self.image = image
        self.class_index = class_index

    def compute_curve(self, divisions, verbose=False):
        (divisions_list_in, conf_insertion_list) = self.compute_metric(divisions)
        auc_insertion = metrics.auc(divisions_list_in, conf_insertion_list)
        if (verbose):
            plt.plot(divisions_list_in, conf_insertion_list)
            plt.legend([f"Insertion, AUC: {auc_insertion}"])
            plt.show()
        return auc_insertion

    def compute_metric(self, divisions=10):
        saliency_map = self.saliency_map
        image = self.image

        masks = np.empty([saliency_map.shape[0], saliency_map.shape[1], 3])
        conf_insertion_list = []
        divisions_list_in = []
        minimum = np.min(saliency_map) - np.abs(
            (np.min(saliency_map) - np.max(saliency_map)) * 0.2
        )
        maximum = np.max(saliency_map) + np.abs(
            (np.min(saliency_map) - np.max(saliency_map)) * 0.1
        )
        for sub_index in range(0, divisions):
            masks[:, :, :] = False
            threshold = maximum + (sub_index / divisions) * (minimum - maximum)
            pixels = np.where(saliency_map >= threshold)
            masks[pixels[0], pixels[1], :] = True
            divisions_list_in.append(
                len(pixels[0]) / (saliency_map.shape[0] * saliency_map.shape[1])
            )
            min_expl = np.where(masks, image, 0)
            yolo = YOLO("yolov8n.pt")
            detection = yolo(min_expl, classes=self.class_index, verbose=False)
            conf = (
                detection[0].boxes.conf[0]
                if (len(detection[0].boxes.conf) > 0)
                else detection[0].boxes.conf
            )
            conf_insertion_list.append(conf)
        conf_insertion_list = [
            np.sum(conf.numpy()) if conf.numpy().size > 0 else 0
            for conf in conf_insertion_list
        ]
        return (divisions_list_in, conf_insertion_list)
