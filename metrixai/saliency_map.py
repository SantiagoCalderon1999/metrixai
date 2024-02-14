import numpy as np
from metrixai.insertion import Insertion
from metrixai.deletion import Deletion

class SaliencyMap():
    """_summary_ Wrapper class around any saliency map which is passed
    """
    def __init__(self, saliency_map_array: np.ndarray, image: np.ndarray, class_index: int) -> None:
        r"""
        Args:
            saliency_map_tensor
        """
        self.saliency_map_tensor = saliency_map_array
        self.image = image
        self.class_index = class_index
        
    def compute_insertion_curve(self, verbose, divisions=10):
        insertion_metric = Insertion(self.saliency_map_tensor, self.image, self.class_index)
        return insertion_metric.compute_metric(divisions=divisions)
    
    def compute_insertion_auc(self, verbose, divisions=10):
        insertion_metric = Insertion(self.saliency_map_tensor, self.image, self.class_index)
        return insertion_metric.compute_curve(verbose=verbose, divisions=divisions)
    
    def compute_deletion_curve(self, verbose, divisions=10):
        insertion_metric = Deletion(self.saliency_map_tensor, self.image, self.class_index)
        return insertion_metric.compute_metric(divisions=divisions)
    
    def compute_deletion_auc(self, verbose, divisions=10):
        insertion_metric = Deletion(self.saliency_map_tensor, self.image, self.class_index)
        return insertion_metric.compute_curve(verbose=verbose, divisions=divisions)