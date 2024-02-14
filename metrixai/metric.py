import torch

class Metric:
    
    def __init__(self, saliency_map : Callable) -> None:
        r"""
        Args:
            forward_func (Callable or torch.nn.Module): This can either be an instance
                        of pytorch model or any modification of model's forward
                        function.
        """
        self.forward_func = forward_func