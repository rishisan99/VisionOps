# app/backend/services/gradcam.py

import torch
import torch.nn as nn

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, class_score: torch.Tensor) -> torch.Tensor:
        """
        Returns CAM [H,W] normalized to [0,1]
        """
        self.model.zero_grad(set_to_none=True)
        class_score.backward(retain_graph=True)

        grads = self.gradients[0]   # [C,H,W]
        acts  = self.activations[0] # [C,H,W]

        weights = torch.mean(grads, dim=(1, 2))  # [C]
        cam = torch.zeros(acts.shape[1:], device=acts.device)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam
