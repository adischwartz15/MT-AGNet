"""Manual Grad-CAM implementation (no external Grad-CAM library).

Produces separate "model attention visualization" heatmaps for the age
prediction (using the q50 output as the scalar target) and the dataset
gender-label prediction (using the selected class logit). Grad-CAM here
is purely a gradient-weighted activation visualization -- it is not proof
of causality and does not explain human reasoning; all generated reports
must label it only as "Model attention visualization".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.models.multitask_model import MultiTaskFaceModel


class GradCAM:
    """Computes Grad-CAM heatmaps for a target convolutional layer."""

    def __init__(self, model: MultiTaskFaceModel, target_layer_name: str = "layer4") -> None:
        self.model = model
        self.target_layer_name = target_layer_name
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    def _get_target_module(self, task: str):
        backbone = self.model.age_backbone if task == "age" else self.model.gender_backbone
        return getattr(backbone, self.target_layer_name)

    def _forward_hook(self, module, inputs, output):
        self._activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def generate(self, image: torch.Tensor, task: str, target_class: int | None = None) -> dict:
        """Compute a Grad-CAM heatmap for a single image (shape [1, C, H, W]).

        Returns a dict with the normalized heatmap (H, W) in [0, 1] at the
        target layer's resolution, and the scalar target used (age q50, or
        the class index used for the gender logit).
        """
        self.model.eval()
        target_module = self._get_target_module(task)
        fwd_handle = target_module.register_forward_hook(self._forward_hook)
        bwd_handle = target_module.register_full_backward_hook(self._backward_hook)

        try:
            image = image.clone().requires_grad_(True)
            outputs = self.model(image)

            if task == "age":
                target_value = outputs["age_output"]["q50_raw"][0]
                scalar = target_value
                used_class = None
            elif task == "gender":
                logits = outputs["gender_logits"][0]
                used_class = target_class if target_class is not None else int(logits.argmax().item())
                scalar = logits[used_class]
            else:
                raise ValueError("task must be 'age' or 'gender'")

            self.model.zero_grad(set_to_none=True)
            scalar.backward()

            activations = self._activations[0]  # (C, H, W)
            gradients = self._gradients[0]  # (C, H, W)
            weights = gradients.mean(dim=(1, 2))  # (C,)

            cam = torch.einsum("c,chw->hw", weights, activations)
            cam = F.relu(cam)
            cam = cam - cam.min()
            max_val = cam.max()
            if max_val > 1e-8:
                cam = cam / max_val
            heatmap = cam.detach().cpu().numpy()

            return {
                "heatmap": heatmap,
                "task": task,
                "used_class": used_class,
                "scalar_value": float(scalar.detach().cpu().item()),
            }
        finally:
            fwd_handle.remove()
            bwd_handle.remove()


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a (h, w) heatmap to (H, W) = ``size`` using PIL bilinear resampling."""
    from PIL import Image

    img = Image.fromarray((heatmap * 255).astype(np.uint8))
    resized = img.resize(size, Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0
