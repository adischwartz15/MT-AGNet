"""Deep architecture analysis: gradient interference and representation similarity.

Two independent analyses, both computed from real forward/backward passes
on a trained checkpoint (never fabricated):

1. **Gradient cosine similarity** between the age-loss gradient and the
   gender-loss gradient with respect to the *shared backbone* parameters.
   Positive => aligned task gradients; negative => conflicting gradients
   (negative transfer risk); near-zero => weak relationship. Only
   meaningful for shared-backbone architectures (Experiments B/C/D) -- for
   independent backbones (Experiment A) there is no shared parameter set
   to compare, so this function raises for that architecture.

2. **Linear CKA** (Kornblith et al., 2019) between the shared embedding
   ``z`` and each task adapter's output, quantifying how much each
   adapter's transformation moves the representation away from the
   shared backbone's output.
"""

from __future__ import annotations

import numpy as np
import torch

from src.losses.quantile_loss import multi_quantile_pinball_loss
from src.models.multitask_model import MultiTaskFaceModel


def compute_gradient_cosine_similarity(
    model: MultiTaskFaceModel, dataloader, device: str, max_batches: int = 30
) -> np.ndarray:
    """Return an array of per-batch cosine similarities between grad_age and grad_gender.

    Only batches containing at least one age-labeled and one
    gender-labeled sample are used (both losses must be well-defined).
    """
    if model.architecture == "separate":
        raise ValueError(
            "Gradient interference is only defined for shared-backbone architectures "
            "(shared_no_adapters / shared_adapters); Experiment A uses independent backbones."
        )

    model.eval()
    similarities = []
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        age_mask = batch["age_mask"]
        gender_mask = batch["gender_mask"]
        if not age_mask.any() or not gender_mask.any():
            continue

        images = batch["image"].to(device)
        age_target = batch["age"].to(device)
        age_mask = age_mask.to(device)
        gender_target = batch["gender_label"].to(device)
        gender_mask = gender_mask.to(device)

        model.zero_grad(set_to_none=True)
        outputs = model(images)
        age_loss = multi_quantile_pinball_loss(
            outputs["age_output"]["q10_raw"], outputs["age_output"]["q50_raw"],
            outputs["age_output"]["q90_raw"], age_target, age_mask,
        )
        age_loss.backward(retain_graph=True)
        grad_age = torch.cat([p.grad.detach().flatten().clone() for p in backbone_params if p.grad is not None])

        model.zero_grad(set_to_none=True)
        gender_logits = outputs["gender_logits"]
        per_sample = torch.nn.functional.cross_entropy(gender_logits, gender_target, reduction="none")
        gender_loss = (per_sample * gender_mask.float()).sum() / gender_mask.float().sum()
        gender_loss.backward()
        grad_gender = torch.cat([p.grad.detach().flatten().clone() for p in backbone_params if p.grad is not None])

        cos_sim = torch.nn.functional.cosine_similarity(grad_age.unsqueeze(0), grad_gender.unsqueeze(0)).item()
        similarities.append(cos_sim)

    model.zero_grad(set_to_none=True)
    return np.array(similarities)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment similarity between two feature matrices.

    ``x``, ``y`` are (n_samples, n_features) arrays with matching n_samples.
    Returns a value in [0, 1] (numerically may slightly exceed due to
    floating point); 1 means the representations are identical up to
    rotation/isotropic scaling, 0 means completely dissimilar.
    """
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(y.T @ x, ord="fro") ** 2
    denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    if denom < 1e-12:
        return float("nan")
    return float(numerator / denom)


def compute_representation_similarity(
    shared_embeddings: np.ndarray, age_embeddings: np.ndarray, gender_embeddings: np.ndarray
) -> dict:
    """CKA between shared z and each adapter output, and between the two adapter outputs."""
    return {
        "cka_shared_vs_age_adapter": linear_cka(shared_embeddings, age_embeddings),
        "cka_shared_vs_gender_adapter": linear_cka(shared_embeddings, gender_embeddings),
        "cka_age_vs_gender_adapter": linear_cka(age_embeddings, gender_embeddings),
    }


def reduce_embeddings(embeddings: np.ndarray, method: str = "pca", n_components: int = 2, seed: int = 42) -> np.ndarray:
    """Dimensionality-reduce embeddings to 2D for visualization only (no causal claims)."""
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components, random_state=seed).fit_transform(embeddings)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(5, len(embeddings) // 4))
        return TSNE(n_components=n_components, random_state=seed, perplexity=perplexity, init="pca").fit_transform(embeddings)
    raise ValueError(f"Unknown reduction method '{method}', expected 'pca' or 'tsne'")


@torch.no_grad()
def extract_embeddings(model: MultiTaskFaceModel, dataloader, device: str, max_samples: int = 2000) -> dict:
    """Extract shared/age/gender embeddings plus labels for representation analysis."""
    model.eval()
    shared, age_emb, gender_emb = [], [], []
    ages, age_masks, genders, gender_masks = [], [], [], []
    n = 0
    for batch in dataloader:
        images = batch["image"].to(device)
        out = model.encode(images)
        if out["shared_embedding"] is not None:
            shared.append(out["shared_embedding"].cpu().numpy())
        age_emb.append(out["age_embedding"].cpu().numpy())
        gender_emb.append(out["gender_embedding"].cpu().numpy())
        ages.append(batch["age"].numpy())
        age_masks.append(batch["age_mask"].numpy())
        genders.append(batch["gender_label"].numpy())
        gender_masks.append(batch["gender_mask"].numpy())
        n += len(images)
        if n >= max_samples:
            break

    return {
        "shared_embedding": np.concatenate(shared) if shared else None,
        "age_embedding": np.concatenate(age_emb),
        "gender_embedding": np.concatenate(gender_emb),
        "age": np.concatenate(ages),
        "age_mask": np.concatenate(age_masks),
        "gender_label": np.concatenate(genders),
        "gender_mask": np.concatenate(gender_masks),
    }
