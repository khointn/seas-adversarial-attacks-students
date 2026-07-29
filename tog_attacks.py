"""Projected-gradient TOG attacks for compatible object-detector attack models."""

from __future__ import annotations

from typing import Literal
from tqdm import tqdm
import os
from pathlib import Path
import random
import numpy as np

from tog_model import YOLOv3TOGModel
from utils import generate_attack_targets


def _validate_attack_inputs(image: np.ndarray, num_iterations: int, epsilon: float, step_size: float) -> None:
    """Validate common attack parameters before running iterative updates."""
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError("image must have shape (batch, height, width, 3).")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive.")
    if epsilon < 0 or step_size <= 0:
        raise ValueError("epsilon must be non-negative and step_size must be positive.")


def _initialize_adversarial_image(image: np.ndarray, epsilon: float) -> np.ndarray:
    """Randomly initialize an adversarial image inside the L-infinity constraint."""
    perturbation = np.random.uniform(-epsilon, epsilon, size=image.shape)
    return # TODO: clip the image + perturbation by 0.0 and 1.0. Use np.clip


def _project(image: np.ndarray, adversarial_image: np.ndarray, epsilon: float) -> np.ndarray:
    """Project an adversarial image into the valid pixel and L-infinity ranges."""
    perturbation = np.clip(adversarial_image - image, -epsilon, epsilon)
    return # TODO: clip the image + perturbation by 0.0 and 1.0. Use np.clip


def tog_vanishing(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Generate a TOG object-vanishing adversarial image.

    Repeat the following for ``num_iterations``:
       - Compute the object-vanishing gradient.
       - Use the sign of the gradient to update the image in the direction
         that minimizes the vanishing objective.
       - Project the result back into the epsilon-constrained region and
         the valid pixel range ``[0, 1]``.
    """
    _validate_attack_inputs(image, num_iterations, epsilon, step_size)
    adversarial_image = _initialize_adversarial_image(image, epsilon)
    for _ in range(num_iterations):
        # TODO: update the adversarial image by calculating the object vanishing gradiant
    return adversarial_image


def tog_fabrication(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Generate a TOG object-fabrication adversarial image.

    Repeat the following for ``num_iterations``:
       - Compute the object-fabrication gradient.
       - Update the image using the sign of the gradient in the direction
         that minimizes the fabrication objective.
       - Project the image back into the valid L-infinity and pixel ranges.

    The structure is similar to the vanishing attack, but it must call the
    fabrication-gradient method provided by ``victim``.
    """
    _validate_attack_inputs(image, num_iterations, epsilon, step_size)
    # create random noise for image
    adversarial_image = # TODO: initialize the random noisy image
    for _ in range(num_iterations):
        # TODO: student fill here
    return adversarial_image


def tog_mislabeling(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    target: Literal["most_likely", "least_likely"],
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Generate a targeted TOG object-mislabeling adversarial image.

    Your implementation should:

    1. Run the detector once on the original benign image.
    2. Convert the benign detections into targeted detections using
       ``generate_attack_targets()``.
       - ``"most_likely"`` selects a plausible alternative class.
       - ``"least_likely"`` selects the class with the lowest class score.
    3. Keep these target detections fixed throughout the attack.
    4. Randomly initialize the adversarial image inside the epsilon bound.
    5. For every iteration:
       - Compute the mislabeling gradient using the fixed target detections.
       - Apply a signed-gradient update.
       - Project the result into the valid perturbation and pixel ranges.

    Do not run target generation again inside the iterative loop.
    """
    # TODO: student fill here
    adversarial_image = _initialize_adversarial_image(image, epsilon)
    for _ in range(num_iterations):
        # TODO: student fill here
    return adversarial_image


def tog_untargeted(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
        """Generate an untargeted TOG adversarial image.

    Your implementation should:

    1. Run the detector once on the original benign image.
    2. Raise a clear ``ValueError`` if no benign objects are detected.
    3. Keep the original detections fixed during the iterative attack.
    4. Randomly initialize the adversarial image within the epsilon bound.
    5. For every iteration:
       - Compute the untargeted gradient using the fixed benign detections.
       - Apply the signed-gradient update expected by the provided attack
         model.
       - Project the result into the valid L-infinity and pixel ranges.

    The gradient method already defines the correct untargeted objective.
    """
    # TODO: student fill here
    adversarial_image = _initialize_adversarial_image(image, epsilon)
    for _ in range(num_iterations):
        # TODO: student fill here
    return adversarial_image


def tog_universal(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
    data_path: str = '',
    n_train_samples: int = 100
) -> np.ndarray:
    # read data and select number training samples
    fpaths_train = []
    fpaths_train += [os.path.join(data_path, '%s' % file.name) for file in Path(data_path).iterdir()]
    
    # TODO: randomly shuffle and select n_train_samples data to train

    for _ in range(num_iterations):
        pbar = tqdm(fpaths_train) # for tracking the training progress
        pbar.set_description('Epoch %d/%d' % (num_iterations + 1, num_iterations))

        # TODO: Implement the main algorithm for TOG universal here. Can try with object vanishing first.
        
    return # TODO: return the trained noise