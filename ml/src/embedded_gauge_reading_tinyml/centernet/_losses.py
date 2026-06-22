"""Loss functions for CenterNet gauge center detection.

Implements the modified focal loss for heatmap keypoint detection and
L1 offset regression loss, following the Objects as Points (Zhou et al. 2019).
Also provides a knowledge-distillation loss for teacher→student training.
"""

from __future__ import annotations

import keras
import tensorflow as tf


def modified_focal_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> tf.Tensor:
    """Modified focal loss for keypoint heatmaps (CenterNet eq. 1).

    For each pixel:
      - If y_true == 1 (at center):  (1 - y_pred)^alpha * log(y_pred)
      - If y_true < 1  (elsewhere):  (1 - y_true)^beta * y_pred^alpha * log(1 - y_pred)

    The (1 - y_true)^beta term reduces penalty near the Gaussian peak,
    allowing the model to predict slightly off-center without heavy penalty.

    Args:
        y_true: Ground truth heatmap, shape (B, H, W, 1), values in [0, 1].
        y_pred: Predicted heatmap, shape (B, H, W, 1), after sigmoid.
        alpha: Focal loss focusing parameter (default 2.0).
        beta: Gaussian penalty reduction exponent (default 4.0).

    Returns:
        Scalar loss averaged over all pixels.
    """
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    # Positive locations (center peak).
    pos_mask = tf.cast(tf.equal(y_true, 1.0), tf.float32)
    pos_loss = -pos_mask * tf.pow(1.0 - y_pred, alpha) * tf.math.log(y_pred)

    # Negative locations (including Gaussian decay around center).
    neg_mask = tf.cast(tf.less(y_true, 1.0), tf.float32)
    neg_loss = (
        -neg_mask
        * tf.pow(1.0 - y_true, beta)
        * tf.pow(y_pred, alpha)
        * tf.math.log(1.0 - y_pred)
    )

    # Normalize by number of objects (at least 1 to avoid div-by-zero).
    num_objects = tf.maximum(tf.reduce_sum(pos_mask), 1.0)
    loss = tf.reduce_sum(pos_loss + neg_loss) / num_objects
    return loss


def l1_offset_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    heatmap_true: tf.Tensor,
) -> tf.Tensor:
    """L1 loss for center offset regression, applied only at center locations.

    Following CenterNet eq. 2: the offset is supervised only at the
    integer peak location p, and the loss is L1.

    Args:
        y_true: Ground truth offset map, shape (B, H, W, 2).
        y_pred: Predicted offset map, shape (B, H, W, 2).
        heatmap_true: Ground truth heatmap used to mask the loss to
                      only the peak location, shape (B, H, W, 1).

    Returns:
        Scalar offset loss averaged over objects.
    """
    # Mask: only compute loss where the heatmap has value >= 1.0 (peak).
    pos_mask = tf.cast(tf.greater_equal(heatmap_true, 0.99), tf.float32)
    pos_mask = tf.tile(pos_mask, [1, 1, 1, 2])  # broadcast to 2-channel offset

    abs_diff = tf.abs(y_true - y_pred)
    loss = tf.reduce_sum(abs_diff * pos_mask)

    num_objects = tf.maximum(
        tf.reduce_sum(pos_mask[:, :, :, 0]), 1.0
    )
    return loss / num_objects


def centernet_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    heatmap_weight: float = 1.0,
    offset_weight: float = 1.0,
    focal_alpha: float = 2.0,
    focal_beta: float = 4.0,
) -> tf.Tensor:
    """Combined CenterNet loss: focal heatmap + L1 offset regression.

    Args:
        y_true: Concatenated target (B, H, W, 3) — [:,:,:1]=heatmap, [:,:,1:3]=offset.
        y_pred: Concatenated prediction (B, H, W, 3).
        heatmap_weight: Weight for the heatmap focal loss.
        offset_weight: Weight for the offset regression loss.
        focal_alpha: Alpha for modified focal loss.
        focal_beta: Beta for modified focal loss.

    Returns:
        Scalar loss tensor.
    """
    # Split concatenated tensor: channel 0=heatmap, channels 1-2=offset.
    hm_true = y_true[..., 0:1]
    off_true = y_true[..., 1:3]
    hm_pred = y_pred[..., 0:1]
    off_pred = y_pred[..., 1:3]

    hm_loss = modified_focal_loss(
        hm_true, hm_pred, alpha=focal_alpha, beta=focal_beta,
    )
    off_loss = l1_offset_loss(off_true, off_pred, hm_true)

    return heatmap_weight * hm_loss + offset_weight * off_loss


def centernet_kd_loss(
    student_out: tf.Tensor,
    target: tf.Tensor,
    teacher_out: tf.Tensor | None = None,
    *,
    temperature: float = 4.0,
    kd_heatmap_weight: float = 0.5,
    kd_offset_weight: float = 0.1,
    hard_heatmap_weight: float = 0.5,
    hard_offset_weight: float = 1.0,
    focal_alpha: float = 2.0,
    focal_beta: float = 4.0,
) -> tf.Tensor:
    """Standalone knowledge distillation loss for CenterNet student training.

    All tensors are concatenated (B, H, W, 3): [:,:,:1]=heatmap, [:,:,1:3]=offset.

    Args:
        student_out: Student prediction (B, H, W, 3).
        target: Ground truth (B, H, W, 3).
        teacher_out: Teacher prediction (B, H, W, 3), or None for hard-only.
        temperature: Softening temperature for heatmap KD.
        kd_heatmap_weight: Weight of the KD heatmap KL term.
        kd_offset_weight: Weight of the KD offset L2 term.
        hard_heatmap_weight: Weight of the hard heatmap focal term.
        hard_offset_weight: Weight of the hard offset L1 term.
        focal_alpha: Focal loss alpha.
        focal_beta: Focal loss beta.

    Returns:
        Scalar loss tensor.
    """
    hm_student = student_out[..., 0:1]
    off_student = student_out[..., 1:3]
    hm_true = target[..., 0:1]
    off_true = target[..., 1:3]

    # Hard (ground truth) losses.
    hard_hm = modified_focal_loss(
        hm_true, hm_student, alpha=focal_alpha, beta=focal_beta,
    )
    hard_off = l1_offset_loss(off_true, off_student, hm_true)
    total = hard_heatmap_weight * hard_hm + hard_offset_weight * hard_off

    # KD (soft) losses.
    if teacher_out is not None:
        hm_teacher = teacher_out[..., 0:1]
        off_teacher = teacher_out[..., 1:3]

        T = temperature
        eps = tf.keras.backend.epsilon()
        teacher_soft = tf.nn.softmax(
            tf.reshape(hm_teacher, [tf.shape(hm_teacher)[0], -1]) / T
        )
        student_soft = tf.nn.log_softmax(
            tf.reshape(hm_student, [tf.shape(hm_student)[0], -1]) / T
        )
        kd_hm = tf.reduce_mean(
            tf.reduce_sum(
                teacher_soft * (tf.math.log(teacher_soft + eps) - student_soft),
                axis=-1,
            )
        ) * (T * T)

        teacher_peak_mask = tf.cast(
            tf.greater_equal(hm_teacher, 0.5), tf.float32
        )
        teacher_peak_mask = tf.tile(teacher_peak_mask, [1, 1, 1, 2])
        off_diff = tf.square(off_student - off_teacher) * teacher_peak_mask
        num_peaks = tf.maximum(
            tf.reduce_sum(teacher_peak_mask[:, :, :, 0]), 1.0
        )
        kd_off = tf.reduce_sum(off_diff) / num_peaks

        total += kd_heatmap_weight * kd_hm + kd_offset_weight * kd_off

    return total
