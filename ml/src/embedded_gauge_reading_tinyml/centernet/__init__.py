"""CenterNet (Objects as Points) package for gauge center detection.

Implements the Zhou et al. 2019 architecture adapted for single-class
gauge dial center keypoint detection, used as a parent/teacher model
for knowledge distillation into MobileNetV2 student models.
"""

from embedded_gauge_reading_tinyml.centernet._model import (
    CenterNetConfig,
    build_centernet_resnet50,
    build_centernet_mobilenetv2_student,
)
from embedded_gauge_reading_tinyml.centernet._losses import (
    centernet_loss,
    centernet_kd_loss,
    modified_focal_loss,
    l1_offset_loss,
)
from embedded_gauge_reading_tinyml.centernet._targets import (
    GaussianTargetConfig,
    gaussian_radius_from_bbox,
    draw_gaussian_heatmap,
    build_centernet_targets,
)
from embedded_gauge_reading_tinyml.centernet._decode import (
    decode_centernet_heatmap,
    decode_centernet_batch,
    centernet_nms,
    heatmap_to_canvas_coords,
    canvas_to_source_coords,
)
from embedded_gauge_reading_tinyml.centernet._data import (
    GeometryManifestRow,
    load_geometry_manifest,
    build_centernet_tf_dataset,
)

__all__ = [
    "CenterNetConfig",
    "build_centernet_resnet50",
    "build_centernet_mobilenetv2_student",
    "centernet_loss",
    "centernet_kd_loss",
    "modified_focal_loss",
    "l1_offset_loss",
    "GaussianTargetConfig",
    "gaussian_radius_from_bbox",
    "draw_gaussian_heatmap",
    "build_centernet_targets",
    "decode_centernet_heatmap",
    "decode_centernet_batch",
    "centernet_nms",
    "heatmap_to_canvas_coords",
    "canvas_to_source_coords",
    "GeometryManifestRow",
    "load_geometry_manifest",
    "build_centernet_tf_dataset",
]
