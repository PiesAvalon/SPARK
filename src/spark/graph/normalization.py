"""Normalization hooks for operator names and graph labels."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import NodeLabel


@dataclass(slots=True)
class LabelNormalizer:
    alias_map: dict[str, str] = field(
        default_factory=lambda: {
            "conv1d": "Conv",
            "conv2d": "Conv",
            "conv3d": "Conv",
            "convolution": "Conv",
            "batchnorm1d": "BatchNorm",
            "batchnorm2d": "BatchNorm",
            "batchnorm3d": "BatchNorm",
            "relu6": "ReLU",
            "leakyrelu": "LeakyReLU",
            "maxpool1d": "MaxPool",
            "maxpool2d": "MaxPool",
            "maxpool3d": "MaxPool",
            "avgpool1d": "AvgPool",
            "avgpool2d": "AvgPool",
            "avgpool3d": "AvgPool",
            "adaptiveavgpool2d": "AdaptiveAvgPool",
            "linear": "Linear",
            "matmul": "MatMul",
            "addmm": "MatMul",
            "layernorm": "LayerNorm",
            "dropout": "Dropout",
            "softmax": "Softmax",
            "gelu": "GELU",
        }
    )

    def normalize_operator_name(self, raw_name: str) -> str:
        stripped = raw_name.split(".")[-1].replace("_", "").replace("-", "").strip()
        lowered = stripped.lower()
        if lowered in self.alias_map:
            return self.alias_map[lowered]
        if stripped:
            return stripped[0].upper() + stripped[1:]
        return "UnknownOp"

    def normalize_label(self, label: NodeLabel) -> NodeLabel:
        normalized_attrs = tuple(sorted((str(key), str(value)) for key, value in label.attr_summary))
        return NodeLabel(
            op_type=self.normalize_operator_name(label.op_type),
            shape_summary=None if label.shape_summary is None else str(label.shape_summary),
            dtype_summary=None if label.dtype_summary is None else str(label.dtype_summary),
            attr_summary=normalized_attrs,
        )
