"""Adapters that connect released datasets and upstream tool projects to SPARK."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Protocol, Sequence

from ..core.types import ExecutionEnv, InputFeature, TestCase
from .manifest import ToolRunManifest

REPO_ROOT = Path(__file__).resolve().parents[3]

PROJECT_DIRS: dict[str, str] = {
    "comet": "COMET",
    "devmut": "DevMuT",
    "gandalf": "Gandalf",
    "gencog": "GenCoG",
    "lemon": "LEMON",
    "modelmeta": "ModelMeta",
    "muffin": "Muffin",
    "neuri": "neuri-artifact",
    "nnsmith": "nnsmith",
}

TOKEN_OPERATOR_ALIASES: dict[str, str] = {
    "abs": "Abs",
    "acos": "Acos",
    "acosh": "Acosh",
    "add": "Add",
    "addn": "AddN",
    "addv2": "AddV2",
    "argmax": "ArgMax",
    "argmin": "ArgMin",
    "avgpool": "AvgPool",
    "avgpool2d": "AvgPool2D",
    "batchnorm": "BatchNorm",
    "batchnorm2d": "BatchNorm2D",
    "conv": "Conv",
    "conv1d": "Conv1D",
    "conv2d": "Conv2D",
    "conv3d": "Conv3D",
    "dense": "Dense",
    "dropout": "Dropout",
    "flatten": "Flatten",
    "gelu": "GELU",
    "gru": "GRU",
    "input": "Input",
    "inputlayer": "InputLayer",
    "instancenorm": "InstanceNorm",
    "layernorm": "LayerNorm",
    "leakyrelu": "LeakyReLU",
    "linear": "Linear",
    "lstm": "LSTM",
    "matmul": "MatMul",
    "maxpool": "MaxPool",
    "maxpool2d": "MaxPool2D",
    "pad": "Pad",
    "pool": "Pool",
    "relu": "ReLU",
    "reshape": "Reshape",
    "sigmoid": "Sigmoid",
    "slice": "Slice",
    "softmax": "Softmax",
    "squeeze": "Squeeze",
    "stridedslice": "StridedSlice",
    "sub": "Sub",
    "tanh": "Tanh",
}

TOKEN_STOPWORDS = {
    "artifact",
    "batch",
    "bug",
    "case",
    "cases",
    "cfg",
    "cli",
    "config",
    "configs",
    "count",
    "cov",
    "coverage",
    "data",
    "dataset",
    "debug",
    "demo",
    "detect",
    "detection",
    "eval",
    "evaluate",
    "example",
    "executor",
    "exp",
    "experiments",
    "export",
    "filter",
    "gen",
    "generate",
    "generated",
    "generator",
    "history",
    "infer",
    "input",
    "legacy",
    "load",
    "localization",
    "logger",
    "main",
    "metric",
    "metrics",
    "model",
    "models",
    "mutation",
    "operators",
    "ops",
    "patch",
    "plot",
    "process",
    "project",
    "records",
    "release",
    "render",
    "report",
    "results",
    "run",
    "sample",
    "scripts",
    "selection",
    "src",
    "test",
    "tests",
    "tool",
    "train",
    "utils",
}

MODEL_FAMILY_TEMPLATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("alexnet", ("Conv2D", "ReLU", "MaxPool2D", "Conv2D", "ReLU", "MaxPool2D", "Flatten", "Dense")),
    ("crnn", ("Conv2D", "BatchNorm", "ReLU", "LSTM", "Dense")),
    ("densenet", ("Conv2D", "BatchNorm", "ReLU", "Conv2D", "Concat", "GlobalAvgPool", "Dense")),
    ("efficientnet", ("Conv2D", "BatchNorm", "SiLU", "DepthwiseConv2D", "Conv2D", "GlobalAvgPool", "Dense")),
    ("inception", ("Conv2D", "Conv2D", "Concat", "Conv2D", "GlobalAvgPool", "Dense")),
    ("lenet", ("Conv2D", "AvgPool2D", "Conv2D", "AvgPool2D", "Flatten", "Dense", "Dense")),
    ("lstm", ("InputLayer", "LSTM", "Dense")),
    ("mobilenet", ("Conv2D", "DepthwiseConv2D", "Conv2D", "GlobalAvgPool", "Dense")),
    ("openpose", ("Conv2D", "Conv2D", "Concat", "Conv2D")),
    ("patchcore", ("Conv2D", "BatchNorm", "ReLU", "Conv2D", "GlobalAvgPool")),
    ("resnet", ("Conv2D", "BatchNorm", "ReLU", "Conv2D", "Add", "ReLU", "GlobalAvgPool", "Dense")),
    ("srgan", ("Conv2D", "ResidualBlock", "ResidualBlock", "Upsample", "Conv2D")),
    ("ssd", ("Conv2D", "Conv2D", "Concat", "DetectionHead")),
    ("transformer", ("Embedding", "MatMul", "Add", "LayerNorm", "MatMul", "Softmax", "Dense")),
    ("unet", ("Conv2D", "Conv2D", "MaxPool2D", "Conv2D", "Concat", "Conv2D")),
    ("vgg", ("Conv2D", "ReLU", "Conv2D", "ReLU", "MaxPool2D", "Flatten", "Dense")),
    ("xception", ("Conv2D", "DepthwiseConv2D", "PointwiseConv2D", "GlobalAvgPool", "Dense")),
    ("yolo", ("Conv2D", "BatchNorm", "LeakyReLU", "Concat", "Conv2D", "DetectionHead")),
)


def _sorted_unique(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in sorted(paths):
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _safe_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sanitize_case_id(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    return compact.upper() or "CASE"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+[A-Za-z0-9]*", text.replace(".", "_").replace("-", "_"))


def _default_operator(token: str) -> str:
    return token[0].upper() + token[1:] if token else "UnknownOp"


def _infer_operator_sequence(parts: Sequence[str], fallback: str = "UnknownOp") -> tuple[str, ...]:
    sequence: list[str] = []
    for part in parts:
        for token in _tokenize(part):
            lowered = token.lower()
            if lowered in TOKEN_STOPWORDS:
                continue
            op_name = TOKEN_OPERATOR_ALIASES.get(lowered, _default_operator(token))
            if not sequence or sequence[-1] != op_name:
                sequence.append(op_name)
    return tuple(sequence[:12]) if sequence else (fallback,)


def _sequence_from_model_family(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    lowered = value.lower()
    for fragment, sequence in MODEL_FAMILY_TEMPLATES:
        if fragment in lowered:
            return sequence
    return ()


def _build_chain_payload(operators: Sequence[str]) -> dict[str, object] | None:
    if not operators:
        return None
    nodes = [{"id": f"n{index}", "op": op_name} for index, op_name in enumerate(operators)]
    edges = [{"src": f"n{index}", "dst": f"n{index + 1}"} for index in range(len(nodes) - 1)]
    return {"nodes": nodes, "edges": edges}


def _summarize_attrs(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {}
    keys = ("filters", "kernel_size", "strides", "padding", "activation", "units", "pool_size", "groups")
    return {key: raw[key] for key in keys if key in raw}


def _keras_graph_payload(payload: dict[str, object]) -> dict[str, object] | None:
    config = payload.get("config")
    if not isinstance(config, dict):
        return None
    layers = config.get("layers")
    if not isinstance(layers, list):
        return None

    nodes: list[dict[str, object]] = []
    edges: list[dict[str, str]] = []
    known_ids: set[str] = set()

    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            continue
        layer_name = str(layer.get("name", f"n{index}"))
        layer_config = layer.get("config")
        layer_config = layer_config if isinstance(layer_config, dict) else {}
        nodes.append(
            {
                "id": layer_name,
                "op": str(layer.get("class_name") or layer_config.get("name") or "UnknownOp"),
                "dtype": layer_config.get("dtype"),
                "shape": layer_config.get("batch_input_shape"),
                "attrs": _summarize_attrs(layer_config),
            }
        )
        known_ids.add(layer_name)

    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            continue
        layer_name = str(layer.get("name", f"n{index}"))
        inbound = layer.get("inbound_nodes")
        if not isinstance(inbound, list):
            continue
        for group in inbound:
            if not isinstance(group, list):
                continue
            for item in group:
                if isinstance(item, list) and item:
                    src = str(item[0])
                    if src in known_ids:
                        edges.append({"src": src, "dst": layer_name})

    return {"nodes": nodes, "edges": edges}


def _devmut_graph_payload(payload: dict[str, object]) -> dict[str, object] | None:
    raw_edges = payload.get("edges")
    sequence: list[str] = []
    if isinstance(raw_edges, list):
        for item in raw_edges:
            if not isinstance(item, list) or len(item) < 2:
                continue
            src = str(item[0])
            dst = str(item[1])
            if not sequence:
                sequence.append(src)
            elif sequence[-1] != src:
                sequence.append(src)
            sequence.append(dst)

    layer_config = payload.get("layer_config")
    attrs_by_op: dict[str, dict[str, object]] = {}
    if isinstance(layer_config, dict):
        for op_name, records in layer_config.items():
            if isinstance(records, list) and records and isinstance(records[0], dict):
                attrs_by_op[str(op_name)] = _summarize_attrs(records[0])

    if not sequence and attrs_by_op:
        sequence.extend(attrs_by_op.keys())
    if not sequence:
        return None

    nodes: list[dict[str, object]] = []
    edges: list[dict[str, str]] = []
    for index, op_name in enumerate(sequence):
        node = {"id": f"n{index}", "op": op_name}
        if op_name in attrs_by_op:
            node["attrs"] = attrs_by_op[op_name]
        nodes.append(node)
        if index > 0:
            edges.append({"src": f"n{index - 1}", "dst": f"n{index}"})
    return {"nodes": nodes, "edges": edges}


def _gandalf_graph_payload(payload: dict[str, object]) -> dict[str, object] | None:
    network = payload.get("network")
    if not isinstance(network, list):
        return None

    nodes: list[dict[str, object]] = []
    edges: list[dict[str, str]] = []
    aliases: dict[str, str] = {}
    previous_id: str | None = None

    for index, raw_node in enumerate(network):
        if not isinstance(raw_node, dict):
            continue
        node_id = str(raw_node.get("index") or f"n{index}")
        aliases[node_id] = node_id
        params = raw_node.get("params")
        nodes.append(
            {
                "id": node_id,
                "op": str(raw_node.get("name", "UnknownOp")),
                "attrs": params if isinstance(params, dict) else {},
            }
        )
        if previous_id is not None:
            edges.append({"src": previous_id, "dst": node_id})
        previous_id = node_id

    for index, raw_node in enumerate(network):
        if not isinstance(raw_node, dict):
            continue
        src = str(raw_node.get("index") or f"n{index}")
        branch_to = raw_node.get("branch_to")
        if isinstance(branch_to, str) and branch_to in aliases:
            edges.append({"src": src, "dst": aliases[branch_to]})

    return {"nodes": nodes, "edges": edges}


class ToolAdapter(Protocol):
    tool_name: str

    def discover_cases(self) -> Sequence[TestCase]:
        """Load cases from one dataset release entry or project-backed catalog slice."""


@dataclass(slots=True)
class BaseToolAdapter:
    manifest: ToolRunManifest
    tool_name: str = "generic"

    def project_dir_name(self) -> str | None:
        return PROJECT_DIRS.get(self.tool_name.lower())

    def project_patterns(self) -> tuple[str, ...]:
        return ()

    def case_kind(self) -> str:
        return "project-export"

    def project_root(self) -> Path | None:
        explicit = self.manifest.project_root()
        if explicit is not None:
            return explicit
        project_dir = self.project_dir_name()
        if project_dir is None:
            return None
        candidate = REPO_ROOT / "tools" / project_dir
        return candidate if candidate.exists() else None

    def _discover_project_files(self) -> list[Path]:
        root = self.project_root()
        if root is None:
            return []
        files: list[Path] = []
        for pattern in self.project_patterns():
            files.extend(root.glob(pattern))
        return _sorted_unique(files)[: self.manifest.expected_suite_size]

    def _discover_files(self) -> list[Path]:
        if not self.manifest.case_root.exists():
            return []
        patterns = ("*.json", "*.yaml", "*.yml", "*.py", "*.onnx", "*.pb", "*.txt")
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(self.manifest.case_root.rglob(pattern)))
        return files[: self.manifest.expected_suite_size]

    def _make_case(self, index: int, spec: dict[str, object] | None = None, source_path: Path | None = None) -> TestCase:
        spec = spec or {}
        original_order_raw = spec.get("original_order", index)
        try:
            original_order = int(original_order_raw)
        except (TypeError, ValueError):
            original_order = index

        input_features = tuple(
            InputFeature(dtype=str(item.get("dtype", "unknown")), rank=int(item.get("rank", -1)))
            for item in spec.get("input_features", [])
            if isinstance(item, dict)
        ) if isinstance(spec.get("input_features"), list) else ()

        env = None
        raw_env = spec.get("env")
        if isinstance(raw_env, dict):
            env = ExecutionEnv(
                device=str(raw_env.get("device", "unknown")),
                backend=str(raw_env.get("backend", "unknown")),
            )
        elif "device" in spec or "backend" in spec:
            env = ExecutionEnv(
                device=str(spec.get("device", "unknown")),
                backend=str(spec.get("backend", "unknown")),
            )

        graph_payload = spec.get("graph_payload")
        if not isinstance(graph_payload, dict):
            graph_payload = None

        raw_root_causes = spec.get("expected_root_causes", spec.get("root_causes", spec.get("bug_report_id")))
        if isinstance(raw_root_causes, (list, tuple)):
            expected_root_causes = tuple(str(item) for item in raw_root_causes if item not in {None, ""})
        elif raw_root_causes in {None, ""}:
            expected_root_causes = ()
        else:
            expected_root_causes = (str(raw_root_causes),)

        metadata = {
            key: value
            for key, value in spec.items()
            if key
            not in {
                "case_id",
                "graph_payload",
                "operator_sequence",
                "input_features",
                "env",
                "expected_root_causes",
            }
        }

        return TestCase(
            case_id=str(spec.get("case_id", f"{self.manifest.run_id}:{index:04d}")),
            tool_name=self.manifest.tool_name,
            source_path=source_path,
            original_order=original_order,
            graph_payload=graph_payload,
            operator_sequence=tuple(str(item) for item in spec.get("operator_sequence", []))
            if isinstance(spec.get("operator_sequence"), list)
            else (),
            input_features=input_features,
            env=env,
            expected_root_causes=expected_root_causes,
            metadata=metadata,
        )

    def _default_project_spec(self, index: int, source_path: Path) -> dict[str, object]:
        sequence = _infer_operator_sequence(
            (
                source_path.stem,
                source_path.parent.name,
                source_path.parent.parent.name if source_path.parent.parent != source_path.parent else "",
            ),
            fallback=f"{self.tool_name}Op",
        )
        family_hint = source_path.stem.split("-")[0].split("_")[0]
        return {
            "case_id": _sanitize_case_id(f"{self.tool_name}-{source_path.stem}"),
            "model_name": source_path.stem,
            "model_family": family_hint,
            "case_kind": self.case_kind(),
            "original_order": index + 1,
            "operator_sequence": list(sequence),
            "graph_payload": _build_chain_payload(sequence),
            "source_hint": str(source_path),
            "source_kind": source_path.suffix.lstrip(".") or "file",
            "source_project": self.project_root().name if self.project_root() is not None else self.tool_name,
        }

    def _case_spec_from_project_file(self, index: int, source_path: Path) -> dict[str, object]:
        return self._default_project_spec(index, source_path)

    def _enrich_inline_spec(self, index: int, spec: dict[str, object]) -> dict[str, object]:
        enriched = dict(spec)
        if "source_project" not in enriched and self.project_root() is not None:
            enriched["source_project"] = self.project_root().name
        enriched.setdefault("dataset_split", "release")

        if not isinstance(enriched.get("operator_sequence"), list) and not isinstance(enriched.get("graph_payload"), dict):
            family_candidates = (
                enriched.get("model_family"),
                enriched.get("source_seed_model"),
                enriched.get("model_name"),
            )
            sequence: tuple[str, ...] = ()
            for candidate in family_candidates:
                if isinstance(candidate, str):
                    sequence = _sequence_from_model_family(candidate)
                    if sequence:
                        break
            if not sequence:
                sequence = _infer_operator_sequence(
                    tuple(str(item) for item in family_candidates if isinstance(item, str)),
                    fallback=f"{self.tool_name}Op",
                )
            enriched["operator_sequence"] = list(sequence)
            enriched["graph_payload"] = _build_chain_payload(sequence)

        original_order = enriched.get("original_order")
        if not isinstance(original_order, int):
            enriched["original_order"] = index + 1
        return enriched

    def discover_cases(self) -> Sequence[TestCase]:
        project_files = self._discover_project_files()
        if project_files and self.manifest.prefer_project_files():
            return [
                self._make_case(index, spec=self._case_spec_from_project_file(index, path), source_path=path)
                for index, path in enumerate(project_files)
            ]

        inline = self.manifest.inline_cases()
        if inline:
            return [self._make_case(index, spec=self._enrich_inline_spec(index, spec)) for index, spec in enumerate(inline)]

        if project_files:
            return [
                self._make_case(index, spec=self._case_spec_from_project_file(index, path), source_path=path)
                for index, path in enumerate(project_files)
            ]

        files = self._discover_files()
        if files:
            return [
                self._make_case(
                    index,
                    spec={"case_id": path.stem, "source_hint": str(path)},
                    source_path=path,
                )
                for index, path in enumerate(files)
            ]

        return [self._make_case(index) for index in range(self.manifest.expected_suite_size)]


@dataclass(slots=True)
class COMETAdapter(BaseToolAdapter):
    tool_name: str = "COMET"

    def project_patterns(self) -> tuple[str, ...]:
        return ("data/synthesized_models/*.json",)

    def case_kind(self) -> str:
        return "synthesized-model"

    def _case_spec_from_project_file(self, index: int, source_path: Path) -> dict[str, object]:
        spec = self._default_project_spec(index, source_path)
        payload = _safe_json(source_path)
        graph_payload = _keras_graph_payload(payload) if payload is not None else None
        if graph_payload is not None:
            sequence = tuple(str(node.get("op", "UnknownOp")) for node in graph_payload.get("nodes", []))
            spec["graph_payload"] = graph_payload
            spec["operator_sequence"] = list(sequence)
        spec["model_family"] = source_path.stem.split("-")[0]
        spec["source_seed_model"] = source_path.stem.split("_")[0]
        spec["framework"] = "TensorFlow/Keras"
        spec["backend"] = "ONNXRuntime"
        return spec


@dataclass(slots=True)
class DevMuTAdapter(BaseToolAdapter):
    tool_name: str = "DevMuT"

    def project_patterns(self) -> tuple[str, ...]:
        return ("RQ1/COMET_modeljson/*.json",)

    def case_kind(self) -> str:
        return "model-mutation"

    def _case_spec_from_project_file(self, index: int, source_path: Path) -> dict[str, object]:
        spec = self._default_project_spec(index, source_path)
        payload = _safe_json(source_path)
        graph_payload = _devmut_graph_payload(payload) if payload is not None else None
        if graph_payload is not None:
            sequence = tuple(str(node.get("op", "UnknownOp")) for node in graph_payload.get("nodes", []))
            spec["graph_payload"] = graph_payload
            spec["operator_sequence"] = list(sequence)
        stem = source_path.stem
        spec["model_family"] = stem.split("_")[0].split("-")[0]
        if "_origin-" in stem:
            spec["mutation_trace"] = stem.split("_origin-", 1)[1]
        spec["framework"] = "PyTorch"
        spec["backend"] = "MindSpore"
        return spec


@dataclass(slots=True)
class ModelMetaAdapter(BaseToolAdapter):
    tool_name: str = "ModelMeta"

    def project_patterns(self) -> tuple[str, ...]:
        return (
            "mindspore_mutation/*.py",
            "mindspore_mutation/generate_models/*.py",
            "models/*/src/model*.py",
            "models/*/model_*.py",
            "configs/*.yaml",
        )

    def case_kind(self) -> str:
        return "model-level-metamorphic"

    def _case_spec_from_project_file(self, index: int, source_path: Path) -> dict[str, object]:
        spec = self._default_project_spec(index, source_path)
        family_hint = source_path.parent.name if source_path.parent.name != "configs" else source_path.stem
        sequence = _sequence_from_model_family(family_hint) or tuple(spec.get("operator_sequence", []))
        spec["operator_sequence"] = list(sequence)
        spec["graph_payload"] = _build_chain_payload(sequence)
        spec["metamorphic_relation"] = source_path.parent.name
        spec["framework"] = "PyTorch/MindSpore"
        spec["backend"] = "CUDA"
        return spec


@dataclass(slots=True)
class MuffinAdapter(BaseToolAdapter):
    tool_name: str = "Muffin"

    def project_patterns(self) -> tuple[str, ...]:
        return (
            "src/cases_generation/*.py",
            "scripts/incons/*.py",
            "test/*.py",
            "testing_config.json",
        )

    def case_kind(self) -> str:
        return "neural-architecture-fuzzing"

    def _case_spec_from_project_file(self, index: int, source_path: Path) -> dict[str, object]:
        spec = self._default_project_spec(index, source_path)
        family_hint = source_path.stem if source_path.suffix == ".json" else source_path.parent.name
        sequence = _sequence_from_model_family(family_hint) or tuple(spec.get("operator_sequence", []))
        spec["operator_sequence"] = list(sequence)
        spec["graph_payload"] = _build_chain_payload(sequence)
        spec["mutation_stage"] = source_path.parent.name
        spec["framework"] = "PyTorch"
        spec["backend"] = "TensorFlow"
        return spec


@dataclass(slots=True)
class NEURIAdapter(BaseToolAdapter):
    tool_name: str = "NEURI"

    def project_patterns(self) -> tuple[str, ...]:
        return ("data/tf_augmented_records/*.pkl",)

    def case_kind(self) -> str:
        return "recorded-op-sample"

    def _case_spec_from_project_file(self, index: int, source_path: Path) -> dict[str, object]:
        spec = self._default_project_spec(index, source_path)
        stem = source_path.stem
        op_fragment = stem.split(".")[-1]
        if "-" in op_fragment:
            op_fragment = op_fragment.rsplit("-", 1)[0]
        sequence = (TOKEN_OPERATOR_ALIASES.get(op_fragment.lower(), _default_operator(op_fragment)),)
        spec["operator_sequence"] = list(sequence)
        spec["graph_payload"] = _build_chain_payload(sequence)
        spec["target_ir"] = "TensorFlow raw_ops"
        spec["record_group"] = source_path.parent.name
        spec["framework"] = "PyTorch/TensorFlow"
        spec["backend"] = "TensorRT"
        return spec


@dataclass(slots=True)
class GenCoGAdapter(BaseToolAdapter):
    tool_name: str = "GenCoG"

    def project_patterns(self) -> tuple[str, ...]:
        return ("bug/*.py", "bug/**/*.py")

    def case_kind(self) -> str:
        return "bug-reproducer"

    def _case_spec_from_project_file(self, index: int, source_path: Path) -> dict[str, object]:
        spec = self._default_project_spec(index, source_path)
        spec["target_pass"] = source_path.parent.name if source_path.parent.name != "bug" else "relay"
        companion_doc = source_path.with_suffix(".md")
        if companion_doc.exists():
            spec["bug_note"] = str(companion_doc)
        spec["framework"] = "TVM Relay"
        spec["backend"] = "TVM"
        return spec


@dataclass(slots=True)
class LEMONAdapter(BaseToolAdapter):
    tool_name: str = "LEMON"

    def project_patterns(self) -> tuple[str, ...]:
        return ("run/*.py", "scripts/mutation/*.py", "scripts/localization/*.py", "config/*.conf")

    def case_kind(self) -> str:
        return "mutation-localization"

    def _case_spec_from_project_file(self, index: int, source_path: Path) -> dict[str, object]:
        spec = self._default_project_spec(index, source_path)
        phase = source_path.parent.name
        family_hint = source_path.stem if phase == "config" else phase
        sequence = _sequence_from_model_family(family_hint) or tuple(spec.get("operator_sequence", []))
        spec["operator_sequence"] = list(sequence)
        spec["graph_payload"] = _build_chain_payload(sequence)
        spec["pipeline_phase"] = phase
        spec["framework"] = "PyTorch"
        spec["backend"] = "TensorFlow"
        return spec


@dataclass(slots=True)
class GandalfAdapter(BaseToolAdapter):
    tool_name: str = "Gandalf"

    def project_patterns(self) -> tuple[str, ...]:
        return ("src/example_*.json", "trials/**/*.json")

    def case_kind(self) -> str:
        return "grammar-generated-network"

    def _case_spec_from_project_file(self, index: int, source_path: Path) -> dict[str, object]:
        spec = self._default_project_spec(index, source_path)
        payload = _safe_json(source_path)
        graph_payload = _gandalf_graph_payload(payload) if payload is not None else None
        if graph_payload is not None:
            sequence = tuple(str(node.get("op", "UnknownOp")) for node in graph_payload.get("nodes", []))
            spec["graph_payload"] = graph_payload
            spec["operator_sequence"] = list(sequence)
            if isinstance(payload.get("framework"), str):
                spec["framework"] = payload.get("framework")
        spec["trial_group"] = source_path.parent.name
        spec["backend"] = "TFLite"
        return spec


@dataclass(slots=True)
class NNSmithAdapter(BaseToolAdapter):
    tool_name: str = "NNSmith"

    def project_patterns(self) -> tuple[str, ...]:
        return ("experiments/legacy/*.py", "nnsmith/cli/*.py", "tests/**/*.py")

    def case_kind(self) -> str:
        return "generator-pipeline"

    def _case_spec_from_project_file(self, index: int, source_path: Path) -> dict[str, object]:
        spec = self._default_project_spec(index, source_path)
        phase = source_path.parent.name
        sequence = _infer_operator_sequence((source_path.stem, phase, "onnx"), fallback="GraphOp")
        spec["operator_sequence"] = list(sequence)
        spec["graph_payload"] = _build_chain_payload(sequence)
        spec["generator"] = "nnsmith"
        spec["pipeline_phase"] = phase
        spec["framework"] = "ONNX"
        spec["backend"] = "TVM"
        return spec


@dataclass(slots=True)
class AdapterRegistry:
    adapters: dict[str, type[BaseToolAdapter]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.adapters.setdefault("comet", COMETAdapter)
        self.adapters.setdefault("devmut", DevMuTAdapter)
        self.adapters.setdefault("modelmeta", ModelMetaAdapter)
        self.adapters.setdefault("muffin", MuffinAdapter)
        self.adapters.setdefault("neuri", NEURIAdapter)
        self.adapters.setdefault("gencog", GenCoGAdapter)
        self.adapters.setdefault("lemon", LEMONAdapter)
        self.adapters.setdefault("gandalf", GandalfAdapter)
        self.adapters.setdefault("nnsmith", NNSmithAdapter)

    def register(self, tool_name: str, adapter_cls: type[BaseToolAdapter]) -> None:
        self.adapters[tool_name.lower()] = adapter_cls

    def get(self, tool_name: str) -> type[BaseToolAdapter]:
        return self.adapters.get(tool_name.lower(), BaseToolAdapter)

    def create(self, manifest: ToolRunManifest) -> ToolAdapter:
        adapter_cls = self.get(manifest.tool_name)
        return adapter_cls(manifest=manifest)
