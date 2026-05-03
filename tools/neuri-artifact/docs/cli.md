## Graph generation

```shell
# Generate 5-node onnx model.
python neuri/cli/model_gen.py mgen.max_nodes=5 model.type=onnx debug.viz=true
# See: neuri_output/* (default output folder)

# TensorFlow model.
python neuri/cli/model_gen.py debug.viz=true model.type=tensorflow

# User-spec. output directory
python neuri/cli/model_gen.py debug.viz=true model.type=tensorflow mgen.save=tf_output
```

## Locally debug a model

```python
# Generate a onnx model
python neuri/cli/model_gen.py model.type=onnx mgen.max_nodes=5

# Check the model
pip install onnxruntime # use ONNXRuntime to execute the model
python neuri/cli/model_exec.py model.type=onnx backend.type=onnxruntime model.path=neuri_output/model.onnx
# `model.path` should point to the exact model, instead of a folder.
# It will first compile and run to see if there's any bug.
# By default it will search `oracle.pkl` and do verification.

# Check the model and do diff testing with tvm
python neuri/cli/model_exec.py  model.type=onnx                        \
                                backend.type=onnxruntime               \
                                model.path=neuri_output/model.onnx   \
                                cmp.with='{type:tvm, optmax:true, target:cpu}'
```

## Experimental: Gradient checking

For `pt2` and `torchjit`, we have initial supports for examining the gradients.

To enable that, just need to append `mgen.grad_check=true` to the examples illustrated above.

## Data type testing

Many compilers do not support a full set of operators (in ONNX and TensorFlow). Thus, we infer the support set by doing single operator testing.

```shell
# Infer the support set of onnxruntime to ONNX format.
python neuri/cli/dtype_test.py model.type="onnx" backend.type="onnxruntime"
# Results are often cached in `~/.cache/neuri`.
```

## Fuzzing

```shell
python neuri/cli/fuzz.py fuzz.time=30s model.type=onnx backend.type=tvm fuzz.root=fuzz_report debug.viz=true
# Bug reports are stored in `./fuzz_report`.
```

## Limit operator types, ranks and data types

To limit:
- rank only to be 4 (needed by Conv2d);
- data type only to be float32;
- only include Conv2d and ReLU.

```shell
yes | python neuri/cli/model_gen.py model.type=torch           \
                                    mgen.method=symbolic-cinit \
                                    mgen.rank_choices="[4]"    \
                                    mgen.dtype_choices="[f32]" \
                                    mgen.include="[core.NCHWConv2d, core.ReLU]" \
                                    debug.viz=true
```

## Add extra constraints

```shell
# Create patch file as `patch.py`
echo 'from neuri.abstract.arith import nnsmith_lt
from neuri.abstract.extension import patch_requires


@patch_requires("global", "core.NCHWConv2d")
def limit_conv2d(self, _):
    # let the kernels to be > 3
    return [nnsmith_lt(3, self.kernel_h_size), nnsmith_lt(3, self.kernel_w_size)]
' > patch.py
# Apply the patch with `mgen.patch_requires=./tests/mock/requires_patch.py` (can also be a list of paths)
yes | python neuri/cli/model_gen.py model.type=torch mgen.method=symbolic-cinit \
                                                     mgen.rank_choices="[4]"    \
                                                     mgen.dtype_choices="[f32]" \
                                                     mgen.include="[core.NCHWConv2d, core.ReLU]" \
                                                     mgen.patch_requires=./patch.py \
                                                     debug.viz=true
```

## Misc

TensorFlow logging can be very noisy. Use `TF_CPP_MIN_LOG_LEVEL=3` as environmental variable to depress that.
