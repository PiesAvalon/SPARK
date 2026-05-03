import pytest

from neuri.abstract.dtype import DType
from neuri.backends import BackendFactory
from neuri.graph_gen import model_gen
from neuri.materialize import Model, TestCase
from neuri.materialize.tensorflow import TFModelCPU
from neuri.narrow_spec import auto_opconfig, auto_opset

TestCase.__test__ = False  # supress PyTest warning

# skip tflite for now
pytest.skip("tflite backend is not ready yet", allow_module_level=True)


def test_narrow_spec_cache_make_and_reload():
    factory = BackendFactory.init("tflite", target="cpu", optmax=True)
    ModelType = Model.init("tensorflow")
    opset_lhs = auto_opconfig(ModelType, factory)
    assert opset_lhs, "Should not be empty... Something must go wrong."
    opset_rhs = auto_opconfig(ModelType, factory)
    assert opset_lhs == opset_rhs

    # Assert types
    assert isinstance(opset_lhs["core.ReLU"].in_dtypes[0][0], DType)

    # Assert Dictionary Type Equality
    assert type(opset_lhs) == type(opset_rhs)
    assert type(opset_lhs["core.ReLU"]) == type(opset_rhs["core.ReLU"])
    assert type(opset_lhs["core.ReLU"].in_dtypes[0][0]) == type(
        opset_rhs["core.ReLU"].in_dtypes[0][0]
    )


def test_synthesized_tf_model(tmp_path):
    d = tmp_path / "test_tflite"
    d.mkdir()

    ModelType = Model.init("tensorflow")
    factory = BackendFactory.init("tflite", target="cpu", optmax=False)

    gen = model_gen(
        opset=auto_opset(TFModelCPU, factory),
        seed=23132,
        max_nodes=4,
    )  # One op should not be easily wrong... I guess.

    model = ModelType.from_gir(gen.make_concrete())

    # model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()

    testcase = TestCase(model, oracle)
    testcase.dump(root_folder=d)

    assert factory.verify_testcase(testcase) is None
