from spherepar.benchmark.dataset_generator import _effective_param_method


def test_mnist_deformation_cases_keep_parametrization():
    assert _effective_param_method("mnist", "case2_small", "flash") == "flash"
    assert _effective_param_method("mnist", "case3_large", "cem") == "cem"
    assert _effective_param_method("mnist", "case1_no", "flash") is None
