
is_simple_core = False

if is_simple_core:
    from mytorch.simple_core import Variable
    from mytorch.simple_core import Function
    from mytorch.simple_core import using_config
    from mytorch.simple_core import no_grad
    from mytorch.simple_core import as_array
    from mytorch.simple_core import as_variable
    from mytorch.simple_core import setup_variable

else:
    from mytorch.core import Variable
    from mytorch.core import Function
    from mytorch.core import Config
    from mytorch.core import using_config
    from mytorch.core import no_grad
    from mytorch.core import test_mode
    from mytorch.core import as_array
    from mytorch.core import as_variable
    from mytorch.core import setup_variable

    from mytorch import utils
    from mytorch import functions
    from mytorch import functions_conv
    from mytorch import optimizers
    from mytorch import datasets
    from mytorch import cuda
    from mytorch.layers import Layer
    from mytorch.models import Model
    from mytorch.datasets import Dataset
    from mytorch.dataloaders import DataLoader
    import mytorch.functions_conv


setup_variable()
