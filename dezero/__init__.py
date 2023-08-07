is_simple_core = False

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import set_variable
else:
    from dezero.core import Variable
    from dezero.core import Function
    from dezero.core import Parameter
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import set_variable

    from dezero.layers import Layer
    from dezero.models import Model
    from dezero.dataloaders import DataLoader

    import dezero.functions
    import dezero.utils
    import dezero.datasets
    import dezero.cuda

set_variable()
