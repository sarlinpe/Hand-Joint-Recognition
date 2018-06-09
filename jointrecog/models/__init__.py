def get_model(name):
    mod = __import__('jointrecog.models.{}'.format(name), fromlist=[''])
    return getattr(mod, _module_to_class(name))


def _module_to_class(name):
    return ''.join(n.capitalize() for n in name.split('_'))
