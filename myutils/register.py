class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, module, name=None):
        # if name in ["Patchcore","CONTROLDDPMTrainer"]:
        #     print(1)
        if name is None:
            name = module.__name__
        if name in self._module_dict:
            raise KeyError(f"{name} is already registered in {self._name}")
        self._module_dict[name] = module
        return module

    def get_module(self, name):
        if name not in self._module_dict:
            raise KeyError(f"{name} is not registered in {self._name}")
        return self._module_dict[name]

    def __repr__(self):
        format_str = self.__class__.__name__ + f"(name={self._name}, items={list(self._module_dict.keys())})"
        return format_str
