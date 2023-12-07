import pkgutil
import inspect
import importlib

# 遍历当前目录下的所有模块
for _, module_name, _ in pkgutil.iter_modules(__path__):
    # 动态导入模块
    module = importlib.import_module('.' + module_name, __name__)

    # 遍历模块中的所有类
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            # 动态导入类
            globals()[name] = obj