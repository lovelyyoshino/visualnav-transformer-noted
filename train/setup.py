from setuptools import setup, find_packages

# 使用setuptools库来设置Python包的相关信息
setup(
    name="vint_train",  # 包的名称
    version="0.1.0",   # 包的版本号
    packages=find_packages(),  # 自动查找当前目录下所有的包（包含__init__.py文件的目录）
)
