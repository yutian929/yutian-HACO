from setuptools import setup, find_packages

setup(
    name='haco',                          # 包名
    version='0.1.0',                       # 版本号
    packages=find_packages(),             # 自动发现包
    install_requires=[                    # 包依赖
        # 'numpy', 'torch', 'opencv-python',  # 示例依赖项
    ],
    include_package_data=True,             # 包含其他文件
    package_data={                         # 额外文件
        # 'haco': ['data/*.txt'],            # 假设你有数据文件
    },
    tests_require=[
        'pytest',                         # 测试框架
    ],
    test_suite='tests',                    # 测试路径
    entry_points={                         # 可选: 可执行脚本
        'console_scripts': [
            'haco-demo=haco.demo:main',    # 示例：创建命令行工具
        ],
    },
)
