from os import path
import setuptools
from shutil import copyfile
from sys import platform


dirname = path.dirname(path.abspath(__file__))

if platform == "linux" or platform == "linux2":
	lib_path = path.abspath(path.join(dirname, '../build/lib/libthundergbm.so'))
elif platform == "win32":
	lib_path = path.abspath(path.join(dirname, '../build/bin/Debug/thundergbm.dll'))
elif platform == "darwin":
	lib_path = path.abspath(path.join(dirname, '../build/lib/libthundergbm.dylib'))
else :
	print ("OS not supported!")
	exit()
if not path.exists(path.join(dirname, "thundergbm", path.basename(lib_path))):
	copyfile(lib_path, path.join(dirname, "thundergbm", path.basename(lib_path)))
setuptools.setup(name="thundergbm",
			     version="0.0.6",
				 packages=["thundergbm"],
				 package_dir={"python": "thundergbm"},
			     description="A Fast GBM Library on GPUs and CPUs",
			     long_description="""The mission of ThunderGBM is to help users easily and efficiently apply GBDTs and Random Forests to solve problems. ThunderGBM exploits GPUs and multi-core CPUs to achieve high efficiency""",
			     long_description_content_type="text/plain",
			     url="https://github.com/zeyiwen/thundergbm",
			     package_data = {"thundergbm": [path.basename(lib_path)]},
			     install_requires=['numpy','scipy','scikit-learn'],
			     classifiers=[
			                  "Programming Language :: Python :: 3",
			                  "License :: OSI Approved :: Apache Software License",
			    ],
				 python_requires=">=3"
)
