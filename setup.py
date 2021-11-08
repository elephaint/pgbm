import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
	name="pgbm",
	version="1.5",
	description="Probabilistic Gradient Boosting Machines in Pytorch",
	author="Olivier Sprangers",
	author_email="o.r.sprangers@uva.nl",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elephaint/pgbm",
    packages=setuptools.find_packages(where="src"),
    include_package_data = True,
	package_dir={"pgbm": "src/pgbm",
              "pgbm_nb": "src/pgbm_nb",
		"pgbm_dist": "src/pgbm_dist"},
    classifiers=[
         "Programming Language :: Python :: 3.7",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent"],
    python_requires='>=3.7',
    install_requires=["scikit-learn>=0.22.0",
                      "matplotlib>=2.2.3"],
    extras_require={
        "torch-gpu":["torch==1.10.0+cu102",
                     "ninja>=1.10.2.2"],
        "torch-cpu":["torch>=1.8.0",
                     "ninja>=1.10.2.2"],
        "numba":["numba>=0.53.1"],
        "all":["torch==1.10.0+cu102",
               "ninja>=1.10.2.2",
               "numba>=0.53.1"]
        },
    zip_safe=False)