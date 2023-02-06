from setuptools import setup, Extension
import contextlib
import numpy
import sys
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

# Cythonize extensions: https://github.com/scikit-learn/scikit-learn/blob/964189df31dd2aa037c5bc58c96f88193f61253b/sklearn/_build_utils/__init__.py#L39
def cythonize_extensions(extension):
    from Cython.Build import cythonize
    
    n_jobs = 1
    with contextlib.suppress(ImportError):
        import joblib

        n_jobs = joblib.cpu_count()

    return cythonize(
        extension,
        nthreads=n_jobs,
        compiler_directives={
            "language_level": 3,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
            "boundscheck": False,
        },
    )

def get_openmp_flag():
    if sys.platform == "win32":
        return ["/openmp"]
    elif sys.platform == "darwin" and "openmp" in os.getenv("CPPFLAGS", ""):
        return []
    # Default flag for GCC and clang:
    return ["-fopenmp"]

# Cython modules
extensions = ["_gradient_boosting.pyx", "histogram.pyx", "splitting.pyx", 
            "_binning.pyx", "_bitset.pyx", "_predictor.pyx", "common.pyx", "utils.pyx"]

ext_modules = []
for extension in extensions:
    new_ext = Extension(name = "pgbm.sklearn." + os.path.splitext(extension)[0], 
                        sources = ["pgbm/sklearn/" + extension],
                        include_dirs=[numpy.get_include()],
                        extra_link_args= get_openmp_flag(),
                        extra_compile_args= get_openmp_flag() + ["-O3"],
                        define_macros=[("NPY_NO_DEPRECATED_API", 
                                        "NPY_1_9_API_VERSION")])
    ext_modules.append(new_ext)

# Setup
if __name__ == "__main__":
    setup(
        name="pgbm",
        version="2.0.0",
        description="Probabilistic Gradient Boosting Machines",
        author="Olivier Sprangers",
        author_email="o.r.sprangers@uva.nl",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/elephaint/pgbm",
        packages=["pgbm"],
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent"],
        python_requires='>=3.8',
        install_requires=["scikit-learn>=1.1.2",
                        "ninja>=1.10.2.2",
                        "numba>=0.56"],
        zip_safe=False,
        ext_modules=cythonize_extensions(ext_modules),
        )