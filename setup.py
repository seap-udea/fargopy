import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    # ######################################################################
    # BASIC DESCRIPTION
    # ######################################################################
    name="fargopy",
    author="Jorge Zuluaga, Alejandro Murillo-González, Matias Montesinos",
    author_email="jorge.zuluaga@udea.edu.co",
    description="FARGO3D Wrapper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/fargopy",
    keywords="astronomy MHD CFD",
    license="AGPL-3.0-only",
    # ######################################################################
    # CLASSIFIER
    # ######################################################################
    # For Development status see: https://martin-thoma.com/software-development-stages/
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    version='1.1.2',
    # ######################################################################
    # FILES
    # ######################################################################
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    # ######################################################################
    # ENTRY POINTS
    # ######################################################################
    entry_points={
        "console_scripts": ["install=fargopy.install:main"],
    },
    # ######################################################################
    # TESTS
    # ######################################################################
    test_suite="tests",
    tests_require=["pytest"],
    # ######################################################################
    # DEPENDENCIES
    # ######################################################################
    install_requires=[
        "scipy",
        "matplotlib",
        "PyQt5",
        "tqdm",
        "numpy",
        "ipython",
        "scikit-learn",
        "ipympl",
        "joblib",
        "celluloid",
        "psutil",
        "gdown",
        "pandas",
        "plotly",
        "pytest",
        "nbformat>=4.2.0",
    ],
    python_requires=">=3.7",
    # ######################################################################
    # OPTIONS
    # ######################################################################
    include_package_data=True,
    package_data={"fargopy": ["data/*.*", "tests/*"]},
    scripts=["src/fargopy/bin/ifargopy", "src/fargopy/bin/vfargopy"],
)
