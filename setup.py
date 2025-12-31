from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stampede-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-Camera Stampede Detection System using YOLOv8 and Optical Flow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stampede-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "PyQt5>=5.15.0",
        "pyqtgraph>=0.13.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "stampede-detection=main:main",
        ],
    },
)
