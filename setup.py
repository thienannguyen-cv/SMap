from setuptools import setup, find_packages
import re

def read_version():
    with open("smap/__init__.py", "r", encoding="utf8") as f:
        content = f.read()
    version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Could not find __version__ in __init__.py")

# Đọc file requirements.txt và loại bỏ các dòng trống hoặc comment
with open("requirements.txt", "r", encoding="utf-8") as req_file:
    install_requires = [
        line.strip() for line in req_file
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="smap-torch",  # Tên package của bạn
    version=read_version(),  # Phiên bản khởi đầu
    description="An open source pytorch library for spatial mapping based on 2D representations",  # Mô tả ngắn
    author="Thien An L. Nguyen",
    author_email="thienannguyen.cv@gmail.com",
    url="https://github.com/thienannguyen-cv/SMap",
    packages=find_packages(),  # Tự động tìm các package con
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
