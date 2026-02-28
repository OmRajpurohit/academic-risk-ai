from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    with open(file_path, encoding="utf-8") as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements


setup(
    name="academic-risk-ai",
    version="0.0.1",
    author="Om Singh",
    author_email="omsinghpurohit13@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements("requirements.txt"),
)