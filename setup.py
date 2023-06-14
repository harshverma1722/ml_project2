from setuptools import find_packages,setup
from typing import List 

HYPEN_E_DOT = '-e .'
## Whenever the requirements.txt is run then setup.py is run automatically and packages are created 
def get_requirements(file_path:str)->List[str]:
    requirements =[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements =[req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements
setup(
    name="DiamondPricePrediction",
    version = '0.0.1',
    author  = 'Harsh Verma',
    author_email='vermaharsh.1722@gmail.com',
    install_requires = get_requirements("requirements.txt"),
    packages = find_packages()

)

