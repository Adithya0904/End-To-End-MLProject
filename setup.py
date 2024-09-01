from setuptools import find_packages,setup
from typing import List


HIPHEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[package.replace("/n","") for package in requirements]

        if HIPHEN_E_DOT in requirements:
            requirements.remove(HIPHEN_E_DOT)

    return requirements



setup(
    name='ML PROJECT',
    version='0.0.1',
    author='Adithya',
    author_email='hebbaradithya215@gmail.com',
    packages=find_packages(),
    install_requirements=get_requirements('requirements.txt')
)