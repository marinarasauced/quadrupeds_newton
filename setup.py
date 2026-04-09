from setuptools import find_packages, setup
import os

package_name = 'qsim_newton'
executable_path = os.path.join(os.path.expanduser('~/'), '.virtualenvs', 'newton', 'bin', 'python')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml'])
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Marina Nelson',
    maintainer_email='marinarasauced@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'newton = scripts.newton:main',
        ],
    },
    options={
        'build_scripts': {
            'executable': executable_path,
    }
},
)
