from setuptools import find_packages, setup

package_name = 'my_robot_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='asus',
    maintainer_email='asus@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'teleop_keyboard = my_robot_teleop.teleop_keyboard:main',
        	'cmdvel_to_serial = my_robot_teleop.cmdvel_to_serial:main',

        ],
    },
)
