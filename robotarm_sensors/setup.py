from setuptools import find_packages, setup

package_name = 'robotarm_sensors'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'robotarm_common',
    ],
    zip_safe=True,
    maintainer='lst7910',
    maintainer_email='gyuridadd@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'udp_camera_sender = robotarm_sensors.udp_camera_sender:main',
            'udp_camera_sender_debug = robotarm_sensors.udp_camera_sender_debug:main',
            'isaac_moveit_bridge = robotarm_sensors.isaac_moveit_bridge_node:main',
        ],
    },
)
