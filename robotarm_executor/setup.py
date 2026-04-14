from setuptools import find_packages, setup

package_name = 'robotarm_executor'

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
            'chair_grasp_moveit = robotarm_executor.chair_grasp_moveit:main',
            'chair_grasp_moveit_vertical_move = robotarm_executor.chair_grasp_moveit_vertical_move:main',
            'chair_grasp_moveit_debug = robotarm_executor.chair_grasp_moveit_debug:main',
            'chair_grasp_moveit_dataset = robotarm_executor.chair_grasp_moveit_dataset:main',
            'chair_grasp_moveit_openvla_dataset = robotarm_executor.chair_grasp_moveit_openvla_dataset:main',
            'chair_grasp_moveit_diffusion_vla_dataset = robotarm_executor.chair_grasp_moveit_diffusion_vla_dataset:main',
            'chair_grasp_moveit_openvla_strict_policy = robotarm_executor.chair_grasp_moveit_openvla_strict_policy:main',
            'chair_grasp_moveit_diffusion_policy = robotarm_executor.chair_grasp_moveit_diffusion_policy:main',
            'chair_grasp_moveit_diffusion_policy_lda = robotarm_executor.chair_grasp_moveit_diffusion_policy_lda:main',
        ],
    },
)
