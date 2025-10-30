from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # === LiDAR driver ===
        Node(
            package='ydlidar_ros2_driver',
            executable='ydlidar_launch.py',
            name='ydlidar_driver',
            parameters=[{'port': '/dev/ttyUSB0', 'baudrate': 128000}],
            output='screen'
        ),

        # === Slam Toolbox ===
        Node(
            package='slam_toolbox',
            executable='sync_slam_toolbox_node',
            name='slam_toolbox',
            parameters=['/home/$USER/ros2_ws/config/my_slam_config.yaml'],
            output='screen'
        ),

        # === Static TF: base_footprint -> base_link ===
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', 'base_link']
        ),

        # === Static TF: base_link -> laser_frame ===
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser_frame']
        ),

        # === Odometry Publisher (avec encodeurs) ===
        Node(
            package='odom_pub',
            executable='odom_pub',
            name='odom_publisher',
            output='screen'
        ),

        # === Téléop clavier (optionnel) ===
        Node(
            package='my_robot_teleop',
            executable='teleop_keyboard',
            name='teleop_keyboard',
            output='screen'
        )
    ])
