from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rune_package',
            executable='dafu_node',
            name='dafu_node'
        ),
        Node(
            package='rune_package',
            executable='video_publisher',
            name='video_publisher'
        ),
        # Node(
        #     package='rune_package',
        #     executable='tiaocan_node',
        #     name='tiaocan'
        # ),
        Node(
            package='rune_package',
            executable='onnx_detect',
            name='onnx_detect'
        )
    ])