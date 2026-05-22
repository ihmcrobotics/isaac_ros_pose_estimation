#!/usr/bin/env python3

from vision_msgs.msg import Detection3DArray
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node


class PoseDistributionVisualizer(Node):
    def __init__(self):
        super().__init__('pose_distribution_visualizer')

        topic_name = '/foundationpose/door_lever/door_lever_1/tracking/output'
        # topic_name = '/foundationpose/door_lever/door_lever_1/pose_estimation/output'

        self._pose_sub = self.create_subscription(
            Detection3DArray,
            topic_name,
            self.pose_callback,
            10
        )

        self._pose_history = {
            'position': {'x': [], 'y': [], 'z': []},
            'orientation': {'x': [], 'y': [], 'z': [], 'w': []}
        }

        self.get_logger().info(f'Subscribing to: {topic_name}')

    def visualize_data(self):
        for key, data_dict in self._pose_history.items():
            fig, ax = plt.subplots(1, len(data_dict))
            for i, (label, value) in enumerate(data_dict.items()):
                ax[i].hist(value, bins=30)
                ax[i].set_title(label)
            fig.suptitle(key)

        # Time-series plots are better for drift
        for key, data_dict in self._pose_history.items():
            fig, ax = plt.subplots(1, len(data_dict))
            for i, (label, value) in enumerate(data_dict.items()):
                ax[i].plot(value)
                ax[i].set_title(f'{key} {label} over time')
                ax[i].set_xlabel('frame')
            fig.suptitle(f'{key} time series')

        plt.show()

    def pose_callback(self, msg):
        if len(msg.detections) == 0:
            return

        detection = msg.detections[0]

        pose = detection.bbox.center

        self._pose_history['position']['x'].append(pose.position.x)
        self._pose_history['position']['y'].append(pose.position.y)
        self._pose_history['position']['z'].append(pose.position.z)

        self._pose_history['orientation']['x'].append(pose.orientation.x)
        self._pose_history['orientation']['y'].append(pose.orientation.y)
        self._pose_history['orientation']['z'].append(pose.orientation.z)
        self._pose_history['orientation']['w'].append(pose.orientation.w)

        self.get_logger().info(
            f'pose: x={pose.position.x:.4f}, y={pose.position.y:.4f}, z={pose.position.z:.4f}'
        )


def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = PoseDistributionVisualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.visualize_data()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()