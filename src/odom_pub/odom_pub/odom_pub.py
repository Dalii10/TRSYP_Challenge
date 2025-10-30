import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf_transformations
import tf2_ros
import math
import serial

class OdomNode(Node):
    def __init__(self):
        super().__init__('odom_node')

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Robot pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Robot parameters
        self.wheel_base = 0.16        # distance entre les roues (m)
        self.ticks_per_meter = 2522   # ticks pour 1 mètre

        # Arduino Serial
        self.ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

        # Derniers ticks
        self.last_ticks_left = 0
        self.last_ticks_right = 0

        # Timer
        self.last_time = self.get_clock().now().nanoseconds / 1e9
        self.create_timer(0.05, self.update_odom)  # 20 Hz

    def update_odom(self):
        try:
            # Lire Serial
            line = self.ser.readline().decode(errors='ignore').strip()
            if not line:
                return

            left_str, right_str = line.split(",")
            left_ticks = int(left_str)
            right_ticks = int(right_str)

        except:
            return

        # Calcul delta ticks
        delta_left = left_ticks - self.last_ticks_left
        delta_right = right_ticks - self.last_ticks_right
        self.last_ticks_left = left_ticks
        self.last_ticks_right = right_ticks

        # Temps écoulé
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.last_time
        if dt == 0:
            return
        self.last_time = now

        # Conversion ticks → distance
        d_left = delta_left / self.ticks_per_meter
        d_right = delta_right / self.ticks_per_meter

        # Distance centrale et rotation
        d_center = (d_left + d_right) / 2.0
        dtheta = (d_right - d_left) / self.wheel_base

        # Mise à jour pose
        self.theta += dtheta
        self.x += d_center * math.cos(self.theta)
        self.y += d_center * math.sin(self.theta)

        # Vitesses
        vx = d_center / dt
        vtheta = dtheta / dt

        # Message Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_footprint'   # ✅ changé ici

        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0

        q = tf_transformations.quaternion_from_euler(0, 0, self.theta)
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        # Twist
        odom_msg.twist.twist.linear.x = vx
        odom_msg.twist.twist.angular.z = vtheta

        self.odom_pub.publish(odom_msg)

        # Publier TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'   # ✅ changé ici
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

