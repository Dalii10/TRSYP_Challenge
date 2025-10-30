import sys
import termios
import tty
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


def get_key():
    """Lire une touche du clavier sans attendre ENTER."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info("Teleop Node lancé ! Utilise ZQSD pour bouger, espace pour stop, Ctrl+C pour quitter.")

        self.linear_speed = 0.2   # m/s
        self.angular_speed = 0.5  # rad/s

    def run(self):
        twist = Twist()
        while rclpy.ok():
            key = get_key()

            # Réinitialiser les vitesses
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            if key == 'z':       # avant
                twist.linear.x = self.linear_speed
            elif key == 's':     # arrière
                twist.linear.x = -self.linear_speed
            elif key == 'q':     # gauche
                twist.angular.z = self.angular_speed
            elif key == 'd':     # droite
                twist.angular.z = -self.angular_speed
            elif key == ' ':     # stop
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            elif key == '\x03':  # Ctrl+C
                break

            self.publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
