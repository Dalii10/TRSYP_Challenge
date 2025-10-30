import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import time

class CmdVelToSerial(Node):
    def __init__(self):
        super().__init__('cmdvel_to_serial')

        # Souscription au topic /cmd_vel
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.listener_callback,
            10
        )

        # Connexion série (adapter /dev/ttyACM0 ou /dev/ttyUSB0)
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)
            self.get_logger().info("Connecté à Arduino sur /dev/ttyACM0")
        except Exception as e:
            self.get_logger().error(f"Erreur ouverture série: {e}")
            exit(1)

        # Limitation fréquence
        self.last_send_time = 0.0
        self.send_interval = 0.1  # 10 Hz max

    def listener_callback(self, msg: Twist):
        now = time.time()
        if now - self.last_send_time < self.send_interval:
            return  # Attendre le prochain intervalle
        self.last_send_time = now

        # Construire la commande exactement comme avant
        v_lin = msg.linear.x * 5
        v_ang = msg.angular.z * 5
        cmd = f"V {v_lin:.2f} {v_ang:.2f}\n"

        try:
            self.ser.write(cmd.encode())
            # Petit délai pour laisser Arduino lire la commande
            time.sleep(0.005)  # 5 ms
            self.get_logger().info(f"Envoyé: {cmd.strip()}")
        except Exception as e:
            self.get_logger().error(f"Erreur envoi série: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToSerial()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

