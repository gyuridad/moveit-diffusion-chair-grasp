import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class IsaacToMoveItJointStateBridge(Node):
    def __init__(self):
        super().__init__("isaac_to_moveit_joint_state_bridge")

        # IsaacSim에서 나오는 joint_states
        self.sub = self.create_subscription(
            JointState,
            "/isaac_joint_states",   # Isaac Publish 토픽명
            self._cb,
            10,
        )

        # MoveIt / robot_state_publisher가 보는 표준 joint_states
        self.pub = self.create_publisher(
            JointState,
            "/joint_states",
            10,
        )

        self.get_logger().info("Bridge: /isaac_joint_states --> /joint_states 시작")

    def _cb(self, msg: JointState):
        out = JointState()
        out.header = msg.header
        out.name = list(msg.name)
        out.position = list(msg.position)
        out.velocity = list(msg.velocity)
        out.effort = list(msg.effort)
        self.pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacToMoveItJointStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


### 실행
# ros2 run isaac_joint_bridge isaac_joint_state_bridge

## 실행위치
# cd ~/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws



