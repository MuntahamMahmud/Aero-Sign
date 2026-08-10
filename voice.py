import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import speech_recognition as sr
import threading

class VoiceToCmdVelPublisher(Node):
    def __init__(self):
        super().__init__('voice_to_cmd_vel_publisher')
        
        # Change this topic string if your bridge uses a specific /model/ namespace
        self.publisher_ = self.create_publisher(Twist, '/X3/cmd_vel', 10)
        self.get_logger().info('Voice Control Publisher Node Initialized.')
        
        # Setup audio input processing modules
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Run microphone loop in an independent thread to keep ROS 2 responsive
        self.listening_thread = threading.Thread(target=self.run_audio_listener)
        self.listening_thread.daemon = True
        self.listening_thread.start()

    def run_audio_listener(self):
        with self.microphone as source:
            # Calibrate microphone for room background noise before starting
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.get_logger().info("Microphone calibrated. Awaiting vocal command paths...")
            
            while rclpy.ok():
                try:
                    audio = self.recognizer.listen(source, timeout=4, phrase_time_limit=2)
                    text_input = self.recognizer.recognize_google(audio).lower()
                    self.get_logger().info(f"Speech Recognized: '{text_input}'")
                    self.evaluate_voice_command(text_input)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    self.get_logger().warn("Audio detected but phrase could not be resolved.")
                except Exception as error:
                    self.get_logger().error(f"Audio interface exception: {str(error)}")

    def evaluate_voice_command(self, text):
        twist_cmd = Twist()
        
        # Mapping vocal commands to geometric twist vectors
        if "forward" in text:
            twist_cmd.linear.x = 1.0
            self.get_logger().info("Executing: Pitch Forward")
        elif "backward" in text:
            twist_cmd.linear.x = -1.0
            self.get_logger().info("Executing: Pitch Backward")
        elif "up" in text or "takeoff" in text:
            twist_cmd.linear.z = 0.8
            self.get_logger().info("Executing: Climb Throttle")
        elif "down" in text or "land" in text:
            twist_cmd.linear.z = -0.8
            self.get_logger().info("Executing: Descend Throttle")
        elif "left" in text:
            twist_cmd.angular.z = 0.6
            self.get_logger().info("Executing: Yaw Left Turn")
        elif "right" in text:
            twist_cmd.angular.z = -0.6
            self.get_logger().info("Executing: Yaw Right Turn")
        elif "stop" in text or "hover" in text:
            # Leaves all parameters at exactly 0.0 to lock position
            self.get_logger().info("Executing: Hard Zero Hover State")
            
        self.publisher_.publish(twist_cmd)

def main(args=None):
    rclpy.init(args=args)
    node = VoiceToCmdVelPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()