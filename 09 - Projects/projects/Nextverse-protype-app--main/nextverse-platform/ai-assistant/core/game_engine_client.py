import socket
import json
import logging

logger = logging.getLogger(__name__)

class GameEngineClient:
    """
    Client to send commands to the C++ Game Engine's AICommandServer.
    """
    def __init__(self, host: str, port: int):
        """
        Initializes the GameEngineClient.

        Args:
            host (str): The hostname or IP address of the Game Engine server.
            port (int): The port number of the Game Engine server.
        """
        self.host = host
        self.port = port
        logger.info(f"GameEngineClient initialized to connect to {self.host}:{self.port}")

    def send_command(self, command_dict: dict) -> bool:
        """
        Sends a command to the Game Engine server.

        Args:
            command_dict (dict): A dictionary representing the command to send.
                                 This will be serialized to JSON.

        Returns:
            bool: True if the command was sent and a basic "OK" response was received,
                  False otherwise.
        """
        client_socket = None  # Initialize to None for finally block
        try:
            # Create a TCP/IP socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5.0) # 5 second timeout for connect and recv

            # Connect the socket to the server's address and port
            logger.debug(f"Attempting to connect to Game Engine at {self.host}:{self.port}")
            client_socket.connect((self.host, self.port))
            logger.info(f"Successfully connected to Game Engine at {self.host}:{self.port}")

            # Serialize command dictionary to JSON string
            json_command = json.dumps(command_dict)
            logger.debug(f"Sending command: {json_command}")

            # Send the JSON string, UTF-8 encoded, followed by a newline
            # The newline can help the server side to delimit messages if it reads line-by-line
            client_socket.sendall((json_command + "\n").encode('utf-8'))
            logger.debug("Command sent to Game Engine.")

            # Receive response (optional for MVP, but good for confirmation)
            # The C++ server sends "OK\n" or "ERROR...\n"
            response_bytes = client_socket.recv(1024)
            response_str = response_bytes.decode('utf-8').strip()
            logger.info(f"Received response from Game Engine: '{response_str}'")

            return response_str == "OK"

        except socket.timeout:
            logger.error(f"GameEngineClient: Socket timeout when trying to reach {self.host}:{self.port}")
            return False
        except socket.error as e:
            logger.error(f"GameEngineClient: Socket error when communicating with {self.host}:{self.port}: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"GameEngineClient: Error decoding JSON response from server: {e}")
            return False
        except Exception as e:
            logger.error(f"GameEngineClient: An unexpected error occurred: {e}")
            return False
        finally:
            if client_socket:
                try:
                    client_socket.shutdown(socket.SHUT_RDWR) # Gracefully shutdown
                except socket.error:
                    pass # Ignore if already closed or error during shutdown
                client_socket.close()
                logger.debug("GameEngineClient: Socket closed.")
            else:
                logger.debug("GameEngineClient: Socket was not opened.")

if __name__ == '__main__':
    # Example Usage (requires the C++ Game Engine with AICommandServer to be running)
    # Configure logger for example
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Assuming game engine is running on localhost:8081 for this test
    client = GameEngineClient(host='localhost', port=8081)

    # Example 1: Valid move_to command for an existing avatar
    cmd1 = {
        "avatar_id": "ai_1", # Assumes "ai_1" exists in the game engine
        "action": "move_to",
        "target_x": 10.0,
        "target_y": 12.0,
        "target_z": 0.5
    }
    print(f"\n--- Sending Command 1 ---")
    if client.send_command(cmd1):
        print("Command 1 sent successfully and OK received.")
    else:
        print("Failed to send Command 1 or non-OK response received.")

    # Example 2: Another valid command
    cmd2 = {
        "avatar_id": "ai_2", # Assumes "ai_2" exists
        "action": "move_to",
        "target_x": -5.0,
        "target_y": 8.0,
        "target_z": 0.0
    }
    print(f"\n--- Sending Command 2 ---")
    if client.send_command(cmd2):
        print("Command 2 sent successfully and OK received.")
    else:
        print("Failed to send Command 2 or non-OK response received.")

    # Example 3: Command for a non-existent avatar (server should handle gracefully)
    cmd3 = {
        "avatar_id": "ai_non_existent",
        "action": "move_to",
        "target_x": 1.0,
        "target_y": 1.0,
        "target_z": 1.0
    }
    print(f"\n--- Sending Command 3 (non-existent avatar) ---")
    if client.send_command(cmd3): # Server should still respond OK if command format is valid
        print("Command 3 sent, server responded OK (as expected for valid format).")
    else:
        print("Failed to send Command 3.")

    # Example 4: Invalid command format (e.g., missing action)
    cmd4 = {
        "avatar_id": "ai_1"
        # "action": "move_to", # Missing action
    }
    print(f"\n--- Sending Command 4 (invalid format) ---")
    if client.send_command(cmd4): # Server might respond ERROR
        print("Command 4 sent, but it was invalid. Check server response log in client output.")
    else:
        # This 'else' might be hit if the server sends "ERROR" and send_command expects "OK"
        print("Command 4 sent (invalid), server likely responded with an error (as expected).")

    print("\nGameEngineClient example usage finished.")

# ```
