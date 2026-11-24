import unittest
from unittest.mock import patch, MagicMock
import json
import socket
import sys
import os

# Add parent directory of 'core' to sys.path if 'core' itself isn't directly on path
# This assumes test_game_engine_client.py is in 'tests/' and game_engine_client.py is in 'core/'
# relative to the ai-assistant module root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.game_engine_client import GameEngineClient # Class to test

class TestGameEngineClient(unittest.TestCase):

    def setUp(self):
        self.host = "localhost"
        self.port = 8081
        self.client = GameEngineClient(host=self.host, port=self.port)
        self.test_command_dict = {"avatar_id": "test_av", "action": "move", "target_x": 1.0}

    @patch('socket.socket')
    def test_send_command_success(self, mock_socket_constructor):
        """Test successful command sending and receiving 'OK' response."""
        # Configure the mock socket instance
        mock_sock_instance = MagicMock()
        mock_socket_constructor.return_value = mock_sock_instance

        # Simulate server sending "OK\n"
        mock_sock_instance.recv.return_value = b"OK\n"

        success = self.client.send_command(self.test_command_dict)

        # Assertions
        mock_socket_constructor.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_sock_instance.settimeout.assert_called_once_with(5.0)
        mock_sock_instance.connect.assert_called_once_with((self.host, self.port))

        expected_json_command = json.dumps(self.test_command_dict) + "\n"
        mock_sock_instance.sendall.assert_called_once_with(expected_json_command.encode('utf-8'))

        mock_sock_instance.recv.assert_called_once_with(1024)
        mock_sock_instance.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        mock_sock_instance.close.assert_called_once()

        self.assertTrue(success, "send_command should return True on success.")

    @patch('socket.socket')
    def test_send_command_connection_error(self, mock_socket_constructor):
        """Test handling of a connection error."""
        mock_sock_instance = MagicMock()
        mock_socket_constructor.return_value = mock_sock_instance
        mock_sock_instance.connect.side_effect = socket.error("Connection refused")

        success = self.client.send_command(self.test_command_dict)

        mock_socket_constructor.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_sock_instance.connect.assert_called_once_with((self.host, self.port))
        mock_sock_instance.sendall.assert_not_called() # Should not be called if connect fails
        mock_sock_instance.close.assert_called_once() # Ensure socket is closed even on error

        self.assertFalse(success, "send_command should return False on connection error.")

    @patch('socket.socket')
    def test_send_command_send_error(self, mock_socket_constructor):
        """Test handling of an error during sendall."""
        mock_sock_instance = MagicMock()
        mock_socket_constructor.return_value = mock_sock_instance
        mock_sock_instance.sendall.side_effect = socket.error("Send failed")

        success = self.client.send_command(self.test_command_dict)

        mock_socket_constructor.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_sock_instance.connect.assert_called_once_with((self.host, self.port))
        mock_sock_instance.sendall.assert_called_once() # sendall was attempted
        mock_sock_instance.close.assert_called_once()

        self.assertFalse(success, "send_command should return False on send error.")

    @patch('socket.socket')
    def test_send_command_recv_error(self, mock_socket_constructor):
        """Test handling of an error during recv (e.g., timeout)."""
        mock_sock_instance = MagicMock()
        mock_socket_constructor.return_value = mock_sock_instance
        mock_sock_instance.recv.side_effect = socket.timeout("Recv timed out")

        success = self.client.send_command(self.test_command_dict)

        mock_socket_constructor.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_sock_instance.connect.assert_called_once_with((self.host, self.port))
        mock_sock_instance.sendall.assert_called_once()
        mock_sock_instance.recv.assert_called_once_with(1024)
        mock_sock_instance.close.assert_called_once()

        self.assertFalse(success, "send_command should return False on recv error/timeout.")

    @patch('socket.socket')
    def test_send_command_server_sends_error_response(self, mock_socket_constructor):
        """Test when server sends an ERROR response instead of 'OK'."""
        mock_sock_instance = MagicMock()
        mock_socket_constructor.return_value = mock_sock_instance
        mock_sock_instance.recv.return_value = b"ERROR: Something went wrong\n"

        success = self.client.send_command(self.test_command_dict)

        self.assertFalse(success, "send_command should return False if server response is not 'OK'.")
        mock_sock_instance.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        mock_sock_instance.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
