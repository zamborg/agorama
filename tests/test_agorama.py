import unittest
from datetime import datetime, timezone
from agorama import ChatMessage, ChatRoom, LiteLLMAgent, YamlAgorama

class TestChatMessage(unittest.TestCase):
    def test_message_creation(self):
        message = ChatMessage(message="Hello", created_by="User")
        self.assertEqual(message.message, "Hello")
        self.assertEqual(message.created_by, "User")
        self.assertIsInstance(message.created_at, datetime)

class TestChatRoom(unittest.TestCase):
    def setUp(self):
        self.chat_room = ChatRoom(room_name="Test Room")

    def test_add_message(self):
        message = ChatMessage(message="Hello", created_by="User")
        self.chat_room.add_message(message)
        self.assertEqual(len(self.chat_room.messages), 1)

    def test_to_yaml_and_from_yaml(self):
        message = ChatMessage(message="Hello", created_by="User")
        self.chat_room.add_message(message)
        self.chat_room.to_yaml("test_room.yaml")
        loaded_chat_room = ChatRoom.from_yaml("test_room.yaml")
        self.assertEqual(len(loaded_chat_room.messages), 1)

class TestLiteLLMAgent(unittest.TestCase):
    def test_agent_initialization(self):
        # This is a placeholder test. Actual test would require a valid YAML file and model.
        pass

class TestYamlAgorama(unittest.TestCase):
    def test_agorama_initialization(self):
        # This is a placeholder test. Actual test would require a valid YAML file.
        pass

if __name__ == '__main__':
    unittest.main()
