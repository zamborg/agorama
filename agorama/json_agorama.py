"""
    JsonAgorama is a specialized Agorama runner for agents defined in JSON format.
    It inherits from the standard YamlAgorama and expects all agents to be `ReasoningAgent`s.
    """

import asyncio
import logging
import json
from agorama.models import ChatMessage, ChatRoom
from agorama.reasoning_agent import ReasoningAgent

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JsonAgorama:
    """
    A JsonAgorama is the runner for a chat room.
    Instantiated from a JSON file, this expects all agents to be `ReasoningAgent`s.
    """
    def __init__(self, json_file: str):
        try:
            with open(json_file, "r") as f:
                loaded = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"JSON file {json_file} not found.")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {json_file}: {e}")
            raise ValueError(f"Error parsing JSON file {json_file}: {e}")

        if "agents" not in loaded:
            raise ValueError("Agents must be defined in the JSON file.")

        self.agents = [ReasoningAgent(**agent_dict) for agent_dict in loaded["agents"]]
        self.chat_room = ChatRoom(room_name="Json Agorama")

        # Set up initial state for the chat room
        if "initial_state" in loaded:
            for chat_message in loaded["initial_state"]:
                self.chat_room.add_message(ChatMessage(**chat_message))

    async def tick(self):
        """
        One tick of the JsonAgorama. Calls all the agents in parallel and adds their messages to the chat room.
        """
        tasks = [agent.act(self.chat_room) for agent in self.agents]
        responses = await asyncio.gather(*tasks)
        for response in sorted(responses, key=lambda x: x.created_at, reverse=True):
            self.chat_room.add_message(response)

    def show(self):
        print(self.chat_room)

    async def run_iters(self, num_iters: int):
        for _ in range(num_iters):
            await self.tick()
            await asyncio.sleep(1)
        self.show()