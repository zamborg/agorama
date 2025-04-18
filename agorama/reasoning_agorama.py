"""
    ReasoningYamlAgorama is a specialized Agorama runner for reasoning agents.
    It inherits from the standard YamlAgorama and expects all agents to be `ReasoningAgent`s.
    """

import asyncio
import logging
from agorama.models import ChatMessage, ChatRoom
from agorama.reasoning_agent import ReasoningAgent

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningYamlAgorama:
    """
    A specialized Agorama runner for reasoning agents.
    Instantiated from a YAML file, this expects all agents to be `ReasoningAgent`s.
    """
    def __init__(self, yaml_file: str):
        try:
            with open(yaml_file, "r") as f:
                loaded = yaml.safe_load(f)
        except FileNotFoundError:
            raise ValueError(f"YAML file {yaml_file} not found.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {yaml_file}: {e}")
            raise yaml.YAMLError(f"Error parsing YAML file {yaml_file}: {e}")

        if "agents" not in loaded:
            raise ValueError("Agents must be defined in the YAML file.")

        self.agents = [ReasoningAgent(**agent_dict) for agent_dict in loaded["agents"]]
        self.chat_room = ChatRoom(room_name="Reasoning Agorama")

        # Set up initial state for the chat room
        if "initial_state" in loaded:
            for chat_message in loaded["initial_state"]:
                self.chat_room.add_message(ChatMessage(**chat_message))

    async def tick(self):
        """
        One tick of the reasoning agorama. Calls all the agents in parallel and adds their messages to the chat room.
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