import asyncio
import time
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from tqdm import trange
import yaml
from agorama.models import ChatMessage, ChatRoom
from agorama.reasoning_agent import ReasoningAgent

class YamlAgorama:
    # Existing implementation

class JsonAgorama:
    # Existing implementation

class ReasoningYamlAgorama:
    """
    An Agorama is the runner for a chat room.
    Instantiated from a yaml file. This expcts all the agents to be `LMAgent`s
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
            raise ValueError("agents must be defined in the yaml file")
        
        self.agents = [ReasoningAgent(**agent_dict) for agent_dict in loaded["agents"] if "model_hub_pair" in agent_dict]
# Ensure that the ReasoningAgent is used in the YamlAgorama class
        
        self.chat_room = ChatRoom(room_name="Agorama")

        # let us also setup initial state for the chat room
        if "initial_state" in loaded:
            for chat_message in loaded["initial_state"]:
                self.chat_room.add_message(ChatMessage(**chat_message))

    async def tick(self):
        """
        One tick of the agorama. Calls all the agents in parallel and adds their messages to the chat room.
        """
        tasks = [agent.act(self.chat_room) for agent in self.agents]
        responses = await asyncio.gather(*tasks)
        for response in sorted(responses, key=lambda x: x.created_at, reverse=True): # we get the responses in sorted order with the largest timestamp last
            self.chat_room.add_message(response)

    def show(self):
        print(self.chat_room)

    # this is a debug function:
    async def run_iters(self, num_iters: int):
        for _ in trange(num_iters):
            await self.tick()
            time.sleep(1)

        self.show()