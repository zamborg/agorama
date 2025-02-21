import asyncio
import time
from tqdm import trange
import yaml
from agorama.models import ChatMessage, ChatRoom, LMAgent

class YamlAgorama:
    """
    An Agorama is the runner for a chat room.
    Instantiated from a yaml file. This expcts all the agents to be `LMAgent`s
    """
    def __init__(self, yaml_file: str):
        with open(yaml_file, "r") as f:
            loaded = yaml.safe_load(f)

        if "agents" not in loaded:
            raise ValueError("agents must be defined in the yaml file")
        
        self.agents = [LMAgent(**agent_dict) for agent_dict in loaded["agents"]]
        
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