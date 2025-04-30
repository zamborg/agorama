from abc import ABC, abstractmethod
from dataclasses import field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import yaml

from pydantic.dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

from litellm import acompletion
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """
    A message in the Agorama
    """
    message: str
    created_by: str
    #datetime is going to bite me in the ass huh
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def __str__(self):
        return f"{self.created_by} : {self.message}"
    
    def to_oai_dict(self) -> Dict[str, Any]:
        return {
            "role": self.created_by,
            "content": self.message,
        }
    
    def to_model_response(self, model_name: Optional[str] = None) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content=self.message)],
            timestamp=self.created_at,
            model_name=model_name
        )

@dataclass
class ChatRoom:
    """
    A room in the Agorama
    """
    room_name: str
    messages: List[ChatMessage] = field(default_factory=list)

    def add_message(self, message: ChatMessage):
        self.messages.append(message)

    def add_messages(self, messages: List[ChatMessage]):
        # extend the message list by the created_at field
        self.messages.extend(
            sorted(messages, key=lambda x: x.created_at)
        )

    def __str__(self):
        return "\n".join(str(message) for message in self.messages)
    
    def to_yaml(self, file_name: str):
        with open(file_name, "w") as f:
            yaml.dump(asdict(self), f)

    @staticmethod
    def from_yaml(yaml_file: str) -> "ChatRoom":
        with open(yaml_file, 'r') as file:
            yaml_data: dict = yaml.safe_load(file)
        return ChatRoom(
            room_name=yaml_data["room_name"],
            messages=[ChatMessage(**message) for message in yaml_data["messages"]]
        )

class BaseAgent(ABC):
    """
    A base class for all agents
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def act(self, chat_room: ChatRoom) -> ChatMessage:
        """
        An agent's action in the chat room
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def __str__(self):
        return f"{self.name}"

class YamlAgent(BaseAgent):
    """
    An agent that uses a yaml file to store its configuration
    """
    def __init__(self, yaml_file: str, name: Optional[str] = None):
        with open(yaml_file, 'r') as file:
            loaded_yaml = yaml.safe_load(file)
        if name is None and "name" not in loaded_yaml:
            raise ValueError("Agent name must be provided in yaml or constructor")
        if name is not None and "name" in loaded_yaml:
            logger.warning(f"{name=} provided in constructor will override yaml name: {loaded_yaml['name']}")
        super().__init__(name or loaded_yaml["name"])
        logger.warning("THIS AGENT USES YAML.SAFE_LOAD and can be dangerous")
        for key, value in loaded_yaml.items():
            setattr(self, key, value)
    
    async def act(self, chat_room: ChatRoom) -> ChatMessage:
        """
        An agent's action in the chat room
        """
        raise NotImplementedError("Subclasses must implement this method")
    

class LiteLLMAgent(YamlAgent):
    """
    An agent that uses a language model to generate a response. Maintains a chat history of its own messages.
    MUST HAVE:
        - name: str
        - model_hub_pair: str
    SHOULD HAVE:
        - system_prompt: str
        - chat_history_length: int
    """
    def __init__(self, yaml_file: str, name: Optional[str] = None, model_hub_pair: Optional[KnownModelName] = None):
        super().__init__(yaml_file, name)
        self.model_hub_pair = model_hub_pair or getattr(self, "model_hub_pair") # you best hope you're right buddy
        self.model_hub_pair = self.model_hub_pair.replace(":", "/") # replace with a / for litellm
        self.chat_history_length = getattr(self, "chat_history_length", 10)
        self.system_prompt = getattr(self, "system_prompt", None)
        self.chat_history: List[ModelMessage] = []

    async def chat(self, input_messages: Union[List[ChatMessage], str]) -> ChatMessage:
        if isinstance(input_messages, str):
            input_messages = [ChatMessage(message=input_messages, created_by=self.name, created_at=datetime.now(tz=timezone.utc))]
        messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        response = await acompletion(
            model=self.model_hub_pair,
            messages= messages + [
                {"role": "user", "content": message.message} if message.created_by != self.name 
                else {"role": "assistant", "content": message.message} 
                for message in input_messages
            ]
        )
        return response
    
    async def act(self, chat_room: ChatRoom) -> ChatMessage:
        response = await self.chat(chat_room.messages[-self.chat_history_length:])
        return ChatMessage(message=response['choices'][0]['message']['content'], created_by=self.name, created_at=datetime.now(tz=timezone.utc))

            

class PydanticLMAgent(YamlAgent):
    """
    An agent that uses a language model to generate a response. Maintains a chat history of its own messages.
    MUST HAVE:
        - name: str
        - model_hub_pair: str
    SHOULD HAVE:
        - system_prompt: str
        - chat_history_length: int
    """
    def __init__(self, yaml_file: str, name: Optional[str] = None, model_hub_pair: Optional[KnownModelName] = None):
        super().__init__(yaml_file, name)
        if model_hub_pair is None and getattr(self, "model_hub_pair", None) is None:
            raise ValueError("Model must be provided in yaml or constructor")
        
        if getattr(self, "chat_history_length", None) is None:
            logger.warning("CHAT HISTORY LENGTH NOT SET IN YAML. USING DEFAULT OF 10.")
            self.chat_history_length = 10

        self.model_hub_pair = model_hub_pair or getattr(self, "model_hub_pair")
        # now we need to instantiate a pydantic AI agent
        self.chat_history: List[ModelMessage] = []
        self.system_prompt = getattr(self, "system_prompt", None)

        if self.system_prompt is None:
            logger.info("SYSTEM PROMPT NOT SET IN YAML.")
            self.ai = Agent(model=self.model_hub_pair)
        else:
            self.ai = Agent(model=self.model_hub_pair, system_prompt=self.system_prompt)

    async def chat(self, user_prompt: str, chat_history: Optional[List[ModelMessage]] = None) -> ChatMessage:
        response = await self.ai.run(
            user_prompt=user_prompt,
            message_history=chat_history or self.chat_history
        )
        _, response_message = response.new_messages() 
        response_message: ModelResponse  # tell mypy that response_message is of type ModelMessage
        self.chat_history.append(response_message) # append to the chat history of self
        return ChatMessage(
            message=response.data,
            created_by=self.name,
            created_at=response_message.timestamp #type: ignore
        )
    
    def reset_chat_history(self):
        self.chat_history = []

    def get_chat_context(self, message_list: List[ChatMessage]) -> str:
        return "\n".join([str(message) for message in message_list[-self.chat_history_length:]])


    async def act(self, chat_room: ChatRoom) -> ChatMessage:
        # get the messages from the chat room
        context = self.get_chat_context(chat_room.messages)
        # convert the messages to ModelMessage: -- WE SKIP THIS FOR NOW
        # model_messages = [message.to_model_response(model_name=self.model_hub_pair) for message in chat_room.messages]
        # chat with the model
        response = await self.chat(
            user_prompt=context,
            # chat_history=model_messages # type: ignore
        )
        return response
    
