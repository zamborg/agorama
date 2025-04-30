"""
    ReasoningAgent is an agent that decides whether to respond based on the conversation context.
    It inherits from LiteLLMAgent and adds a reasoning step before generating a response.
    """

from pydantic import BaseModel, ValidationError
from agorama import LiteLLMAgent, ChatMessage
from typing import List, Optional

class ReasoningResponse(BaseModel):
    should_respond: bool

class ReasoningAgent(LiteLLMAgent):
    async def reason(self, chat_room_messages: List[ChatMessage]) -> bool:
        logger.info(f"{self.name} is reasoning whether to respond.")
        logger.info(f"{self.name} is reasoning whether to respond.")
        # Generate a prompt to ask if it should respond
        context = "".join([msg.message for msg in chat_room_messages])
        prompt = f"Given the following conversation: {context}, should I respond? (yes/no)"
        response = await self.chat(prompt)
        try:
            reasoning_output = ReasoningResponse(should_respond=response['choices'][0]['message']['content'].strip().lower() == 'yes')
            return reasoning_output.should_respond
        except ValidationError:
            return False  # Default to not responding if validation fails

    async def act(self, chat_room: ChatRoom) -> ChatMessage:
        # Call the reasoning method to determine if the agent should respond
        if not await self.reason(chat_room.messages):
            logger.info(f"{self.name} decided not to respond.")
            return ChatMessage(message="", created_by=self.name)  # No response
        # Call the reasoning method to determine if the agent should respond
        if not await self.reason(chat_room.messages):
            logger.info(f"{self.name} decided not to respond.")
            return ChatMessage(message="", created_by=self.name)  # No response
        if await self.reason(chat_room.messages):
            response = await self.chat(chat_room.messages[-self.chat_history_length:])
            return ChatMessage(message=response['choices'][0]['message']['content'], created_by=self.name, created_at=datetime.now(tz=timezone.utc))
        return ChatMessage(message="", created_by=self.name)  # No response