from typing import Callable, Dict, Any, List
import json
from litellm import completion
from pydantic import BaseModel

from agorama.models import ChatMessage

def action(func: Callable) -> Callable:
    """
    Decorator to mark a method as an action.
    """
    func._is_action = True
    return func

class FunctionParameter(BaseModel):
    type: str # the type of the parameter
    description: str # a description of what the parameter is
    required: bool # whether the parameter is required

class ToolClass(BaseModel):
    name: str
    description: str
    parameters: Dict[str, FunctionParameter]


    def payload(self):
        required_parameters = [param_name for param_name, param in self.parameters.items() if param.required]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param_name: {
                            "type": param.type,
                            "description": param.description
                        }
                        for param_name, param in self.parameters.items()
                    },
                    "required": required_parameters
                }
            }
        }

class BasePerceptionAgent:
    """
    This is a perception agent. This agent maintains a `state` dictionary that is used to store information about the world. 
    A variety of helper functions are provided but the key flow is:
        1. `perceive` - observe the world and update the state, this function *reads* a series of keys from the state dictionary
        2. `decide` - use the retrieved state context from `perceive` to decide *what* actions to take.
        3. `act` - take the decided actions and update the state.

    To support this there are a few key member variables that must be set:
        - `state` - a dictionary of k:v pairs that represent the state of the world *to the agent*
        - `actions` - a dictionary of k:v pairs with keys being the name for actions, and values being a *functor* that takes the current state and returns a new state.
            - Action values might subsequently be factored into a class that implements reprs and prompting helpers for LLMs
        - `llm` - a language model that can be used to decide actions; this must implement an OAI-style interface
        - `history` - a debugging tool to keep track of the history of the agent's actions, state, and decisions.
            - This is a list of dictionaries with keys: 'state', 'action', 'decision', 'observation' (where observation is the set of keys *read* from the state dictionary)

    The perception-action cycle works as follows:
        1. The agent first calls `perceive()` which reads the current chat room state and updates relevant entries in `self.state`
        2. The agent then calls `decide()` which uses the LLM to determine what action to take based on the perceived state
        3. Finally, `act()` executes the chosen action by calling the appropriate functor from `self.actions`
            - This function takes in relevant kwargs, mutates the state as it sees fit, and returns the new state

    The outputs from each of the above functions are fed into the next perceive -> decide -> act cycle.
        
    This cycle allows the agent to maintain an internal model of the world state and make decisions based on that model.
    The separation of perception, decision making and action allows for modular development and testing of each component.

    Subclasses must implement:
        - `perceive()` - Define what aspects of the chat room to observe
        - `decide()` - Implement the decision making logic using the LLM
        - Define the available actions in `self.actions` (you can use the `@action` decorator to help with this)
    Base class implements:
        - `run()` a convenience function that runs the perception -> decision -> act cycle
    """
    def __init__(self, name: str):
        self.name = name
        self.state: Dict[str, Any] = {}
        self.actions: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.history: List[Dict[str, Any]] = []
        # Automatically register methods decorated with @action as actions
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and getattr(attr, '_is_action', False):
                self.actions[attr_name] = attr

    def run_iteration(self):
        """
        Run one cycle of the perception-action loop.
        """
        perception = self.perceive()
        decision = self.decide()
        action_result = self.act(decision)
        self.history.append({
            'state': self.state, # does this need to be deep copied?
            'decision': decision,
            'action': action_result,
            'perception': perception
        })
        return action_result

    def perceive(self):
        """
        Subclasses must implement this method.
        It should observe the world, update self.state, and return the observed keys.
        """
        raise NotImplementedError("Subclasses must implement 'perceive()'.")
    
    @action # this is a base noop action that all agents should have
    def noop(self, **kwargs):
        """
        This is a no-op action that returns the state unchanged.
        """
        return

    def decide(self):
        """
        Subclasses must implement this method.
        It should use self.llm and the current self.state to decide on an action.
        Return the action name to be executed.
        """
        raise NotImplementedError("Subclasses must implement 'decide()'.")

    def act(self, action_name: str):
        """
        Execute the action corresponding to action_name.
        The action callable takes the current state and returns a new state.
        """
        if action_name not in self.actions:
            raise ValueError(f"Action '{action_name}' is not defined in this agent.")
        return self.actions[action_name]() # the action always just gets the state

class MemoryKeyAccess(BaseModel):
    """
    This is a helper class to store the memory key access information
    """
    key_list: List[str]

class ChatAgentWithMemory(BasePerceptionAgent):
    """
    This agent uses a memory to store important information about the chat history
    and can perform memory operations (store, retrieve, remove) based on the conversation.
    """
    def __init__(self, name: str, llm_model: str = "gpt-4o-mini"):
        super().__init__(name)
        self.llm_model = llm_model
        self.state['memory'] = {}  # Changed to dict for key-value storage
        self.system_prompts = {
            "base_prompt": """You are a helpful AI assistant with memory capabilities. You can:
            1. Store important information in your memory bank
            2. Retrieve relevant information from memory when needed
            3. Remove outdated or incorrect information
            4. Respond to user queries using both current context and memory
            """,
            "memory_prompt": """Return a list of keys from {memory_keys} that are relevant to the conversation.
            If there are no relevant keys, return an empty list.""",
            
            "decision_prompt": """Based on the current conversation and memory state, what action should you take?
            Available actions:
            1. `store` - Store new information in memory
            2. `remove` - Remove information from memory
            3. `respond` - Send a message to the user
            """,

            "respond_prompt": """Based on the current conversation and memory state, respond to the user."""
        }

        self.state['action_methods'] = [
            ToolClass(
                name="store",
                description="Store important information in memory",
                parameters={
                    "key": FunctionParameter(type="string", description="The key to store the information under", required=True),
                    "value": FunctionParameter(type="string", description="The information to store", required=True)
                }
            ),
            ToolClass(
                name="remove",
                description="Remove information from memory",
                parameters={
                    "key": FunctionParameter(type="string", description="The key to remove the information from", required=True)
                }
            ),
            ToolClass(
                name="respond",
                description="Send a message to the user",
                parameters={}
            ),
            ToolClass(
                name="noop",
                description="Do nothing",
                parameters={}
            )
        ]

        self.state['chat_history'] = []
        self.state['contextual_memory'] = {}

    def perceive(self):
        """
        """
        # Get memory keys for the LLM to consider
        memory_keys = list(self.state['memory'].keys())
        
        # Get LLM decision about memory access
        memory_decision = completion(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompts["base_prompt"]},
                {"role": "user", "content": self.system_prompts["memory_prompt"].format(memory_keys=memory_keys)}
            ] + 
            [val.to_oai_dict() for val in self.state['chat_history']], # append the chat history to the system prompt
            response_format=MemoryKeyAccess
        )

        memory_response = MemoryKeyAccess.model_validate_json(memory_decision.choices[0].message.content)

        if len(memory_response.key_list) > 0:
            contextual_memory = {}
            for key in memory_response.key_list:
                contextual_memory[key] = self.state['memory'][key]
            self.state['contextual_memory'] = contextual_memory
        else:
            self.state['contextual_memory'] = {} # set it to empty


    def decide(self):
        """
        Decides what action to take based on the current state and memory.
        Returns the name of the action to execute.
        """
        self.state['action_payload'] = {}
        # get your contextual memory
        contextual_memory = ChatMessage(
            message="## CONTEXTUAL MEMORY\n" + "\n".join([f"{key}: {self.state['contextual_memory'][key]}" for key in self.state['contextual_memory']]),
            created_by="assistant",
        )
        
        # Get LLM decision
        decision = completion(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompts["base_prompt"]},
                {"role": "system", "content": self.system_prompts["decision_prompt"]}
            ] + 
            [message.to_oai_dict() for message in self.state['chat_history']] + 
            [contextual_memory.to_oai_dict()],
            tools=[action.payload() for action in self.state['action_methods']]
        )

        function_call = decision.choices[0].message.tool_calls[0].function
        name, arguments = function_call['name'], function_call['arguments']
        self.state['action_payload'] = json.loads(arguments) # set the action payload
        return name

    @action
    def store(self):
        """
        Stores information in the memory bank.
        """
        key, value = self.state['action_payload']['key'], self.state['action_payload']['value']
        self.state['memory'][key] = value
        return f"Stored '{key}' in memory"

    @action
    def remove(self):
        """
        Removes information from the memory bank.
        """
        remove_key = self.state['action_payload']['key']
        if remove_key in self.state['memory']:
            del self.state['memory'][remove_key]
            return f"Removed '{remove_key}' from memory"
        else:
            return f"Key '{remove_key}' not found in memory"
        
    @action
    def respond(self):
        """
        Sends a message to the user.
        """
        contextual_memory = ChatMessage(
            message="## CONTEXTUAL MEMORY\n" + "\n".join([f"{key}: {self.state['contextual_memory'][key]}" for key in self.state['contextual_memory']]),
            created_by="assistant",
        )

        message_list = [
            {"role": "system", "content": self.system_prompts["base_prompt"]},
            {"role": "system", "content": self.system_prompts["respond_prompt"]}
        ]

        message_list.extend([message.to_oai_dict() for message in self.state['chat_history']])
        message_list.append(contextual_memory.to_oai_dict())

        response = completion(
            model=self.llm_model,
            messages=message_list
        )

        self.state['chat_history'].append(ChatMessage(message=response.choices[0].message.content, created_by="assistant"))

        return response.choices[0].message.content
        

# Example subclass implementation for demonstration purposes
class SimpleAgent(BasePerceptionAgent):
    def perceive(self):
        # For demonstration, simply add a dummy sensor value
        self.state['sensor'] = 42
        # Return the observed keys
        return list(self.state.keys())

    def decide(self):
        # Decision logic: if sensor value is 42, choose action 'do_nothing'
        return 'do_nothing'

    @action
    def do_nothing(self, state: Dict[str, Any]):
        # Action that does nothing, simply returns the state unchanged
        return state


if __name__ == "__main__":
    agent = SimpleAgent("TestAgent")
    new_state = agent.run_iteration()
    print("New state:", new_state)
    print("History:", agent.history) 