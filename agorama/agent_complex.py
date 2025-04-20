import inspect
from typing import Callable, Dict, Any, List


def action(func: Callable) -> Callable:
    """
    Decorator to mark a method as an action.
    """
    func._is_action = True
    return func


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
    def __init__(self, name: str, llm: Any = None):
        self.name = name
        self.llm = llm
        self.state: Dict[str, Any] = {}
        self.actions: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.history: List[Dict[str, Any]] = []
        # Automatically register methods decorated with @action as actions
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and getattr(attr, '_is_action', False):
                self.actions[attr_name] = attr

    def run(self):
        """
        Run one cycle of the perception-action loop.
        """
        observation = self.perceive()
        decision = self.decide()
        new_state = self.act(decision)
        self.history.append({
            'state': self.state.copy(),
            'action': decision,
            'decision': decision,
            'observation': observation
        })
        return new_state

    def perceive(self):
        """
        Subclasses must implement this method.
        It should observe the world, update self.state, and return the observed keys.
        """
        raise NotImplementedError("Subclasses must implement 'perceive()'.")

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
        new_state = self.actions[action_name](self.state)
        self.state = new_state
        return new_state


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
    new_state = agent.run()
    print("New state:", new_state)
    print("History:", agent.history) 