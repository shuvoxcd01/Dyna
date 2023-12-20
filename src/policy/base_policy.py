from abc import ABC, abstractmethod


class BasePolicy(ABC):
    @abstractmethod
    def get_prob(self, selected_action, state):
        pass

    @abstractmethod
    def get_action(self, state):
        pass
