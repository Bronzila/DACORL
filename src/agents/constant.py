from __future__ import annotations


class ConstantAgent:
    def act(self, state):
        return state[1].item()

    def reset(self):
        pass
