"""Conversation widgets without retrieval or provider logic."""

from typing import ClassVar

from textual.binding import Binding
from textual.message import Message
from textual.widgets import Static, TextArea

from app.agents.rendering import response_renderable
from app.agents.schemas import AgentResponse


class Composer(TextArea):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "submit", "Send", priority=True),
        Binding("ctrl+o,f4", "newline", "Newline", key_display="Ctrl+O / F4"),
    ]

    class Submitted(Message):
        pass

    def action_submit(self) -> None:
        self.post_message(self.Submitted())

    def action_newline(self) -> None:
        self.insert("\n")


class AnswerView(Static):
    def __init__(self, response: AgentResponse, *, debug: bool = False):
        super().__init__(response_renderable(response, debug=debug), classes="answer")
        self.response = response
