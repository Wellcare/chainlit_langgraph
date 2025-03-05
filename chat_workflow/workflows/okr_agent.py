import chainlit as cl
from chainlit.input_widget import Select
from langgraph.graph import StateGraph
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from .base import BaseWorkflow, BaseState
from ..llm import llm_factory, ModelCapability
from ..tools import BasicToolNode
from ..tools.gsheet import get_gsheet_tool  # Import the Google Sheets tool
from ..tools.time import get_datetime_now


class GraphState(BaseState):
    # Model name of the chatbot
    chat_model: str
    # OKR data retrieved from Google Sheets
    okr_data: list


class OKRChatWorkflow(BaseWorkflow):
    def __init__(self):
        super().__init__()

        # Define capabilities and tools
        self.capabilities = {
            ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING
        }
        self.tools = [get_datetime_now] + get_gsheet_tool()  # Add Google Sheets tool

    def create_graph(self) -> StateGraph:
        """
        Create the LangGraph state machine for the OKR Chat Agent.
        """
        graph = StateGraph(GraphState)

        # Add nodes
        graph.add_node("chat", self.chat_node)
        graph.add_node("tools", BasicToolNode(self.tools))

        # Define the workflow
        graph.set_entry_point("chat")
        graph.add_conditional_edges("chat", self.tool_routing)
        graph.add_edge("tools", "chat")

        return graph

    async def chat_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        """
        Handle the chat interaction with the user.
        """
        # Define the prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an OKR Chat Assistant. Help users query and understand OKR data for the company."),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # Initialize the LLM with tools
        llm = llm_factory.create_model(
            self.output_chat_model, model=state["chat_model"], tools=self.tools
        )

        # Create the chain
        chain: Runnable = prompt | llm

        # Invoke the chain and update the state
        return {
            "messages": [await chain.ainvoke(state, config=config)],
            "okr_data": state.get("okr_data", [])  # Preserve OKR data in the state
        }

    def create_default_state(self) -> GraphState:
        """
        Create the default state for the workflow.
        """
        return {
            "name": self.name(),
            "messages": [],
            "chat_model": "",
            "okr_data": [],  # Initialize empty OKR data
        }

    @classmethod
    def name(cls) -> str:
        """
        Return the name of the workflow.
        """
        return "OKR Chat"

    @property
    def output_chat_model(self) -> str:
        """
        Return the output chat model key.
        """
        return "chat_model"

    @classmethod
    def chat_profile(cls) -> cl.ChatProfile:
        """
        Define the chat profile for the OKR Chat Agent.
        """
        return cl.ChatProfile(
            name=cls.name(),
            markdown_description="An AI-powered assistant for querying and understanding OKR data.",
            icon="https://cdn1.iconfinder.com/data/icons/3d-front-color/128/chat-text-front-color.png",
            default=True,
            starters=[
                cl.Starter(
                    label="What are the company’s Q1 objectives?",
                    message="What are the company’s Q1 objectives?",
                    icon="https://cdn1.iconfinder.com/data/icons/3d-dynamic-color/128/target-dynamic-color.png",
                ),
                cl.Starter(
                    label="Show me the key results for the Sales department.",
                    message="Show me the key results for the Sales department.",
                    icon="https://cdn1.iconfinder.com/data/icons/3d-dynamic-color/128/chart-dynamic-color.png",
                ),
                cl.Starter(
                    label="Summarize our overall OKR progress for the year.",
                    message="Summarize our overall OKR progress for the year.",
                    icon="https://cdn1.iconfinder.com/data/icons/3d-dynamic-color/128/calendar-dynamic-color.png",
                ),
            ],
        )

    @property
    def chat_settings(self) -> cl.ChatSettings:
        """
        Define the chat settings for the OKR Chat Agent.
        """
        return cl.ChatSettings([
            Select(
                id="chat_model",
                label="Chat Model",
                values=sorted(llm_factory.list_models(capabilities=self.capabilities)),
                initial_index=0,
            ),
        ])