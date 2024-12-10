from ..llm import llm_factory, ModelCapability
from ..tools import BasicToolNode
from chainlit.input_widget import Select
from chat_workflow.workflows.base import BaseWorkflow, BaseState
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import StateGraph, END
from typing import List, Literal
from typing_extensions import TypedDict
import chainlit as cl
from datetime import date


class SupervisorResponse(TypedDict):
    """Worker to route to next. If no workers are needed, route to __end__"""
    next: Literal["hospital_finder", "date_picker", "responder"]
    instruction: str

class GeneralPractionerResponse(TypedDict):
    """Structured response for hospital_finder_node."""
    chief_complaint: str  # Chief complaint provided by the user
    hospital: Literal["BV ĐH Y Dược", "BV Chợ Rẫy", "BV Ung Bướu"]  # Selected hospital based on the user's input
    next: Literal["date_picker", "responder"]
    instruction: str  # Any follow-up messages or notes

class DatePickerResponse(TypedDict):
    """Structured response for date_picker_node."""
    date: date  # Preferred date or time
    instruction: str  # Any follow-up messages or notes

# Define the State Class
class GraphState(BaseState):
    chat_model: str  # Model used by the chatbot
    chief_complaint: str  # Chief complaint
    date: str  # Preferred date or time
    hospital: str  # Selected hospital
    next: str
    instruction: str

# Define the Workflow
class HospitalBookingWorkflow(BaseWorkflow):
    def __init__(self):
        super().__init__()
        self.capabilities = {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING}
        self.tools = []

    @classmethod
    def name(self) -> str:
        return "Hospital Booking Assistant"

    @property
    def output_chat_model(self) -> str:
        return "gpt-4o-mini"  # Example: Use GPT-4 for final responses

    @classmethod
    def chat_profile(cls):
        return cl.ChatProfile(
            name=cls.name(),
            markdown_description="An assistant that helps with hospital bookings by selecting the appropriate hospital and scheduling based on the user's needs.",
            icon="https://cdn3.iconfinder.com/data/icons/hospital-outline/128/hospital-bed.png",
        )

    @property
    def chat_settings(self) -> cl.ChatSettings:
        return cl.ChatSettings([
            Select(
                id="chat_model",
                label="Chat Model",
                values=sorted(llm_factory.list_models(
                    capabilities=self.capabilities)),
                initial_index=-1,
            )
        ])

    def create_default_state(self) -> GraphState:
        # Initialize the default state variables
        return {
            "name": self.name(),
            "messages": [],
            "chat_model": "gpt-4o-mini",
            "chief_complaint": "",
            "date": "",
            "hospital": "",
            "next": "responder",
            "instruction": ""
        }

    def create_graph(self) -> StateGraph:
        # Create the state graph
        graph = StateGraph(GraphState)

        # Add nodes (agents)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("hospital_finder", self.hospital_finder_node)
        graph.add_node("date_picker", self.date_picker_node)
        graph.add_node("responder", self.responder_node)

        # Add edges (transitions) to create a directional flow
        graph.set_entry_point("supervisor")
        graph.add_conditional_edges("supervisor", lambda state: state["next"])
        graph.add_edge("hospital_finder", "date_picker")
        graph.add_edge("date_picker", "responder")
        graph.add_edge("responder", END)

        return graph


    ### Define Node Methods ###
    async def supervisor_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        print("Running: supervisor_node")
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a supervisor coordinating AI agents (`hospital_finder`, `date_picker`, `responder`) to help the user book a hospital visit.

Current progress:
- chief_complaint: {chief_complaint}
- date: {date}
- hospital: {hospital}

Instructions:
- Determine the next agent to route to, and give instruction to complete the task
  1. `hospital_finder` if the chief complaint or hospital is not specified clearly.
  2. `date_picker` if you had the hopsital but the date is not clear.
  3. `responder` if all information is complete and ready to finalize the booking.
        """)
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
        ])

        print(f"supervisor_node state: {state}")

        llm = llm_factory.create_model(
            self.output_chat_model,
            model=state["chat_model"]
        ).with_structured_output(SupervisorResponse)

        chain: Runnable = prompt | llm
        response = await chain.ainvoke(state, config=config)
        print(f"supervisor_node response: {response}")
        state.update({"next": response["next"]})
        return state

    async def hospital_finder_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        
        print("Running: hospital_finder_node")
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a hospital finder assistant, you are coordinating with the other agents helping the user book a hospital visit.
Your responsibility is to determine the user's chief complaint and suggest the appropriate hospital.

Available hospitals for booking:
- BV Chợ Rẫy: General health and emergency services.
- BV ĐH Y Dược: Specialized in internal medicine and diagnostics.
- BV Ung Bướu: Focused on cancer treatment.

Instructions:
- Update the chief complaint and hospital based on the user's input.
- Give questions to user to understand their health conditions in order to write the chief complaint meaningfully.
- Your next agents are: `date_picker` to chooese a date or `responder` to talk to user. Give instruction to the next agent.

Current progress:
- chief_complaint: {chief_complaint}
- hospital: {hospital}

If other agent had an instruction, please follow it: {instruction}
        """)
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
        ])

        llm = llm_factory.create_model(
            self.output_chat_model,
            model=state["chat_model"]
        ).with_structured_output(GeneralPractionerResponse)
        chain: Runnable = prompt | llm

        response = await chain.ainvoke(state, config=config)
        print(f"hospital_finder_node response: {response}")

        state.update(response)
        return state

    async def date_picker_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        print("Running: date_picker_node")
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a scheduling agent tasked with selecting a suitable date for the hospital visit.

Hospital schedules:
- BV Chợ Rẫy: Monday to Saturday, 8 AM to 4 PM.
- BV ĐH Y Dược: Monday to Sunday, 10 AM to 5 PM.
- BV Ung Bướu: Monday to Saturday, 8 AM to 4 PM.

Instructions:
- Ask the user for their preferred date and time.
- Verify if the chosen date fits the hospital's schedule.
- Ensure the date is in the future (now it's 4:00 PM, Monday, December 9, 2024 GMT+07).
- Your next agent is: `responder`, give him instruction to finalize the booking
- The date must be in ISO 8601 format.

Current booking details:
- date: {date}

If other agent had an instruction, please follow it: {instruction}
        """)
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
        ])

        llm = llm_factory.create_model(
            self.output_chat_model,
            model=state["chat_model"]
        ).with_structured_output(DatePickerResponse)
        chain: Runnable = prompt | llm

        response = await chain.ainvoke(state, config=config)
        print(f"date_picker_node response: {response}")

        state.update(response)
        return state


    async def responder_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        print("Running: responder_node")
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful AI Assistant finalizing the hospital booking process. 
You are coordinating with and assisted by the other agents (`hospital_finder`, `date_picker`).

Current booking details:
- Chief complaint: {chief_complaint}
- Hospital: {hospital}
- Date: {date}  

Instructions:
- If other agent had an instruction, please follow it: {instruction}
- The required info for booking details are chief complaint, hospital, and date.
- For the time, use "HH:SS DD-MMM-YYYY (GMT+07)" when communicating with the user.
- If any of the booking details is not clear or missing, ask the user.
- When all clear, confirm the booking details with the user.
- Always respond in the same language as user.
        """)
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="messages"),
        ])

        llm = llm_factory.create_model(
            self.output_chat_model,
            model=state["chat_model"]
        )
        chain: Runnable = prompt | llm

        response = await chain.ainvoke(state, config=config)
        print(f"responder_node response: {response}")
        state.update({"messages": [response]})
        return state
