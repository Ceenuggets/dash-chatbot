from dash import Dash, html, dcc, Output, Input, State, callback, no_update
import dash_bootstrap_components as dbc

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import  find_dotenv, load_dotenv
import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",  max_output_tokens=512, google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             max_output_tokens=1024,
                             google_api_key=GOOGLE_API_KEY)
config = {"configurable": {"thread_id": "abc123"}}


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Give your best answer to all questions.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

#######################################################
# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define the function that calls the model
def call_model(state: MessagesState):
    ####### If you don't want to use ChatPromptTemplate you may use  this #####
    # response = llm.invoke(state["messages"])
   
    prompt = prompt_template.invoke(state)
    response = llm.invoke(prompt)
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
chat_app = workflow.compile(checkpointer=memory)

##########################################################

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, '/assets/chatbot.css'],)
app.layout= dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    "CeeMyne"
                ], className="title",),
                html.Div([

                    html.Div([

                    ], id="result", className="result")
                ], className="`input_result_div`"),



            ], className="parent_div"),
            html.Div([
                dcc.Textarea(id="user_input", placeholder="Ask your question here!"),
                html.Button("Answer", id="btn_answer", n_clicks=0)
            ], className="input_div"),
        ], xs=6, sm=6, md=6, lg=6, xl=6)
    ], justify="center",className="g-0")
], fluid=True)


@callback(
    Output("result", "children"),
    Output("user_input", "value"),
    Input("btn_answer", "n_clicks"),
    State("user_input", "value"),
    State("result", "children"),
)
def display_answer(n_clicks, user_input, parent_div_children):
    if user_input is not None:
        try:

            input_messages = [HumanMessage(user_input)]
            output = chat_app.invoke({"messages": input_messages},  config)
            prompt_feedback_container = html.Div([
                dcc.Markdown([
                    user_input
                ], className="user_div"),
                dcc.Markdown([
                    output["messages"][-1].content
                ], className="feedback_div")
            ], className="prompt_feedback_div")
            # print(output["messages"][-1].content)
            if not parent_div_children:
                return  [prompt_feedback_container], ""
            else:
                return parent_div_children + [prompt_feedback_container], ""
        except Exception as e:
            return html.Div(f"An error occurred: {e}")
    else:
        return no_update, no_update



if __name__ == "__main__":
    app.run_server(debug=True, port=3030)