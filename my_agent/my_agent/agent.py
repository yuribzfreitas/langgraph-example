from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
import os
from dotenv import load_dotenv  # Certifique-se de carregar dotenv
from langchain_openai import AzureChatOpenAI
from IPython.display import Image, display

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY']= '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'

# Obter as configurações do .env ou diretamente do ambiente
openai_api_version = os.getenv("OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
azure_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Configurar o LLM usando as variáveis de ambiente para o OpenAI via Azure
model = AzureChatOpenAI(
    openai_api_version=openai_api_version,
    deployment_name=deployment_name,
    azure_endpoint=azure_endpoint,
    api_key=api_key,
)

# Define diferentes personas através de prompts customizados
personas = {
    "friendly": "You are a friendly and cheerful assistant. Respond in a warm, welcoming tone.",
    "professional": "You are a highly professional assistant. Respond with formality and precision.",
    "humorous": "You are a humorous assistant. Respond with witty comments and a lighthearted tone."
}

# Função para escolher a persona e preparar a mensagem
def get_personalized_prompt(persona: str, user_message: str) -> HumanMessage:
    if persona in personas:
        prompt = personas[persona] + f" The user says: '{user_message}'. Respond accordingly."
    else:
        prompt = f"The user says: '{user_message}'. Respond naturally."
    return HumanMessage(content=prompt)

# Função que determina se deve continuar ou parar
def should_continue(state: MessagesState) -> Literal[END]:
    messages = state['messages']
    last_message = messages[-1]
    # Decisão de continuar ou encerrar o atendimento
    return END

# Funções para etapas do atendimento
def greeting_stage(state: MessagesState):
    personalized_message = get_personalized_prompt("friendly", "Olá, como posso ajudar?")
    response = model.invoke([personalized_message])
    return {"messages": [response]}

def info_collection_stage(state: MessagesState):
    personalized_message = get_personalized_prompt("professional", "Posso coletar algumas informações?")
    response = model.invoke([personalized_message])
    return {"messages": [response]}

# Função de decisão com 3 opções
def decision_stage(state: MessagesState):
    personalized_message = get_personalized_prompt("professional", "Agora precisamos decidir a melhor direção.")
    response = model.invoke([personalized_message])
    return {"messages": [response]}

# Tomada de decisão - escolha entre 3 direções
def decision_logic(state: MessagesState) -> Literal["option1", "option2", "option3"]:
    messages = state['messages']
    user_choice = messages[-1].content.lower()
    if "opção 1" in user_choice:
        return "option1"
    elif "opção 2" in user_choice:
        return "option2"
    else:
        return "option3"

def option1_stage(state: MessagesState):
    personalized_message = get_personalized_prompt("friendly", "Você escolheu a opção 1. Vamos seguir por esse caminho.")
    response = model.invoke([personalized_message])
    return {"messages": [response]}

def option2_stage(state: MessagesState):
    personalized_message = get_personalized_prompt("friendly", "Você escolheu a opção 2. Vamos seguir por esse caminho.")
    response = model.invoke([personalized_message])
    return {"messages": [response]}

def option3_stage(state: MessagesState):
    personalized_message = get_personalized_prompt("friendly", "Você escolheu a opção 3. Vamos seguir por esse caminho.")
    response = model.invoke([personalized_message])
    return {"messages": [response]}

def closing_stage(state: MessagesState):
    personalized_message = get_personalized_prompt("friendly", "Atendimento finalizado. Tenha um ótimo dia!")
    response = model.invoke([personalized_message])
    return {"messages": [response]}

# Define o gráfico de estados com etapas e decisão
workflow = StateGraph(MessagesState)

# Adiciona as etapas ao fluxo
workflow.add_node("greeting", greeting_stage)
workflow.add_node("info_collection", info_collection_stage)
workflow.add_node("decision", decision_stage)
workflow.add_node("option1", option1_stage)
workflow.add_node("option2", option2_stage)
workflow.add_node("option3", option3_stage)
workflow.add_node("closing", closing_stage)

# Define o ponto de entrada como `greeting`
workflow.add_edge(START, "greeting")

# Adiciona as transições entre as etapas
workflow.add_edge("greeting", "info_collection")
workflow.add_edge("info_collection", "decision")

# Adiciona a decisão lógica que escolherá uma das três opções
workflow.add_conditional_edges("decision", decision_logic)
workflow.add_edge("option1", "closing")
workflow.add_edge("option2", "closing")
workflow.add_edge("option3", "closing")

# Inicializa a memória para persistir o estado
checkpointer = MemorySaver()

# Compila o gráfico com as etapas
app = workflow.compile(checkpointer=checkpointer)

# Usa o gráfico com as etapas de atendimento
final_state = app.invoke(
    {"messages": [HumanMessage(content="Iniciar atendimento")]},  # A mensagem do usuário
    config={"configurable": {"thread_id": 45}}
)

print(final_state["messages"][-1].content)

# Salvar o gráfico em uma imagem
image_path = 'graph_output.png'

# Gera e salva a imagem
with open(image_path, 'wb') as f:
    f.write(app.get_graph().draw_mermaid_png())

print(f"Imagem salva em: {image_path}")
