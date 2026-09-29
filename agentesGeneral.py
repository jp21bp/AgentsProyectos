"""
Este archivo servira como la base para construir agentes en general

Recalque:
json.dumps() -> (JSON-)str

Secciones:
Secciones:
---C6---
* Estilo Antiguo
* LangChain Introduccion
  - Chain Basico
  - RAG Chain
  - Funcion al modelo
  - Creando fallbacks
  - LangChain interfaces
* LangChain Expression Language
  - Pydantic basico
  - Convertir Pydantic a OpenAI JSON esquema
* Extraccion y Tag
  - Cadena de Tag
  - Cadena de Extraccion
  - Ejemplo real: tagging
  - Ejemplo real: extraccion
  - Nested chains con custom funciones
* Herramientas y Routing
  - OpenAPI -> OpenAI Json str
  - Creando custom herramientas
  - "AgentActionMessageLog" vs "AgentFinish"
  - Cadena con funcion automatica
* Agente Conversacional
  - Cadena con short-term mem
  - Agent chain
  - Usando mem. largo-plazo
  - Planilla para custom tool
  - Todo junto

---- C7 ----
* ReAct Agente
  - Clase Agente
  - ReAct Loop
* LangGraph Componentes
  - Explorando Agent State
  - Creando LG Clase Agente
  - Explorando LG Agente
* Agentic Search Tool
  - Busqeudad regular, scrapping, y formateo de scrape
  - Agentic busquedad
  - Mejor visualizacion de dictionarios
* Persistence y Streaming
  - Agente con in-mem persistence
  - Haciendo streaming con {"configurable"}
  - Explorando importancia de threads en persistence
  - Haciendo Asynch
* Humano En Loop
  - Custom state aggregator
  - Agente con interrupcion
  - Implementacion HITL
  - Modifiando agent state
  - Time travel modificacion
  - Debugging LLMs
* State Snapshot Memory Agent ** Buena **
  - Planilla para custom agente
  - Explorando state de custom agente
  - Time travel con custom agente
  - Explorando historia post-time travel
  - Exploring snapshot's parent-child realtion after time travel
  - Visualize the graph
  - Modify a prev state to use in new thread
  - Exploring ".update_state.as_node" param.
* Essay Writer


"""
#########################################################################
#########################################################################
### C6 ###
#########################################################################
#########################################################################
    # Estilo Antiguo #
import openai
##### Creando alguno APIS
#### Clima API
def get_curr_weather(location:str, unit:str ='farenheit') -> str:
    # OJO: los params se pasaran como **kwargs dictionario
    # API Verdadero
    ## info_clima = WeatherAPI()
    # API simulado
    info_clima = {
        'location': location,
        'temperature': 27,
        'unit': unit,
        'forecast': ['sunny', 'windy']
    }
    return json.dumps(info_clima)

#### Ciudad API
def get_curr_city(location:str) -> str:
    # OJO: los params se pasaran como **kwargs dictionario
    # API Verdadero
    ## info_ciudad = CityAPI()
    # API Simulado
    info_ciudad = {
        'location': location,
        'longitude': 54.5,
        'latitude': 12.3
    }
    return json.dumps(info_ciudad)


##### Uniendo funciones
functions=[
    {
        "name":"get_curr_weather",
        "description": "gets weather at a given location",
        "parameters":{
            "type": "object",
            "properties":{
                "location":{
                    "type":"string",
                    "description": "city or state",                    
                },
                "unit":{
                    "type":"string",
                    "description":"temp unit",
                }
            },
            "required":["location"],
        }
    },
    {
        "name":"get_curr_city",
        "description": "gets city's long and lat",
        "parameters":{
            "type": "object",
            "properties":{
                "location":{
                    "type":"string",
                    "description": "city or state",                    
                },
            },
            "required":["location"],
        }
    },
]


##### Creando LLM antiguo

SYS_PROMPT = '''
You are a helpful assistant that will be experimenting
with new tools that the user has created
'''

messages = [    # List of Dicts
    {
        "role":"system", "content": SYS_PROMPT
    },
    {
        "role":"user", "content": "what's weather in LA?"
    },
]


##### Invocacion
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo-0613',
    messages=messages,
    functions=functions,
)

##### Guardando respuesta de LLM
messages.append(response['choices'][0]['messages'])
    # 'response['choices'][0]['messages']' es un DICT

##### Funcion seleccionada por LLM
response_msg=response['choices'][0]['messages']['function_call']

##### Convirtiendo LLM Funcion en **kwargs
args=json.loads(response_msg['arguments'])

##### Invocando herramienta EN PARTE del LLM
observation=get_curr_weather(args)

##### Siguiente interaccion con LLM
messages.append(
    {
        "role":"function",
        "name":"get_curr_weather",
        "content": observation,
    }
)

final_response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=messages,
)
















#########################################################################
#########################################################################
    # LangChain Introduccion #
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema.runnable import RunnableMap
from langchain.llms import OpenAI

##### Chain basico
#### Componentes
### Prompt
prompt = ChatPromptTemplate.from_template(
    "tell me 3 facts about {topic}"
)
### Modelo
model = ChatOpenAI()
### Formato final
output_parser = StrOutputParser()

#### Creando chain
chain_one= prompt | model | output_parser

#### Invocando Chain
response = chain_one.invoke({"topic":"bears"})




##### RAG Chain
#### VStore setup
### Creando Vstore
vectorstore = DocArrayInMemorySearch.from_texts(
    [
    'harrison worked at Kensho',
    'bears like to eat honey',
    'ducks like to swim',
    ],
    embedding=OpenAIEmbeddings(),
)
### Configurando VStore
retriever = vectorstore.as_retriever()

#### Chain Componentes
### Runnable Map
    # Para poder invocar Vstore de una pregunta
    # Separa la pregunta inicial en diff secciones
inputs = RunnableMap({
    "question": lambda x: x['question'],
    "context": lambda x: retriever.get_relevant_documents(x['question']),
})
### Prompt
template = '''
Answer the question based on the following context: {context}
Question: {question}
'''
prompt = ChatPromptTemplate.from_template(template)
### Modelo
model = ChatOpenAI()
### Formato final
output_parser = StrOutputParser()
### Cadena
chain_two = inputs | prompt | model | output_parser




##### Funcion al modelo
#### Creando funciones
### Clima API
def get_curr_weather(location:str, unit:str ='farenheit') -> str:
    # OJO: los params se pasaran como **kwargs dictionario
    # API Verdadero
    ## info_clima = WeatherAPI()
    # API simulado
    info_clima = {
        'location': location,
        'temperature': 27,
        'unit': unit,
        'forecast': ['sunny', 'windy']
    }
    return json.dumps(info_clima)

### Ciudad API
def get_curr_city(location:str) -> str:
    # OJO: los params se pasaran como **kwargs dictionario
    # API Verdadero
    ## info_ciudad = CityAPI()
    # API Simulado
    info_ciudad = {
        'location': location,
        'longitude': 54.5,
        'latitude': 12.3
    }
    return json.dumps(info_ciudad)


### Uniendo funciones
functions=[
    {
        "name":"get_curr_weather",
        "description": "gets weather at a given location",
        "parameters":{
            "type": "object",
            "properties":{
                "location":{
                    "type":"string",
                    "description": "city or state",                    
                },
                "unit":{
                    "type":"string",
                    "description":"temp unit",
                }
            },
            "required":["location"],
        }
    },
    {
        "name":"get_curr_city",
        "description": "gets city's long and lat",
        "parameters":{
            "type": "object",
            "properties":{
                "location":{
                    "type":"string",
                    "description": "city or state",                    
                },
            },
            "required":["location"],
        }
    },
]
#### Cadena Componentes
### Prompt
prompt = ChatPromptTemplate.from_messages[("human", "{input}")]
### Model with binding
model = ChatOpenAI(temperature=0).bind(functions=functions)
#### Creando Cadena
chain_three = prompt | model




##### Creando fallbacks
#### Creando cadena con modelo malo
model = OpenAI(temperature=0, max_tokens=1000, model="davinci")
bad_chain = model | json.loads
#### Creando cadena con buen modelo
newModel = ChatOpenAI(temperature=0)
good_chain = newModel | StrOutputParser() |json.loads
#### Creando cadena con fallbacks
chain_four = bad_chain.with_fallbacks([good_chain])






##### LangChain interfaces
#### "invoke[ainvoke]"
chain_one.invoke({"topic": "bears"})
#### "batch[abatch]"
chain_one.invoke({"topic": "bears"})
#### "stream[astream]"
for t in chain_one.stream({"topic": "bears"}):
    print(t)






















#########################################################################
#################################################################
    # LangChain Expression Language #
from pydantic import BaseModel
from typing import List
from pydantic import Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function as conversion
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI



##### Pydantic basico
#### BaseModels
class Student(BaseModel):
    name: str
    age: int
    email: str
class Classroom(BaseModel):
    students: List[Student]
#### invocando BaseModels
class_one = Classroom(
    students=[
        Student(name="Jane", age=32, email="jane@email.com"),
        Student(name="John", age=33, email="john@email.com"),
    ]
)



##### Convertir Pydantic a OpenAI JSON esquema
#### Crear Pydantic Modelos
class WeatherSearch(BaseModel):
    """Gets weather at an airport code"""
    airport_code: str = Field(description="aircode to get weather for")

class ArtistSeach(BaseModel):
    """Songs by a particular artist"""
    artist_name: str = Field(description='artist name')
    n: int = Field(description='number of results')
#### Convertir Pydantic a funciones
artist_json_str = conversion(ArtistSeach)   # Output = JSON-str
weather_json_str = conversion(WeatherSearch)
    # Output
        #{
        #'name':'WeatherSearch', 'description':'Gets weather at an airport code',
        #'parameters':
            #{
            #'title':'WeatherSearch', 'description':'Gets weather at an airport code',
            #'type':'object', 'properties':
                #{
                #'airport_code':
                    #{
                    #'title':Airport Code', 'description':'aircode to get weather for',
                    #'type':'string'
                    #}    
                #}, 'required':['airport_code']
            #}
        #}
#### Cadena Componentes
### Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant"),
    ("user", "{input}")
])
### Modelo con funciones
model = OpenAI().bind(functions = [artist_json_str, weather_json_str])
#### Cadena
chain_one = prompt | model
#### INvocaciones
chain_one.invoke('weather in SF?')
chain_one.invoke('three songs by taylor swift')
chain_one.invoke('hi!')















#########################################################################
#############################################################
    # Extraccion y Tag #
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from typing import Optional
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionParser
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableLambda


##### Cadena de Tagging
#### Pydantic model
class Tagging(BaseModel):
    """Tag text with desired information"""
    sentiment: str = Field(description='text sentiment')
    language: str = Field(description='text language')
#### Pydantic -> OpenAI JSON str
tag_json_str = conversion(Tagging)
#### Cadena Componentes
### Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system","Think and tag the text as instructed"),
    ("user", "{input}")
])
### Modelo con fcn
model = ChatOpenAI().bind(
    functions=tag_json_str,
    function_call={'name':'Tagging'},  
        # FORZANDO un fcn call
        # Bueno para debugging
)
#### Cadena
tag_chain = prompt | model | JsonOutputFunctionsParser()
    # Parser: JSON blob -> human readible form



##### Cadena de Extraccion
#### Pydantic Modelos
class Person(BaseModel):
    """Info about ONE person"""
    name: str = Field(description="person name")
    age: Optional[int] = Field(description="person's age")
class Info(BaseModel):
    """All information desired to be extracted"""
    people: List[Person] = Field(description="list of people information")
#### Pydantic -> OpenAI json str
extract_json_str = [conversion(Info)]
#### Cadena Componentes
### Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract relevant information, do not guess"),
    ("human", "{input}")
])
### Model c/ fcn

extract_model = model.bind(
    functions=extract_json_str,
    function_call={"name":"Info"}
)
#### Crear cadena
extract_chain = prompt | extract_model | JsonKeyOutputFunctionParser(key_name='people')



##### Ejemplo real: Tagging
#### Extrayendo blog
loader = WebBaseLoader("https://...")
documents = loader.load()
doc = documents[0]
page_content = doc.page_content[:10000]
    #Only retrieves the first 10000 characters
#### Creando Pydantic del Taggin objetivo
class Overview(BaseModel):
    """Overview of a section of text"""
    summary: str = Field(description="Provide concise summary of context")
    langauge: str = Field(description="Provide context's language")
    keywords: str = Field(description="Provide context's keywords")
overview_json_str = conversion(Overview)
#### Creando cadena
prompt = ChatPromptTemplate.from_messages([
    ("system","Extract relevant info, do not guess"),
    ("human", "{input}")
])
tag_model = model.bind(functions=overview_json_str,
    function_call={"name":"Overview"})
tag_chain = prompt | tag_model | JsonOutputFunctionsParser()



##### Ejemplo real: extraccion
#### Pydantic models
class Paper(BaseModel):
    """Info about papers mentioned"""
    title: str
    author: Optional[str]
class Info(BaseModel):
    """All info desired to be extracted"""
    papers: List[Paper]
info_json_str = conversion(Info)
#### Creando cadena
prompt = ChatPromptTemplate.from_messages([
    ("system","Extract relevant info, do not guess"),
    ("human", "{input}")
])
extract_model=model.bind(functions=info_json_str,
    function_call={"name":"Info"})
extract_chain = prompt | extract_model | JsonKeyOutputFunctionParser(key_name="papers")
#### Invocacion
page_content = doc.page_content[:10000]
extract_chain.invoke({"input": page_content})
#### Mejorando cadena
template = """
    An article will be passed to you. Extract from it all papers
    that are mentioned in the article. Do not extract the name 
    of the article itself.
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}")
])
extract_chain = prompt | extract_model | JsonKeyOutputFunctionParser(key_name="papers")



##### Nested chains con custom fcnes
#### Separando un grande texto
splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)
splits = splitter.split_text(doc.page_content) #Returns a list
#### Creando una funcopn que flatting una lista de listas
def flatten(matrix):
    flat_list = []
    for row in matrix: flat_list += row
    return flat_list
#### Cadena componente: preparacion
    # Prepara los differentes chunks del grande texto
prep = RunnableLambda(
    lambda x : [{"input": doc} for doc in splitter.split_text(x)]
)
    #"x" = entire blog itself, including blog's text contents
#### Creando nested cadena
nested_cadena = prep | extract_chain.map() | flatten
    # Recall: "extract_chain" is applied to a single dict input
        # But "prep" returns a list of dicts
        # Thus we need ".map()" to apply "extract_chain" to each dict indiividually
    # "extract_chain" returns a single list output for a single dict input
        # then "extract_chain.map()" will return a list of lists
        # Thus "flatten" is needed to turn list into a single list



























#########################################################################
############################################################
    # Herramientas y Routing #
from langchain.tools import tool
from pydantic import BaseModel, Field
import requests, datetime
import wikipedia
from langchain.tools.render import format_tool_to_openai_function
from langchain.chains.openai_functions.openapi import openapi_spec_to_openai_fn
    #Used for conversion
from langchain.utilities.openapi import OpenAPISpec
    # USed to load the Open API spec
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.agent import AgentFinish




##### OpenAPI -> OpenAI Json str
#### Create sample Open API spec with multiple paths and endpoints
text = """
{
  "openapi": "3.0.0",
  "info": {
    "version": "1.0.0",
    "title": "Swagger Petstore",
    "license": {
      "name": "MIT"
    }
  },
  "servers": [
    {
      "url": "http://petstore.swagger.io/v1"
    }
  ],
  "paths": {
    "/pets": {
      "get": {
        "summary": "List all pets",
        "operationId": "listPets",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "description": "How many items to return at one time (max 100)",
            "required": false,
            "schema": {
              "type": "integer",
              "maximum": 100,
              "format": "int32"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "A paged array of pets",
            "headers": {
              "x-next": {
                "description": "A link to the next page of responses",
                "schema": {
                  "type": "string"
                }
              }
            },
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pets"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      },
      "post": {
        "summary": "Create a pet",
        "operationId": "createPets",
        "tags": [
          "pets"
        ],
        "responses": {
          "201": {
            "description": "Null response"
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    },
    "/pets/{petId}": {
      "get": {
        "summary": "Info for a specific pet",
        "operationId": "showPetById",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "required": true,
            "description": "The id of the pet to retrieve",
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Expected response to a valid request",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pet"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "Pet": {
        "type": "object",
        "required": [
          "id",
          "name"
        ],
        "properties": {
          "id": {
            "type": "integer",
            "format": "int64"
          },
          "name": {
            "type": "string"
          },
          "tag": {
            "type": "string"
          }
        }
      },
      "Pets": {
        "type": "array",
        "maxItems": 100,
        "items": {
          "$ref": "#/components/schemas/Pet"
        }
      },
      "Error": {
        "type": "object",
        "required": [
          "code",
          "message"
        ],
        "properties": {
          "code": {
            "type": "integer",
            "format": "int32"
          },
          "message": {
            "type": "string"
          }
        }
      }
    }
  }
}
"""
#### Load Open API Spec
spec = OpenAPISpec.from_text(text)
#### Converting OpenAPI Spec
pet_openai_functions, pet_callables = openapi_spec_to_openai_fn(spec)
print(pet_openai_functions)
    #Ouput:
    # [{'name': 'listPets',
    #   'description': 'List all pets',
    #   'parameters': {'type': 'object',
    #    'properties': {'params': {'type': 'object',
    #      'properties': {'limit': {'type': 'integer',
    #        'maximum': 100.0,
    #        'schema_format': 'int32',
    #        'description': 'How many items to return at one time (max 100)'}},
    #      'required': []}}}},
    #  {'name': 'createPets',
    #   'description': 'Create a pet',
    #   'parameters': {'type': 'object', 'properties': {}}},
    #  {'name': 'showPetById',
    #   'description': 'Info for a specific pet',
    #   'parameters': {'type': 'object',
    #    'properties': {'path_params': {'type': 'object',
    #      'properties': {'petId': {'type': 'string',
    #        'description': 'The id of the pet to retrieve'}},
    #      'required': ['petId']}}}}]
#### Creando un modelo
model = ChatOpenAI(temperature=0).bind(functions=pet_openai_functions)
model.invoke("what are three pets names")
model.invoke("tell me about pet with id 42")



##### Creando custom herramientas
#### Tool 1 - Open Meteo API
### Pydantic class
class OpenMeteoInput(BaseModel):
    latitude: float = Field(desciption='latitude of location weather')
    longitude: float = Field(desciption='longitude of location weather')
### Tool itself
@tool(args_schema=OpenMeteoInput)
def get_curr_temp(latitude:float, longitude:float):
    """Fetch currente temperature for a given coordinate"""
    BASE_URL = "https://..."
    params={    #Request params
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }
    response = requests.get(BASE_URL, params=params)
        #ACtually makes the request
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f'APU failed: {response.status_code}')
    curr_utc_time = datetime.datetime.now()
    time_list = [datetime.datetime.fromisoformat
        (time_str.replace('z','+00:00'))
        for time_str in results['hourly']['time']
    ]
    temp_list = results['hourly']['temperature_2m']
    closest_time_idx = min(range(len(time_list)), 
        key=lambda i: abs(time_list[i] - curr_utc_time))
    curr_temp = temp_list[closest_time_idx]
    return f"Current temp: {curr_temp}"
#### Tool 2
@tool
def search_wiki(query: str)-> str:
    """Run wiki and get page sunmmaries"""
    page_titles = wikipedia.search(query)
    sunmmaries = []
    for pg_title in page_titles[:3]:
        try:
            wiki_pg = wikipedia.page(
                title=pg_title, auto_suggest=False
            )
            sunmmaries.append(
                f'Page:{pg_title}\nSummary:{wiki_pg.summary}'
            )
        except:
            pass
    if not sunmmaries:
        return "No good wikis"
    return "\n\n".join(sunmmaries)
#### convirtiendo fcns en OpenAI Json str
functions = [
    format_tool_to_openai_function(f) for f in [
        search_wiki, get_curr_temp
    ]
]
#### Creando Cadena
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
])
model = ChatOpenAI(temperature=0).bind(functions=functions)
model.invoke("what is the weather in sf right now")
model.invoke("what is langchain")
chain_one = prompt | model | OpenAIFunctionsAgentOutputParser()




##### "AgentActionMessageLog" vs "AgentFinish"
#### "AgentActionMessageLog"
result = chain.invoke({"input": "what is the weather in sf right now"})
type(result)    # REturn: "langchain.schema.agent.AgentActionMessageLog"
result.tool     # REturn: 'get_curr_temp'
result.tool_input   #return: {'latitude': 37.7749, 'longitude': -122.4194}
get_curr_temp(result.tool_input) 
    #Output: 'The current temperature is 25.8°C'
#### "AgentFinish"
result = chain.invoke({"input": "hi!"})
type(result)    #Return: "langchain.schema.agent.AgentFinish"
result.return_values    
    #Output: {'output': 'Hello! How can I assist you today?'}




##### Cadena con funcion automatica
#### Crear router
def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "search_wikipedia": search_wiki, 
            "get_current_temperature": get_curr_temp,
        }
        return tools[result.tool].run(result.tool_input)
#### Creando cadena
chain_two = prompt | model | OpenAIFunctionsAgentOutputParser() | route
result = chain_two.invoke({"input": "What is the weather in san francisco right now?"})
result = chain_two.invoke({"input": "What is langchain?"})
chain_two.invoke({"input": "hi!"})



































#########################################################################
###############################################################
    # Agente conversacional #
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import tool
import requests
import datetime
from pydantic import BaseModel, Field
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.agent import AgentFinish
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory

# Cadea prev
tools = [get_curr_temp, search_wiki]
fcns = [format_tool_to_openai_function(f) for f in tools]
model = ChatOpenAI(temperature=0).bind(functions=fcns)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
])
chain_prev = prompt | model | OpenAIFunctionsAgentOutputParser()






##### Cadena con short-term mem
#### Asignar "MessagesPlaceholder" al prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
#### Crear cadena
chain_one = prompt | model | OpenAIFunctionsAgentOutputParser()
#### Primera Invocacion
tools = {
    "search_wikipedia": search_wiki, 
    "get_current_temperature": get_curr_temp,
}
result1 = chain_one.invoke({
    "input": "what is the weather is sf?",
    "agent_scratchpad": []  #Empty
})
type(result1)   #Output: "langchain.schema.agent.AgentActionMessageLog"
observation = tools[result1.tool].run(result1.tool_input)
#### Segunda invocacion
result2 = chain_one.invoke({
    "input": "what is the weather is sf?", 
    "agent_scratchpad": format_to_openai_functions([(result1, observation)])
        # Turning results into scratchpad
})



##### Agent cadena
#### Cadena con short-term mem - loop
def run_agent(user_input):
    intermediate_steps = []
    while True:
        result = chain_one.invoke({
            "input": user_input, 
            "agent_scratchpad": format_to_openai_functions(intermediate_steps)
        })
        if isinstance(result, AgentFinish): return result
        tool = {
            "search_wikipedia": search_wiki, 
            "get_current_temperature": get_curr_temp,
        }[result.tool]
        observation = tool.run(result.tool_input)
        intermediate_steps.append((result, observation))
#### Putting formatting inside chain
### Component to turn results into scratchpad
preprocess = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
)
### Updating chain
agent_chain = preprocess | chain_one
#### Create general memory chain with loop
def run_agent(user_input):
    intermediate_steps = []
    while True:
        result = agent_chain.invoke({
            "input": user_input, 
            "intermediate_steps": intermediate_steps
        })
        if isinstance(result, AgentFinish): return result
        tool = {
            "search_wikipedia": search_wiki, 
            "get_current_temperature": get_curr_temp,
        }[result.tool]
        observation = tool.run(result.tool_input)
        intermediate_steps.append((result, observation))





##### Usando mem. largo-plazo
    # Requiere un 2ndo "MessagePlaceholder"
#### Modifying prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
#### Modify cadena
agent_chain_two = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | prompt | model | OpenAIFunctionsAgentOutputParser()
#### Enabling long-term
memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
#### Usando "AgentExecutor"
agent_executor = AgentExecutor(
    agent=agent_chain_two, tools=tools, verbose=True, memory=memory
)
agent_executor.invoke({"input": "my name is bob"})
agent_executor.invoke({"input": "whats my name"})




##### Planilla para custom tool
from pydantic import BaseModel, Field
class General(BaseModel):
    param1: str = Field(description="first parameter")
    param2: float = Field(description="second param")
    # Feel free to add more
@tool(args_schema=General)
def create_your_own(param1: str, param2: float) -> str:
    """This function can do whatever you would like once you fill it in """
    print(param1)
    print(param2)
    ### Implement some logic using param1 and param2
    return f"Answers: {param1} and {param2}"





##### Todo junto
import panel as pn  # GUI
pn.extension()
import param

tools = [get_curr_temp, search_wiki, create_your_own]

class cbfs(param.Parameterized):
    
    def __init__(self, tools, **params):
        # AGent chain itself
            #Note the use of "tools" in "init"
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.functions = [format_tool_to_openai_function(f) for f in tools]
        self.model = ChatOpenAI(temperature=0).bind(functions=self.functions)
        self.memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are helpful but sassy assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.chain = RunnablePassthrough.assign(
            agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | self.prompt | self.model | OpenAIFunctionsAgentOutputParser()
        self.qa = AgentExecutor(agent=self.chain, tools=tools, verbose=False, memory=self.memory)
    
    def convchain(self, query):
        # GUI setup
        if not query:
            return
        inp.value = ''
        result = self.qa.invoke({"input": query})
        self.answer = result['output'] 
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=450)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=450, styles={'background-color': '#F6F6F6'}))
        ])
        return pn.WidgetBox(*self.panels, scroll=True)


    def clr_history(self,count=0):
        self.chat_history = []
        return 

### Initialize chatbot and GUI
cb = cbfs(tools)

inp = pn.widgets.TextInput( placeholder='Enter text here…')

conversation = pn.bind(cb.convchain, inp) 

tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation,  loading_indicator=True, height=400),
    pn.layout.Divider(),
)

dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# QnA_Bot')),
    pn.Tabs(('Conversation', tab1))
)
dashboard










































#########################################################################
#########################################################################
    ### C7 ###
#########################################################################
#########################################################################
    # ReAct agente #
import openai
import re
import httpx
import os
from dotenv import load_dotenv
from openai import OpenAI



##### Clase Agente
#### Prompt
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()
#### Funciones
def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}
#### Modelo General
client = OpenAI()
#### Clase 
class Agent:
    def __init__(self, system=""):
        # Initializing Agent
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        # Logic for what agent will do
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
                        model="gpt-4o", 
                        temperature=0,
                        messages=self.messages)
        return completion.choices[0].message.content
#### Inicializacion
abot = Agent(prompt)
#### Invocacion
result = abot("How much does a toy poodle weigh?")
print(result)
    #Output:
    # Thought: I should look up the average weight of a Toy Poodle using the average_dog_weight action.
             # Action: average_dog_weight: Toy Poodle
             # PAUSE
result = average_dog_weight("Toy Poodle")
print(result)
    #Output: 'a toy poodles average weight is 7 lbs'
next_prompt = "Observation: {}".format(result)
abot(next_prompt)
    #Output: 'Answer: A Toy Poodle weighs an average of 7 lbs.'
print(abot.messages)
    #Output:
    #[{'role': 'system',
        #'content': 'You run in a loop of Thought, Action, PAUSE, Observation.\nAt the end of the loop you output an Answer\nUse Thought to describe your thoughts about the question you have been asked.\nUse Action to run one of the actions available to you - then return PAUSE.\nObservation will be the result of running those actions.\n\nYour available actions are:\n\ncalculate:\ne.g. calculate: 4 * 7 / 3\nRuns a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary\n\naverage_dog_weight:\ne.g. average_dog_weight: Collie\nreturns average weight of a dog when given the breed\n\nExample session:\n\nQuestion: How much does a Bulldog weigh?\nThought: I should look the dogs weight using average_dog_weight\nAction: average_dog_weight: Bulldog\nPAUSE\n\nYou will be called again with this:\n\nObservation: A Bulldog weights 51 lbs\n\nYou then output:\n\nAnswer: A bulldog weights 51 lbs'},
        #{'role': 'user', 'content': 'How much does a toy poodle weigh?'},
        #{'role': 'assistant',
        #'content': 'Thought: I should look up the average weight of a Toy Poodle using the average_dog_weight action.\nAction: average_dog_weight: Toy Poodle\nPAUSE'},
        #{'role': 'user',
        #'content': 'Observation: a toy poodles average weight is 7 lbs'},
        #{'role': 'assistant',
        #'content': 'Answer: A Toy Poodle weighs an average of 7 lbs.'}]








##### ReAct Loop
#### Regex (para parar)
action_re = re.compile('^Action: (\w+): (.*)$')   
#### Funciones hasta ahora
known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}
#### Loop fcn
def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a) 
            for a in result.split('\n') 
            if action_re.match(a)
        ]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return
#### Invocacion
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
query(question)
    #Output:
    # Thought: I need to find the average weight of both a Border Collie and a Scottish Terrier, then add them together to get the combined weight.
            # Action: average_dog_weight: Border Collie
            # PAUSE
            #  -- running average_dog_weight Border Collie
            # Observation: a Border Collies average weight is 37 lbs
            # Action: average_dog_weight: Scottish Terrier
            # PAUSE
            #  -- running average_dog_weight Scottish Terrier
            # Observation: Scottish Terriers average 20 lbs
            # Thought: Now that I have the average weights of both dogs, I can calculate their combined weight by adding the two values together.
            # Action: calculate: 37 + 20
            # PAUSE
            #  -- running calculate 37 + 20
            # Observation: 57
            # Answer: The combined weight of a Border Collie and a Scottish Terrier is 57 lbs.





























#########################################################################
###############################################################
    # Langgraph Componentes #
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Union
import operator
from langchain_core.messages import AnyMessage, SystemMessage, \
    HumanMessage, ToolMessage, AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from IPython.display import Image



##### Explorando Agent State
#### Simple
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    # "AnyMessage" = umbrella for all "Message" types
    # "operator.add" = "Messages" will be APPENDED to previous "Message"
    # "Annotated" = allows for specifying metadata details 
        # Syntax: "Annotated[Type, meta1, meta2, ...]"
#### Complejo
class ComplexAgentState(TypedDict):
    inputs: str
    chat_history: list[AnyMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
        #"Union" = can be ONLY one of the listed types
    intermediate_steps: Annotated[
        list[tuple[AgentAction, str]], 
        operator.add
    ]






##### Creando LG Clase Agente
#### Clase
class Agent:

    def __init__(self, model, tools, system=""):
        # Creates the graph itself
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
#### Inicializacion
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
tool = TavilySearchResults(max_results=4)
model = ChatOpenAI(model="gpt-3.5-turbo") 
abot = Agent(model, [tool], system=prompt)
#### Visualizando grafo agente
Image(abot.graph.get_graph().draw_png())







##### Explorando LG Agente
#### Simple query
tmp_query = "What is the weather in sf?"
messages = [HumanMessage(content=tmp_query)]
result = abot.graph.invoke({"messages": messages})
print(result)
    #Output:
    # {'messages': [
    #     HumanMessage(content='What is the weather in sf?'),
    #     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_PvPN1v7bHUxOdyn4J2xJhYOX', 'function': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 153, 'total_tokens': 174, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-9c04deba-1c3b-49ce-a45c-5269f4b0c802-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_PvPN1v7bHUxOdyn4J2xJhYOX'}]),
    #     ToolMessage(content='[{\'url\': \'https://www.peoplesweather.com/weather/San+Francisco/?date=2025-10-24\', \'content\': \'Home\\n Weather Forecast\\n News & Highlights\\n MyPhoto\\n Competitions\\n Contact Us\\n\\n# Weather for San Francisco\\n\\n Weather\\n United States\\n San Francisco\\n\\n##### Friday 24 October 2025\\n\\n|  |  |\\n --- |\\n| 14°CFeels like: 14°C | W / 7km/h  Light Breeze |\\n| Partly Cloudy. Cool |\\n\\n|  |  |\\n --- |\\n| Pressure | 1018mb |\\n| Humidity | 87% |\\n| Rain | 0% |\\n| Cloud Cover | 63% |\\n| Dew Point | 12°C |\\n\\n|  |\\n\\n| This Afternoon |\\n| 19°C | SW / 10km/h  Light Breeze |\\n| Mostly Cloudy. Mild |\\n\\n|  |\'}, {\'url\': \'https://weathershogun.com/weather/usa/ca/san-francisco/480/october/2025-10-24\', \'content\': "Friday, October 24, 2025. San Francisco, CA - Weather Forecast \\n\\n☰\\n\\nSan Francisco, CA\\n\\nImage 1: WeatherShogun.com\\n\\nHomeContactBrowse StatesPrivacy PolicyTerms and Conditions\\n\\n°F)°C)\\n\\n❮\\n\\nTodayTomorrowHourly7 days30 daysOctober\\n\\n❯\\n\\nSan Francisco, California Weather: \\n\\nActive Weather Warnings\\n\\n   Beach Hazards Statement\\n\\nFriday, October 24, 2025\\n\\nDay 64°\\n\\nNight 55°\\n\\nPrecipitation 0 %\\n\\nWind 9 mph\\n\\nUV Index (0 - 11+)3\\n\\nSaturday [...] WHAT: A moderate to long period northwesterly swell will result\\n\\nin breaking waves of 15 to 20 feet, with the highest waves up to\\n\\n25 feet in favored locations, and an increased risk for sneaker\\n\\nwaves and rip currents.\\n\\nWHERE: San Francisco, Coastal North Bay Including Point Reyes\\n\\nNational Seashore, San Francisco Peninsula Coast, Northern\\n\\nMonterey Bay and Southern Monterey Bay and Big Sur Coast\\n\\nCounties.\\n\\nWHEN: From late tonight through late Sunday night. [...] Hourly\\n   Today\\n   Current Air Quality\\n   Hourly Air Quality Forecast\\n   7 days\\n   30 days\\n\\nWeather Forecast History\\n\\nLast Year\'s Weather on This Day (October 24, 2024)\\n\\n### Day\\n\\n73°\\n\\n### Night\\n\\n52°\\n\\n#### Wind\\n\\n4 mph\\n\\n#### Precipitation\\n\\n0\\n\\nWeather Alerts and Warnings for\\n\\nModerate Expected Likely\\n\\n### Beach Hazards Statement\\n\\nBeach Hazards Statement issued October 23 at 12:50PM PDT until October 27 at 3:00AM PDT by NWS San Francisco CA\\n\\nOct 24, 3:00 AM → Oct 27, 3:00 AM"}, {\'url\': \'https://wu-next-prod.wunderground.com/hourly/us/ca/san-francisco/KSFO/date/2025-10-24\', \'content\': \'date\\\\_range View Calendar Forecast\\n\\nTop Video Stories\\n\\nplay\\\\_circle\\\\_outline\\n\\nSouth Braces For Severe Storms, Flooding Friday-Sunday\\n\\n[00:01:07]\\n\\n keyboard\\\\_arrow\\\\_left\\n\\n keyboard\\\\_arrow\\\\_right\\n\\nSee more Top Video Stories\\n\\nAdditional Conditions\\n\\nPressure\\n\\n30.08 °in\\n\\nVisibility\\n\\n9 °miles\\n\\nClouds\\n\\nMostly Cloudy\\n\\nDew Point\\n\\n55 °F\\n\\nHumidity\\n\\n91 °%\\n\\nRainfall\\n\\n0 °in\\n\\nSnow Depth\\n\\n0 °in\\n\\nKSFO Station History\\n\\nAlmanac for October 24, 2025\\n\\nForecast\\n\\nAverage \\\\\\n\\nRecord\\n\\nTemperature\\n\\nHigh\\n\\n65 °F\\n\\n71 °F\'}, {\'url\': \'https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=October\', \'content\': \'United States England Australia Canada\\n\\n°F °C\\n\\nSan Francisco\\n\\nWeather in October 2025\\n\\n1. Home\\n2. North America\\n3. United States\\n4. California\\n5. San Francisco\\n6. October\\n\\nLocation was added to My Locations\\n\\nLocation was removed from My Locations\\n\\n# San Francisco weather in October 2025\\n\\nClick on a day for an hourly weather forecast\\n\\nOct 19\\n\\n0 mm\\n\\n20° / 12°Oct 20\\n\\n0 mm\\n\\n23° / 14°Oct 21\\n\\n0 mm\\n\\n21° / 12°Oct 22\\n\\n0 mm\\n\\n17° / 12°Thursday\\n\\nOct 23\\n\\n0 mm\\n\\n18° / 14°Friday\\n\\nOct 24\\n\\n0 mm\\n\\n18° / 14°Saturday [...] The wather in San Francisco in October can vary between cold and nice weather days. Expect a few rainy days but usually not more than 3.\\n\\nOur weather forecast can give you a great sense of what weather to expect in San Francisco in October 2025.\\n\\nIf you’re planning to visit San Francisco in the near future, we highly recommend that you review the 14 day weather forecast for San Francisco before you arrive.\\n\\nTemperatures\\n\\n23° / 13°\\n\\nRainy Days\\n\\n1\\n\\nSnowy Days\\n\\n0\\n\\nDry Days\\n\\n30\\n\\nRainfall\\n\\n26\\n\\nmm [...] Oct 25\\n\\n0.8 mm\\n\\n16° / 14°Sunday\\n\\nOct 26\\n\\n1 mm\\n\\n16° / 12°Monday\\n\\nOct 27\\n\\n0.8 mm\\n\\n18° / 15°Tuesday\\n\\nOct 28\\n\\n0 mm\\n\\n20° / 14°Wednesday\\n\\nOct 29\\n\\n0 mm\\n\\n23° / 15°Thursday\\n\\nOct 30\\n\\n0 mm\\n\\n23° / 16°Friday\\n\\nOct 31\\n\\n0 mm\\n\\n22° / 17°Saturday\\n\\nNov 1\\n\\n0 mm\\n\\n21° / 17° NextMonth >>\\n\\n## The average weather in San Francisco in October\\n\\nThe temperatures in San Francisco in October are comfortable with low of 13°C and and high up to 23°C.\'}]', name='tavily_search_results_json', tool_call_id='call_PvPN1v7bHUxOdyn4J2xJhYOX'),
    #     AIMessage(content='The weather in San Francisco today is partly cloudy with a temperature of 14°C. The humidity is at 87%, and there is a light breeze from the west at 7km/h. The cloud cover is at 63% with no expected rain.', response_metadata={'token_usage': {'completion_tokens': 53, 'prompt_tokens': 1645, 'total_tokens': 1698, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-8c8e5d00-5251-457d-8d6e-d2419e29bb27-0')
    # ]}
for msg in result['messages']: print(type(msg))
    # Output:
    # <class 'langchain_core.messages.human.HumanMessage'>
    # <class 'langchain_core.messages.ai.AIMessage'>
    # <class 'langchain_core.messages.tool.ToolMessage'>
    # <class 'langchain_core.messages.ai.AIMessage'>


















#########################################################################
###############################################################
    # Agentic Search Tool #
from dotenv import load_dotenv
import os
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import re
import json
from pygments import highlight, lexers, formatters
# load environment variables from .env file
_ = load_dotenv()



##### Busqeudad regular, scrapping, y formateo de scrape
    # DDG solo da urls, se necesita scrapping
#### Setup Duckduckgo y search funcion
ddg = DDGS()
def search(query, max_results=6):
    try:
        results = ddg.text(query, max_results=max_results)
        return [i["href"] for i in results]
    except Exception as e:
        print(f"returning previous results due to exception reaching ddg.")
        results = [ # cover case where DDG rate limits due to high deeplearning.ai volume
            "https://weather.com/weather/today/l/USCA0987:1:US",
            "https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8",
        ]
        return results  
#### Scrape fcn
def scrape_weather_info(url):
    """Scrape content from the given URL"""
    if not url:
        return "Weather information could not be found."
    
    # fetch data
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "Failed to retrieve the webpage."

    # parse result
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup
#### Invocacion de funcion
city = "San Francisco"
tmp_query2 = f"""
    what is the current weather in {city}?
    Should I travel there today?
    "weather.com"
"""
for i in search(tmp_query2):
    print(i)
    #output:
    # returning previous results due to exception reaching ddg.
        # https://weather.com/weather/today/l/USCA0987:1:US
        # https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8
url = search(tmp_query2)[0]
soup = scrape_weather_info(url)
print(str(soup.body)[:50000])
    # OUtput: HTML
#### Convirtiendo HTML a human-readible
### Identificando tags
weather_data = []
for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
    text = tag.get_text(" ", strip=True)
    weather_data.append(text)
### Combinando todos los tags
weather_data = "\n".join(weather_data)
### remove all spaces from the combined text
weather_data = re.sub(r'\s+', ' ', weather_data)
print(weather_data)
    #Output: HTML Readible





##### Agentic busquedad
#### Setup
client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
#### run search
result = client.search(tmp_query2, max_results=1)
#### print first result
data = result["results"][0]["content"]
print(type(result))
    #Output: dictionary, lo que AIs necesitan




##### Mejor visualizacion de dictionarios
#### parse JSON
parsed_json = json.loads(data.replace("'", '"'))
#### pretty print JSON with syntax highlighting
formatted_json = json.dumps(parsed_json, indent=4)
colorful_json = highlight(formatted_json,
                          lexers.JsonLexer(),
                          formatters.TerminalFormatter())

print(colorful_json)




















#########################################################################
#########################################################################
    # Persistence y Streaming #
#### SEt up environemt
from dotenv import load_dotenv
_ = load_dotenv()
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver






##### Agente con in-mem persistence
#### Inicializando in-mem persistence
memory = SqliteSaver.from_conn_string(":memory:")
#### Setup general
tool = TavilySearchResults(max_results=2)
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
#### Agente class
class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action, 
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
            # In-mem persistence
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
#### Inicializando Agente
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(model="gpt-4o")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)









##### Haciendo streaming con {"configurable"}
#### Inicializando stream con thread
messages = [HumanMessage(content="What is the weather in sf?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v['messages'])
    #Output:
    # [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_bmfLa92f6oAIKN9KvXtqbKDz', 'function': {'arguments': '{"query":"current weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 151, 'total_tokens': 173, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_831e067d82', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-ec74a787-2389-4bb4-8e61-35d32024c8c8-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_bmfLa92f6oAIKN9KvXtqbKDz'}])]
        # Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_bmfLa92f6oAIKN9KvXtqbKDz'}
        # Back to the model!
        # [ToolMessage(content='[{\'url\': \'https://www.peoplesweather.com/weather/San+Francisco/?date=2025-10-24\', \'content\': \'# Weather for San Francisco\\n\\n##### Friday 24 October 2025\\n\\n|  |  |  |\\n --- \\n|  | 14°CFeels like: 14°C | W / 7km/h  Light Breeze |\\n| Overcast. Cool | | |\\n\\n|  |  |\\n --- |\\n| Pressure | 1017mb |\\n| Humidity | 90% |\\n| Rain | 0% |\\n| Cloud Cover | 98% |\\n| Dew Point | 13°C |\\n\\n|  |  |  |\\n --- \\n| Post Midnight | | |\\n|  | 14°C | WSW / 4km/h  Light Air |\\n| Overcast. Cool | | |\\n\\n|  |  |  |\\n --- \\n| Morning | | |\\n|  | 15°C | SSW / 9km/h  Light Breeze |\\n| Chance of Rain. Cool | | |\'}, {\'url\': \'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629\', \'content\': "Sat\\n\\n10/25\\n\\nA little morning rain\\n\\nMostly cloudy\\n\\nSun\\n\\n10/26\\n\\nAn afternoon shower in spots\\n\\nPartly cloudy\\n\\nMon\\n\\n10/27\\n\\nMostly sunny\\n\\nClear\\n\\nTue\\n\\n10/28\\n\\nBeautiful with sunshine\\n\\nClear to partly cloudy\\n\\nWed\\n\\n10/29\\n\\nMostly sunny and pleasant\\n\\nClear\\n\\nThu\\n\\n10/30\\n\\nMostly sunny and pleasant\\n\\nMainly clear\\n\\nFri\\n\\n10/31\\n\\nMostly sunny\\n\\nClear\\n\\nSat\\n\\n11/1\\n\\nSunshine\\n\\nIncreasing clouds\\n\\nSun\\n\\n11/2\\n\\nPartly sunny\\n\\nPartly cloudy\\n\\n## Sun & Moon\\n\\n## Air Quality [...] Tonight: Mostly cloudy with a shower toward dawn\\nLo: 56°\\n\\n## Current Weather\\n\\n12:12 PM\\n\\n## Looking Ahead\\n\\nExpect showery weather before dawn tomorrow through tomorrow morning\\n\\n## San Francisco Weather Radar\\n\\nSan Francisco Weather Radar\\n\\n## Hourly Weather\\n\\nrain drop\\n\\nrain drop\\n\\nrain drop\\n\\nrain drop\\n\\nrain drop\\n\\nrain drop\\n\\nrain drop\\n\\nrain drop\\n\\nrain drop\\n\\nrain drop\\n\\nrain drop\\n\\nrain drop\\n\\n## 10-Day Weather Forecast\\n\\nToday\\n\\n10/24\\n\\nClouds yielding to some sun\\n\\nNight: Rather cloudy, a shower late [...] # San Francisco, CA\\n\\nSan Francisco\\n\\nCalifornia\\n\\n## Around the Globe\\n\\nAround the Globe\\n\\n### Hurricane Tracker\\n\\n### Severe Weather\\n\\n### Radar & Maps\\n\\n### News & Features\\n\\n### Astronomy\\n\\n### Business\\n\\n### Climate\\n\\n### Health\\n\\n### Recreation\\n\\n### Sports\\n\\n### Travel\\n\\n### Warnings\\n\\n### Data Suite\\n\\n### Forensics\\n\\n### Advertising\\n\\n### Superior Accuracy™\\n\\n### Video\\n\\n### Winter Center\\n\\n## Today\\n\\n## Today\'s Weather\\n\\nFri, Oct 24\\n\\nClouds yielding to some sun\\nHi: 65°"}]', name='tavily_search_results_json', tool_call_id='call_bmfLa92f6oAIKN9KvXtqbKDz')]
        # [AIMessage(content='The current weather in San Francisco is overcast and cool with a temperature of 14°C (feels like 14°C). There is a light breeze coming from the west at 7 km/h. The humidity is high at 90%, and the cloud cover is at 98%, but there is no rain expected. The pressure is measured at 1017 mb, and the dew point is 13°C.', response_metadata={'token_usage': {'completion_tokens': 85, 'prompt_tokens': 910, 'total_tokens': 995, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_65564d8ba5', 'finish_reason': 'stop', 'logprobs': None}, id='run-f7472989-3c4f-45f1-90ef-cf8bcfc783c6-0')]
### Checking out persistence by continuing conversation 
messages = [HumanMessage(content="What about in la?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
    #Output:
    # {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Hf2tY07frZGgFcifdWR270za', 'function': {'arguments': '{"query":"current weather in Los Angeles"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 1007, 'total_tokens': 1029, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_65564d8ba5', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-b3494a6e-12ee-4698-bd0e-e768be9ce37d-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Los Angeles'}, 'id': 'call_Hf2tY07frZGgFcifdWR270za'}])]}
            # Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Los Angeles'}, 'id': 'call_Hf2tY07frZGgFcifdWR270za'}
            # Back to the model!
            # {'messages': [ToolMessage(content='[{\'url\': \'https://www.weather25.com/north-america/usa/california/los-angeles?page=month&month=October\', \'content\': \'weather25.com\\nSearch\\nweather in United States\\nRemove from your favorite locations\\nAdd to my locations\\nShare\\nweather in United States\\n\\n# Los Angeles weather in October 2025\\n\\nPartly cloudy\\nPartly cloudy\\nPartly cloudy\\nPartly cloudy\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\nOvercast\\nPartly cloudy\\nClear\\nClear\\n\\n## The average weather in Los Angeles in October\\n\\nThe temperatures in Los Angeles in October are comfortable with low of 18°C and and high up to 27°C. [...] | 19 Sunny 28° /19° | 20 Sunny 28° /19° | 21 Sunny 27° /19° | 22 Sunny 24° /18° | 23 Partly cloudy 24° /17° | 24 Partly cloudy 29° /13° | 25 Partly cloudy 22° /15° |\\n| 26 Partly cloudy 23° /15° | 27 Partly cloudy 26° /15° | 28 Sunny 29° /19° | 29 Sunny 28° /20° | 30 Sunny 26° /20° | 31 Sunny 27° /19° |  | [...] You can expect a few rainy days in Los Angeles during October, but usually the weather is comfortable in October.\\n\\nOur weather forecast can give you a great sense of what weather to expect in Los Angeles in October 2025.\\n\\nIf you’re planning to visit Los Angeles in the near future, we highly recommend that you review the 14 day weather forecast for Los Angeles before you arrive.\\n\\nTemperatures\\nRainy Days\\nSnowy Days\\nDry Days\\nRainfall\\n11.7\'}, {\'url\': \'https://www.accuweather.com/en/us/los-angeles/90012/october-weather/347625\', \'content\': "# Los Angeles, CA\\n\\nLos Angeles\\n\\nCalifornia\\n\\n## Around the Globe\\n\\nAround the Globe\\n\\n### Hurricane Tracker\\n\\n### Severe Weather\\n\\n### Radar & Maps\\n\\n### News & Features\\n\\n### Astronomy\\n\\n### Business\\n\\n### Climate\\n\\n### Health\\n\\n### Recreation\\n\\n### Sports\\n\\n### Travel\\n\\n### Warnings\\n\\n### Data Suite\\n\\n### Forensics\\n\\n### Advertising\\n\\n### Superior Accuracy™\\n\\n### Video\\n\\n### Winter Center\\n\\n## Monthly\\n\\n## October\\n\\n## 2025\\n\\n## Daily\\n\\n## Temperature Graph\\n\\n## Further Ahead\\n\\nFurther Ahead\\n\\n### November 2025 [...] ### December 2025\\n\\n### January 2026\\n\\n## Around the Globe\\n\\nAround the Globe\\n\\n### Hurricane Tracker\\n\\n### Severe Weather\\n\\n### Radar & Maps\\n\\n### News\\n\\n### Video\\n\\n### Winter Center\\n\\nTop Stories\\n\\nHurricane\\n\\nMelissa may reach Category 5, poses great danger to Jamaica, Cuba, Hai...\\n\\n3 hours ago\\n\\nWeather Forecasts\\n\\nWeather troubles brewing for some trick-or-treaters through Halloween\\n\\n2 hours ago\\n\\nHurricane\\n\\nMelissa, future nor\'easter to team up along US East Coast next week\\n\\n1 hour ago\\n\\nHurricane [...] Coast Guard rescues family stranded on island off Cape Cod\\n\\n1 day ago\\n\\nWeather News\\n\\nPolar bears take over abandoned island in Russia\\n\\n4 days ago\\n\\n## Weather Near Los Angeles:\\n\\n...\\n\\n...\\n\\n..."}]', name='tavily_search_results_json', tool_call_id='call_Hf2tY07frZGgFcifdWR270za')]}
            # {'messages': [AIMessage(content="The current weather in Los Angeles is partly cloudy. The temperatures are comfortable, with a low around 18°C and a high up to 27°C. There is no specific mention of rain, indicating it's likely dry at the moment.", response_metadata={'token_usage': {'completion_tokens': 48, 'prompt_tokens': 1814, 'total_tokens': 1862, 'prompt_tokens_details': {'cached_tokens': 1024, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_65564d8ba5', 'finish_reason': 'stop', 'logprobs': None}, id='run-453482ec-592f-4097-93f6-45292eb8c029-0')]}







##### Explorando importancia de threads en persistence
messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "1"}}
    #NOte: THREAD 1
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
    #Output:
    #{'messages': [AIMessage(content='Los Angeles is warmer than San Francisco at the moment. Los Angeles has temperatures ranging from 18°C to 27°C, while San Francisco is currently experiencing a temperature of 14°C.', response_metadata={'token_usage': {'completion_tokens': 39, 'prompt_tokens': 1874, 'total_tokens': 1913, 'prompt_tokens_details': {'cached_tokens': 1792, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_65564d8ba5', 'finish_reason': 'stop', 'logprobs': None}, id='run-703aaf32-db67-4a2b-b2b6-089bf54025ca-0')]}

messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
    #Output:
    #{'messages': [AIMessage(content="Could you please clarify what you're comparing to determine which is warmer? Are you comparing two specific locations, types of clothing, materials, or something else? Let me know so I can provide the appropriate information.", response_metadata={'token_usage': {'completion_tokens': 43, 'prompt_tokens': 149, 'total_tokens': 192, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_f9f4fb6dbf', 'finish_reason': 'stop', 'logprobs': None}, id='run-b6342196-b872-4f67-8970-b6bf527e419c-0')]}







##### Haciendo Asynch
#### Setup
memory = AsyncSqliteSaver.from_conn_string(":memory:")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)
#### Actually streaming tokens
messages = [HumanMessage(content="What is the weather in SF?")]
thread = {"configurable": {"thread_id": "4"}}
async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            # Empty content in the context of OpenAI means
            # that the model is asking for a tool to be invoked.
            # So we only print non-empty content
            print(content, end="|")
    # Output:
    # Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_GiHnpbt7P6kuX4g2n6imqDXI'}
            # Back to the model!
            # The| current| weather| in| San| Francisco| is| over|
            # cast| and| cool|,| with| a| temperature| of| |14|°C| 
            # (|fe|els| like| |14|°C|).| The| wind| is| coming| 
            # from| the| west| at| |7| km|/h|,| and| the| humidity| 
            # is| at| |90|%.| The| cloud| cover| is| |98|%,| and| 
            # there's| no| rain| expected|.| The| pressure| is| 
            # |101|7| mb| with| a| dew| point| of| |13|°C|.|





























#########################################################################
#########################################################################
    # Humano En Loop #
from dotenv import load_dotenv
_ = load_dotenv()
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from uuid import uuid4
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage






##### Custom state aggregator
"""
In previous examples we've annotated the `messages` state key
with the default `operator.add` or `+` reducer, which always
appends new messages to the end of the existing messages array.

Now, to support replacing existing messages, we annotate the
`messages` key with a customer reducer function, which replaces
messages with the same `id`, and appends them otherwise.
"""
#### Custom aggregator function
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break   ##### SE PUEDE CAMBIAR
        else:
            # append any new messages to the end
            merged.append(message)
    return merged






##### Agente con interrupcion
#### Creando Agente
### Setup
memory = SqliteSaver.from_conn_string(":memory:")
    # Persistence mem. using in-mem
search_tool = TavilySearchResults(max_results=2)
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]
class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"] ##### IMPORTANTE PARA HITL
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        print(state)
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
### Init
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(model="gpt-3.5-turbo")
abot = Agent(model, [search_tool], system=prompt, checkpointer=memory)












##### Analizando Agent State
#### Empezando conversacion
    # Ojo: esto va a parar en " interrupt_before=["action"]"
messages = [HumanMessage(content="Whats the weather in SF?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
#### Analizando Agent state (cuando interrupted)
state = abot.graph.get_state(thread)
for attr in dir(state):
    if attr.startswith("_"):
        continue
    try:
        print(f"Attr: {attr}")
        print(state.__getattribute__(attr))
        print('\n\n')
    except:
        print(f"couldn't print {attr}")
        print('\n\n')
        continue
# Attr: config
# {'configurable': {'thread_id': '1', 'thread_ts': '1f0b1c75-7f54-6abb-8001-9365c5b75911'}}



# Attr: count
# <built-in method count of StateSnapshot object at 0x7f7c34c2b640>



# Attr: created_at
# 2025-10-25T17:23:38.078672+00:00



# Attr: index
# <built-in method index of StateSnapshot object at 0x7f7c34c2b640>



# Attr: metadata
# {'source': 'loop', 'step': 1, 'writes': {'llm': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'id': 'call_i7rGhnzgZf5hW3bsDcqdTrH0', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 152, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 174}}, id='run-5e18196a-dd77-4477-a570-d9eb395ae520-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_i7rGhnzgZf5hW3bsDcqdTrH0'}])]}}}



# Attr: next
# ('action',)



# Attr: parent_config
# {'configurable': {'thread_id': '1', 'thread_ts': '1f0b1c75-7ef0-6eb2-8000-110d8286c564'}}



# Attr: values
# {'messages': [HumanMessage(content='Whats the weather in SF?', id='48b16d43-ecee-43e3-992d-3329cb7925f0'), AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'id': 'call_i7rGhnzgZf5hW3bsDcqdTrH0', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 152, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 174}}, id='run-5e18196a-dd77-4477-a570-d9eb395ae520-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_i7rGhnzgZf5hW3bsDcqdTrH0'}])]}


print(state.values.keys())
    # output: dict_keys(['messages'])
    # I.e., los keys son los items en AgentState
#### Continuando despues the interrupt
for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)





















##### Implementacion HITL
messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    # Creates the INITIAL conversation stream
    for v in event.values():
        print(v)    
        print('\n\n\n')
while abot.graph.get_state(thread).next:
    # Only continues while ".get_state.next" is NOT empty
        # Non empty when there's a node to go after an interrupt
        # Empty once agent reaches "__end__" state
    print("\n", abot.graph.get_state(thread),"\n")
    _input = input("proceed?")
    if _input != "y":
        print("aborting")
        break
    for event in abot.graph.stream(None, thread):
        for v in event.values():
            print(v)
            print('\n\n\n')




















##### Modifiando agent state
#### Investigando current state 
current_values = abot.graph.get_state(thread)
print(current_values.values['messages'][-1])
    #output:
    # AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"query":"weather in Los Angeles"}', 'name': 'tavily_search_results_json'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 152, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 174}}, id='run-b68d1c1b-a0c6-4cb4-a8ce-1cc6aa8387a9-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'weather in Los Angeles'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc'}])
    # The following is important to note from the output:
        # "AIMessage.additionalkwargs['tool_calls'][0]['function']
                # ['name']" = 'tavily_search_results_json'
            #I.e., this is the tool LLM decided to use
            # It also includes LLM's decided inputs for that tool
print(current_values.values['messages'][-1].tool_calls)
    #output:
    # [{'name': 'tavily_search_results_json',
            #   'args': {'query': 'weather in Los Angeles'},
            #   'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc'}]
    # Helpful in analyzing the specific tool call
#### Proponiendo modificaciones
_id = current_values.values['messages'][-1].tool_calls[0]['id']

## Chaning the query but keeping everyrthing else the same
current_values.values['messages'][-1].tool_calls = [
    {'name': 'tavily_search_results_json',
  'args': {'query': 'current weather in Louisiana'},
  'id': _id}
]

#### Updating state
abot.graph.update_state(thread, current_values.values)

#### Running agent from the updated state
for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)















##### Time travel modificacion
#### Getting entire snapshot history
states = []
for state in abot.graph.get_state_history(thread):
    # Starts from most recent, going to most oldest
    print(state)
    print('\n\n--\n\n')
    states.append(state)
#Output:
# StateSnapshot(values={'messages': [HumanMessage(content='Whats the weather in LA?', id='9e477a83-adba-4ee5-912d-d378993ace7e'), AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"query":"weather in Los Angeles"}', 'name': 'tavily_search_results_json'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 152, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 174}}, id='run-b68d1c1b-a0c6-4cb4-a8ce-1cc6aa8387a9-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Louisiana'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc'}]), ToolMessage(content='[{\'url\': \'https://www.weather25.com/north-america/usa/louisiana?page=month&month=October\', \'content\': \'weather25.com\\nSearch\\nweather in United States\\nRemove from your favorite locations\\nAdd to my locations\\nShare\\nweather in United States\\n\\n# Louisiana weather in October 2025\\n\\nPatchy rain possible\\nModerate or heavy rain with thunder\\nCloudy\\nPartly cloudy\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\n\\n## The average weather in Louisiana in October\\n\\nThe weather in Louisiana in October is hot. The average temperatures are between 20°C and 27°C. [...] | 26 Moderate or heavy rain with thunder 27° /21° | 27 Cloudy 27° /21° | 28 Partly cloudy 24° /18° | 29 Sunny 21° /15° | 30 Sunny 18° /11° | 31 Sunny 19° /11° |  | [...] You can expect about 3 to 8 days of rain in Louisiana during the month of October. It’s a good idea to bring along your umbrella so that you don’t get caught in poor weather.\\n\\nOur weather forecast can give you a great sense of what weather to expect in Louisiana in October 2025.\\n\\nIf you’re planning to visit Louisiana in the near future, we highly recommend that you review the 14 day weather forecast for Louisiana before you arrive.\\n\\nTemperatures\\nRainy Days\\nSnowy Days\\nDry Days\\nRainfall\\n11.3\'}, {\'url\': \'https://www.accuweather.com/en/us/new-orleans/70112/october-weather/348585\', \'content\': "# New Orleans, LA\\n\\nNew Orleans\\n\\nLouisiana\\n\\n## Around the Globe\\n\\nAround the Globe\\n\\n### Hurricane Tracker\\n\\n### Severe Weather\\n\\n### Radar & Maps\\n\\n### News & Features\\n\\n### Astronomy\\n\\n### Business\\n\\n### Climate\\n\\n### Health\\n\\n### Recreation\\n\\n### Sports\\n\\n### Travel\\n\\n### Warnings\\n\\n### Data Suite\\n\\n### Forensics\\n\\n### Advertising\\n\\n### Superior Accuracy™\\n\\n### Video\\n\\n### Winter Center\\n\\n## Monthly\\n\\n## October\\n\\n## 2025\\n\\n## Daily\\n\\n## Temperature Graph\\n\\n## Further Ahead\\n\\nFurther Ahead\\n\\n### November 2025 [...] ### December 2025\\n\\n### January 2026\\n\\n## Around the Globe\\n\\nAround the Globe\\n\\n### Hurricane Tracker\\n\\n### Severe Weather\\n\\n### Radar & Maps\\n\\n### News\\n\\n### Video\\n\\n### Winter Center\\n\\nTop Stories\\n\\nHurricane\\n\\nMelissa may reach Category 5, poses great danger to Jamaica, Cuba, Hai...\\n\\n52 minutes ago\\n\\nWeather Forecasts\\n\\nWeather troubles brewing for some trick-or-treaters through Halloween\\n\\n4 hours ago\\n\\nHurricane\\n\\nMelissa, future nor\'easter to team up along US East Coast next week\\n\\n1 hour ago\\n\\nHurricane [...] Coast Guard rescues family stranded on island off Cape Cod\\n\\n2 days ago\\n\\nWeather News\\n\\nPolar bears take over abandoned island in Russia\\n\\n5 days ago\\n\\n## Weather Near New Orleans:\\n\\n...\\n\\n...\\n\\n..."}]', name='tavily_search_results_json', id='c87d40ee-76cb-42cc-8af2-34077e6201e1', tool_call_id='call_6ED1ZQ8nrjYIOY14yqInLPZc'), AIMessage(content="I found information about the weather in Louisiana. The average temperatures in Louisiana in October range between 20°C and 27°C. There may be patchy rain possible, moderate or heavy rain with thunder, cloudy, partly cloudy, and clear days throughout the month. It's advisable to bring an umbrella as there could be 3 to 8 days of rain in October.\n\nIf you were referring to Los Angeles (LA) in California, please let me know so I can provide you with the specific weather information for that location.", response_metadata={'finish_reason': 'stop', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 107, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 903, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 1010}}, id='run-b7e3f01a-ec8d-4a0d-a081-947d9c789216-0')]}, next=(), config={'configurable': {'thread_id': '3', 'thread_ts': '1f0b1c76-05b6-615a-8004-3a780005737e'}}, metadata={'source': 'loop', 'step': 4, 'writes': {'llm': {'messages': [AIMessage(content="I found information about the weather in Louisiana. The average temperatures in Louisiana in October range between 20°C and 27°C. There may be patchy rain possible, moderate or heavy rain with thunder, cloudy, partly cloudy, and clear days throughout the month. It's advisable to bring an umbrella as there could be 3 to 8 days of rain in October.\n\nIf you were referring to Los Angeles (LA) in California, please let me know so I can provide you with the specific weather information for that location.", response_metadata={'finish_reason': 'stop', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 107, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 903, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 1010}}, id='run-b7e3f01a-ec8d-4a0d-a081-947d9c789216-0')]}}}, created_at='2025-10-25T17:23:52.169466+00:00', parent_config={'configurable': {'thread_id': '3', 'thread_ts': '1f0b1c75-f918-6379-8003-47b5ea262549'}})


# --


# StateSnapshot(values={'messages': [HumanMessage(content='Whats the weather in LA?', id='9e477a83-adba-4ee5-912d-d378993ace7e'), AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"query":"weather in Los Angeles"}', 'name': 'tavily_search_results_json'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 152, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 174}}, id='run-b68d1c1b-a0c6-4cb4-a8ce-1cc6aa8387a9-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Louisiana'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc'}]), ToolMessage(content='[{\'url\': \'https://www.weather25.com/north-america/usa/louisiana?page=month&month=October\', \'content\': \'weather25.com\\nSearch\\nweather in United States\\nRemove from your favorite locations\\nAdd to my locations\\nShare\\nweather in United States\\n\\n# Louisiana weather in October 2025\\n\\nPatchy rain possible\\nModerate or heavy rain with thunder\\nCloudy\\nPartly cloudy\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\n\\n## The average weather in Louisiana in October\\n\\nThe weather in Louisiana in October is hot. The average temperatures are between 20°C and 27°C. [...] | 26 Moderate or heavy rain with thunder 27° /21° | 27 Cloudy 27° /21° | 28 Partly cloudy 24° /18° | 29 Sunny 21° /15° | 30 Sunny 18° /11° | 31 Sunny 19° /11° |  | [...] You can expect about 3 to 8 days of rain in Louisiana during the month of October. It’s a good idea to bring along your umbrella so that you don’t get caught in poor weather.\\n\\nOur weather forecast can give you a great sense of what weather to expect in Louisiana in October 2025.\\n\\nIf you’re planning to visit Louisiana in the near future, we highly recommend that you review the 14 day weather forecast for Louisiana before you arrive.\\n\\nTemperatures\\nRainy Days\\nSnowy Days\\nDry Days\\nRainfall\\n11.3\'}, {\'url\': \'https://www.accuweather.com/en/us/new-orleans/70112/october-weather/348585\', \'content\': "# New Orleans, LA\\n\\nNew Orleans\\n\\nLouisiana\\n\\n## Around the Globe\\n\\nAround the Globe\\n\\n### Hurricane Tracker\\n\\n### Severe Weather\\n\\n### Radar & Maps\\n\\n### News & Features\\n\\n### Astronomy\\n\\n### Business\\n\\n### Climate\\n\\n### Health\\n\\n### Recreation\\n\\n### Sports\\n\\n### Travel\\n\\n### Warnings\\n\\n### Data Suite\\n\\n### Forensics\\n\\n### Advertising\\n\\n### Superior Accuracy™\\n\\n### Video\\n\\n### Winter Center\\n\\n## Monthly\\n\\n## October\\n\\n## 2025\\n\\n## Daily\\n\\n## Temperature Graph\\n\\n## Further Ahead\\n\\nFurther Ahead\\n\\n### November 2025 [...] ### December 2025\\n\\n### January 2026\\n\\n## Around the Globe\\n\\nAround the Globe\\n\\n### Hurricane Tracker\\n\\n### Severe Weather\\n\\n### Radar & Maps\\n\\n### News\\n\\n### Video\\n\\n### Winter Center\\n\\nTop Stories\\n\\nHurricane\\n\\nMelissa may reach Category 5, poses great danger to Jamaica, Cuba, Hai...\\n\\n52 minutes ago\\n\\nWeather Forecasts\\n\\nWeather troubles brewing for some trick-or-treaters through Halloween\\n\\n4 hours ago\\n\\nHurricane\\n\\nMelissa, future nor\'easter to team up along US East Coast next week\\n\\n1 hour ago\\n\\nHurricane [...] Coast Guard rescues family stranded on island off Cape Cod\\n\\n2 days ago\\n\\nWeather News\\n\\nPolar bears take over abandoned island in Russia\\n\\n5 days ago\\n\\n## Weather Near New Orleans:\\n\\n...\\n\\n...\\n\\n..."}]', name='tavily_search_results_json', id='c87d40ee-76cb-42cc-8af2-34077e6201e1', tool_call_id='call_6ED1ZQ8nrjYIOY14yqInLPZc')]}, next=('llm',), config={'configurable': {'thread_id': '3', 'thread_ts': '1f0b1c75-f918-6379-8003-47b5ea262549'}}, metadata={'source': 'loop', 'step': 3, 'writes': {'action': {'messages': [ToolMessage(content='[{\'url\': \'https://www.weather25.com/north-america/usa/louisiana?page=month&month=October\', \'content\': \'weather25.com\\nSearch\\nweather in United States\\nRemove from your favorite locations\\nAdd to my locations\\nShare\\nweather in United States\\n\\n# Louisiana weather in October 2025\\n\\nPatchy rain possible\\nModerate or heavy rain with thunder\\nCloudy\\nPartly cloudy\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\nClear\\n\\n## The average weather in Louisiana in October\\n\\nThe weather in Louisiana in October is hot. The average temperatures are between 20°C and 27°C. [...] | 26 Moderate or heavy rain with thunder 27° /21° | 27 Cloudy 27° /21° | 28 Partly cloudy 24° /18° | 29 Sunny 21° /15° | 30 Sunny 18° /11° | 31 Sunny 19° /11° |  | [...] You can expect about 3 to 8 days of rain in Louisiana during the month of October. It’s a good idea to bring along your umbrella so that you don’t get caught in poor weather.\\n\\nOur weather forecast can give you a great sense of what weather to expect in Louisiana in October 2025.\\n\\nIf you’re planning to visit Louisiana in the near future, we highly recommend that you review the 14 day weather forecast for Louisiana before you arrive.\\n\\nTemperatures\\nRainy Days\\nSnowy Days\\nDry Days\\nRainfall\\n11.3\'}, {\'url\': \'https://www.accuweather.com/en/us/new-orleans/70112/october-weather/348585\', \'content\': "# New Orleans, LA\\n\\nNew Orleans\\n\\nLouisiana\\n\\n## Around the Globe\\n\\nAround the Globe\\n\\n### Hurricane Tracker\\n\\n### Severe Weather\\n\\n### Radar & Maps\\n\\n### News & Features\\n\\n### Astronomy\\n\\n### Business\\n\\n### Climate\\n\\n### Health\\n\\n### Recreation\\n\\n### Sports\\n\\n### Travel\\n\\n### Warnings\\n\\n### Data Suite\\n\\n### Forensics\\n\\n### Advertising\\n\\n### Superior Accuracy™\\n\\n### Video\\n\\n### Winter Center\\n\\n## Monthly\\n\\n## October\\n\\n## 2025\\n\\n## Daily\\n\\n## Temperature Graph\\n\\n## Further Ahead\\n\\nFurther Ahead\\n\\n### November 2025 [...] ### December 2025\\n\\n### January 2026\\n\\n## Around the Globe\\n\\nAround the Globe\\n\\n### Hurricane Tracker\\n\\n### Severe Weather\\n\\n### Radar & Maps\\n\\n### News\\n\\n### Video\\n\\n### Winter Center\\n\\nTop Stories\\n\\nHurricane\\n\\nMelissa may reach Category 5, poses great danger to Jamaica, Cuba, Hai...\\n\\n52 minutes ago\\n\\nWeather Forecasts\\n\\nWeather troubles brewing for some trick-or-treaters through Halloween\\n\\n4 hours ago\\n\\nHurricane\\n\\nMelissa, future nor\'easter to team up along US East Coast next week\\n\\n1 hour ago\\n\\nHurricane [...] Coast Guard rescues family stranded on island off Cape Cod\\n\\n2 days ago\\n\\nWeather News\\n\\nPolar bears take over abandoned island in Russia\\n\\n5 days ago\\n\\n## Weather Near New Orleans:\\n\\n...\\n\\n...\\n\\n..."}]', name='tavily_search_results_json', id='c87d40ee-76cb-42cc-8af2-34077e6201e1', tool_call_id='call_6ED1ZQ8nrjYIOY14yqInLPZc')]}}}, created_at='2025-10-25T17:23:50.846518+00:00', parent_config={'configurable': {'thread_id': '3', 'thread_ts': '1f0b1c75-d528-6860-8002-1a0422a4eee9'}})


# --


# StateSnapshot(values={'messages': [HumanMessage(content='Whats the weather in LA?', id='9e477a83-adba-4ee5-912d-d378993ace7e'), AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"query":"weather in Los Angeles"}', 'name': 'tavily_search_results_json'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 152, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 174}}, id='run-b68d1c1b-a0c6-4cb4-a8ce-1cc6aa8387a9-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Louisiana'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc'}])]}, next=('action',), config={'configurable': {'thread_id': '3', 'thread_ts': '1f0b1c75-d528-6860-8002-1a0422a4eee9'}}, metadata={'source': 'update', 'step': 2, 'writes': {'llm': {'messages': [HumanMessage(content='Whats the weather in LA?', id='9e477a83-adba-4ee5-912d-d378993ace7e'), AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"query":"weather in Los Angeles"}', 'name': 'tavily_search_results_json'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 152, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 174}}, id='run-b68d1c1b-a0c6-4cb4-a8ce-1cc6aa8387a9-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Louisiana'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc'}])]}}}, created_at='2025-10-25T17:23:47.078350+00:00', parent_config={'configurable': {'thread_id': '3', 'thread_ts': '1f0b1c75-d4c4-657b-8001-715a6488eca1'}})


# --


# StateSnapshot(values={'messages': [HumanMessage(content='Whats the weather in LA?', id='9e477a83-adba-4ee5-912d-d378993ace7e'), AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"query":"weather in Los Angeles"}', 'name': 'tavily_search_results_json'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 152, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 174}}, id='run-b68d1c1b-a0c6-4cb4-a8ce-1cc6aa8387a9-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'weather in Los Angeles'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc'}])]}, next=('action',), config={'configurable': {'thread_id': '3', 'thread_ts': '1f0b1c75-d4c4-657b-8001-715a6488eca1'}}, metadata={'source': 'loop', 'step': 1, 'writes': {'llm': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"query":"weather in Los Angeles"}', 'name': 'tavily_search_results_json'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 152, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 174}}, id='run-b68d1c1b-a0c6-4cb4-a8ce-1cc6aa8387a9-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'weather in Los Angeles'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc'}])]}}}, created_at='2025-10-25T17:23:47.037306+00:00', parent_config={'configurable': {'thread_id': '3', 'thread_ts': '1f0b1c75-d490-6189-8000-48ad306e2ded'}})


# --


# StateSnapshot(values={'messages': [HumanMessage(content='Whats the weather in LA?', id='9e477a83-adba-4ee5-912d-d378993ace7e')]}, next=('llm',), config={'configurable': {'thread_id': '3', 'thread_ts': '1f0b1c75-d490-6189-8000-48ad306e2ded'}}, metadata={'source': 'loop', 'step': 0, 'writes': None}, created_at='2025-10-25T17:23:47.015907+00:00', parent_config={'configurable': {'thread_id': '3', 'thread_ts': '1f0b1c75-d48d-6cb6-bfff-62a64672aab3'}})


# --


# StateSnapshot(values={'messages': []}, next=('__start__',), config={'configurable': {'thread_id': '3', 'thread_ts': '1f0b1c75-d48d-6cb6-bfff-62a64672aab3'}}, metadata={'source': 'input', 'step': -1, 'writes': {'messages': [HumanMessage(content='Whats the weather in LA?')]}}, created_at='2025-10-25T17:23:47.014968+00:00', parent_config=None)


# --

#### Choosing a specific snapshot to replay
to_replay = states[-3]
for event in abot.graph.stream(None, to_replay.config):
    for k, v in event.items():
        print(v)
#### Modifiying the specific snapshot
_id = to_replay.values['messages'][-1].tool_calls[0]['id']
to_replay.values['messages'][-1].tool_calls = [{'name': 'tavily_search_results_json',
  'args': {'query': 'current weather in LA, accuweather'},
  'id': _id}]
#### Store the updated state in a specific "branch_state" variable
branch_state = abot.graph.update_state(to_replay.config, to_replay.values)
print(branch_state)
    #output:
    # {'messages': [HumanMessage(content='Whats the weather in LA?', id='9e477a83-adba-4ee5-912d-d378993ace7e'), AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"query":"weather in Los Angeles"}', 'name': 'tavily_search_results_json'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 152, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 174}}, id='run-b68d1c1b-a0c6-4cb4-a8ce-1cc6aa8387a9-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in LA, accuweather'}, 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc'}])]}
#### Continue execution from modified state
for event in abot.graph.stream(None, branch_state):
    for k, v in event.items():
        if k != "__end__":
            print(v)













##### Debugging LLMs
    # A new random msg can be appended by making sure that it's 
            #"msg.id" is NOT found in the "Message" snapshot list
        # For more info, see "reduce_message" custom fcn
#### Case details for creating a mock LLM response
    # we'll use "to_replay" snapshot, which does NOT contain the 
            # execution of the tool
        # Only contains the initial query and LLM's tool decision
    # We'll continue with the user query "waether in LA?"
    # The PREDICTED agent path for this initial query is as follows:
        # Create "HumanMessage" for the initial user query
        # "AIMessage" will be LLM's interpretation of query as well
                # as the tools (and inputs) LLM has decided
        # "ToolMessage" will be chosen tool's execution with the 
                # inputs that were decided by the LLM itself
        # "AIMessage" is the final LLM response to user query using
                # the results from the tool's execution
    # NOte: "to_replay" does not have "ToolMessage" bc tool not executed
    # Thus we'll pretend was executed by adding a "ToolMessage" with
            #a randomly absurd temperature for LA weather
        # Something like LA weather is currently 54 degrees celsius
    # We'll investigate how the LLM responds to this mocked info
            #that mocks the results that would be given from the
            # chosen tools' execution
        # Note, we WON'T be changing the LLM's chosen tool or it's inputs
        # We'll only change the "content" (i.e., response) of the tool
#### Select the specific tool that chosen by the LLM
_id = to_replay.values['messages'][-1].tool_calls[0]['id']
#### Creating the fake "ToolMessage"
    #This will suppose that tool replied with an absurd temperature
state_update = {"messages": [ToolMessage(
    tool_call_id=_id,
    name="tavily_search_results_json",
    content="54 degree celcius",
)]}
### Ipdate "to_replay" with fake "ToolMessage"
branch_and_add = abot.graph.update_state(
    to_replay.config, 
    state_update, 
    as_node="action")
    # IMPORTANTE
    # "as_node" = you put what node it's faking to execute
        # This will decide what would be placed in "snapshot.next"
#### continute graph execution using the updated state snapshot
for event in abot.graph.stream(None, branch_and_add):
    for k, v in event.items():
        print(v)



































##########################################################################################
##########################################################################################
    # State Snapshot Memory Agent #
from dotenv import load_dotenv

_ = load_dotenv()
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langgraph.checkpoint.sqlite import SqliteSaver




##### Planilla para custom agente
#### Agent State
class AgentState(TypedDict):
    lnode: str
        # last node
    scratch: str
        # scratchpad location
        # where agent will input info as is needed
    count: Annotated[int, operator.add]
        # counter that increases after each step
#### Define node fcnalities and conditionals
def node1(state: AgentState):
    print(f"node1, count:{state['count']}")
        #Note: since the print is before the return:
            # The count will be the old count
            # It's once return is reached that the counter will increase
                # This will show up in snapshot but not in the print
    return {"lnode": "node_1",
            "count": 1,
           }
def node2(state: AgentState):
    print(f"node2, count:{state['count']}")
    return {"lnode": "node_2",
            "count": 1,
           }

def should_continue(state):
    return state["count"] < 3
#### Create agent graph
builder = StateGraph(AgentState)
builder.add_node("Node1", node1)
builder.add_node("Node2", node2)

builder.add_edge("Node1", "Node2")
builder.add_conditional_edges("Node2", 
                              should_continue, 
                              {True: "Node1", False: END})
builder.set_entry_point("Node1")
    #GRaph pseudo diagram:
        # __start__ -> N1 (count 0) -> N2 (count 1) -> 
                # N1 (count 2) -> N2 (count 3) -> END
    # Note, the conditional is only on node 2
#### SEtup in memory persistence and initialize agent
memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)













##### Explorando state de custom agente
#### Init convo with agent
thread = {"configurable": {"thread_id": str(1)}}
graph.invoke({"count":0, "scratch":"hi"},thread)
    #Output:    #Occurs bc of the prints inside node fcnality
        # node1, count:0
        # node2, count:1
        # node1, count:2
        # node2, count:3

        # {'lnode': 'node_2', 'scratch': 'hi', 'count': 4}
    # The last {} is actually the state snapshot after invoke is done
        #This is printed by default on every "graph.invoke"
    # We can see that the last count (3) != snapshot count
        # This confirms that node fcnality logic happens before
                # the state gets updated
#### EXploring current state snapshot
state = graph.get_state(thread)
print(state)
    # output:   #Will show latest snapshot
    # StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hi', 'count': 4}, next=(), config={'configurable': {'thread_id': '1', 'thread_ts': '1f0b1c76-3bdb-680e-8004-86f40af7cc6b'}}, metadata={'source': 'loop', 'step': 4, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-10-25T17:23:57.847127+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1f0b1c76-3bd8-64a8-8003-5c9a172ac4dc'}})
#### Explorando attributos
for attr in dir(state):
    if attr.startswith("_"):
        continue
    try:
        print(f"Attr: {attr}")
        print(state.__getattribute__(attr))
        print('\n\n')
    except:
        print(f"couldn't print {attr}")
        print('\n\n')
        continue
#### Explorando snapshot historia
states = []
for state in graph.get_state_history(thread):
    print(state, "\n")
    states.append(state.config)
    print(state.config, state.values['count'])
    print('\n\n\n')









##### Time travel con custom agente
#### Choose a specific "StateSnapshot" given a specific "configurable"
state = graph.get_state(states[-3])
### Continue graph execution from the chosen snapshot
graph.invoke(None, states[-3])









##### Explorando historia post-time travel
thread = {"configurable": {"thread_id": str(1)}}
for state in graph.get_state_history(thread):
    print(state.config, state.values['count'])
    print('\n\n\n')












##### Exploring snapshot's parent-child realtion after time travel:
for state in graph.get_state_history(thread):
    print((f"""
        {state.config['configurable']['thread_ts']},
        {state.paren_config['configurable']['thread_ts']},
        {state.values['count']}
    """))
    print('\n')
    #Output:    
            #('a356', '53d4', 4)
            #('53d4', '2720', 3)
            #('2720', '102b', 2) #Note '102b' parent
            #('53df', '4a2e', 4)
            #('4a2e', 'ac4f', 3)
            #('ac4f', '102b', 2) #Note '102b' parent
            #('102b', '9055', 1)
            #('9055', 'b349', 0)
            #('b349', None, 0)
                #Recall, this is state[-1]
                    #I.e., state BEFORE "__start__" node
                    #Thus it won't have parent node
    #Note, these are fake values to match book
        #I.e., these numbers WONT match the numbers found above









#### Visualize the graph
from IPython.display import Image
Image(graph.get_graph().draw_png())

















##### Modify a prev state to use in new thread
#### Start with a fresh thread to clear history
thread2 = {"configurable": {"thread_id": str(2)}}
graph.invoke({"count":0, "scratch":"hi"},thread2)
#### Get state history, store it, and anlyze it
states2 = []
for state in graph.get_state_history(thread2):
    states2.append(state.config)
    print(state.config, state.values['count']) 
    print('\n\n\n') 
#### Select an old state to work with
save_state = graph.get_state(states2[-3])
    # REcall: 'states2[-3]' = snapshot after "node1" but b4 "node2"
state = save_state
print(state)
#### Modify the chosen state above
save_state.values["count"] = -3
save_state.values["scratch"] = "hello"
state = save_state
print(state)
#### Actually use "update_state"
graph.update_state(thread2,save_state.values)
#### Exploring state history after "update_state" step is done
for i, state in enumerate(graph.get_state_history(thread2)):
    print(state, '\n')
    print('\n\n\n')  

















##### Exploring ".update_state.as_node" param.
#### Create modofications to a chosen snapshot
    #this will simply reuse "save_state" from line 2527
save_state.values["count"] = -3
save_state.values["scratch"] = "hello"
state = save_state
print(state)
#### Apply "update_state" with ".as_node" param
graph.update_state(thread2,save_state.values, as_node="Node1")
#### Exploring state hsitory after "update_state.as_node"
for i, state in enumerate(graph.get_state_history(thread2)):
    if i >= 3:  #print latest 3
        break
    print(state, '\n')
    print('\n\n\n') 
#### Invoking on the latest/current state (post-modification)
graph.invoke(None,thread2)
#### Explore state history post invocation
for state in graph.get_state_history(thread2):
    print(state,"\n")
    print('\n\n\n')  
































##########################################################################################
##########################################################################################
    # Essay Writer #
#### Import libraries and setup environment
from dotenv import load_dotenv

_ = load_dotenv()
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

memory = SqliteSaver.from_conn_string(":memory:")

#### Create agent state
class AgentState(TypedDict):
    task: str
        # What we're trying to write an essay about
        # This is where the HUMAN INPUT goes into
    plan: str
        # Keeps track of the plan that "planner" node wil generate
    draft: str
        # Draft of the essay
    critique: str
        # Will be populated by the "reflect" node
    content: List[str]
        # Keeps track of list of docs retrieved by search-tool
    revision_number: int
        # Number of revisions made so far
    max_revisions: int
        # Max revisions that we want to make

#### Init model and create prompts
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""

#### Create pydantic model and search tool
    # Ensures that the queries are a LIST of strings
        # These queries will be passed to the search-tool
from langchain_core.pydantic_v1 import BaseModel
from tavily import TavilyClient
import os

class Queries(BaseModel):  
    #Pydantic model
    queries: List[str]

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

#### Setting up ALL nodes' fcnalities and conditional edge logic
def plan_node(state: AgentState):
    # Creates a plan for how to write essay
    messages = [
        SystemMessage(content=PLAN_PROMPT), 
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
        # Returns plan generated by LLM
    return {"plan": response.content}

def research_plan_node(state: AgentState):
    # Generates search queries and the search results (of those queries)
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
        # GEnerates search queries
    content = state['content'] or []
        # Fetches "content" attr, which will store the search results
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
            #Performs search result
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def generation_node(state: AgentState):
    # Writes the essay draft
    content = "\n\n".join(state['content'] or [])
        # Turns "content" search result into a SINGLE STRING
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = model.invoke(messages)
        # returns essay generated by LLM
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    # Creates essay critique
    messages = [
        SystemMessage(content=REFLECTION_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
        # returns critique generated by LLM
    return {"critique": response.content}

def research_critique_node(state: AgentState):
    # GEnerates NEW RESEARCH based on critique
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
                # appends new research to prev. existing research
    return {"content": content}

def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

#### Creating agent graph and initialize it
builder = StateGraph(AgentState)

builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "generate", 
    should_continue, 
    {END: END, "reflect": "reflect"}
)

builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")

builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

graph = builder.compile(checkpointer=memory)

#### Visualize the graph
from IPython.display import Image

Image(graph.get_graph().draw_png())

#### Start conversation and stream "Message" results
thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream({
    'task': "what is the difference between langchain and langsmith",
    "max_revisions": 2,
    "revision_number": 1,
}, thread):
    print(s)
#Output:    # Will display the agent state after each node 
# {'planner': {'plan': 'I. Introduction\n    A. Brief overview of Langchain and Langsmith\n    B. Thesis statement: Exploring the differences between Langchain and Langsmith\n\nII. Langchain\n    A. Definition and explanation\n    B. Key features and characteristics\n    C. Use cases and applications\n    D. Advantages and disadvantages\n\nIII. Langsmith\n    A. Definition and explanation\n    B. Key features and characteristics\n    C. Use cases and applications\n    D. Advantages and disadvantages\n\nIV. Comparison between Langchain and Langsmith\n    A. Technology stack\n    B. Scalability\n    C. Security\n    D. Performance\n    E. Adoption and popularity\n\nV. Conclusion\n    A. Recap of main differences between Langchain and Langsmith\n    B. Future outlook for both technologies\n    C. Final thoughts on the significance of understanding these differences'}}
# {'research_plan': {'content': ['If you’re responsible for ensuring your AI models work in production, or you need to frequently debug and monitor your pipelines, Langsmith is your go-to tool. In short, while **Langchain** excels at managing and scaling model workflows, **Langsmith** is designed for those times when you need deep visibility and control over large, complex AI systems in production. But if you’re managing a **complex AI pipeline** with multiple models that need debugging and orchestrating, Langsmith’s capabilities become essential. If you’re debugging complex AI models or managing large-scale workflows with multiple moving parts, **Langsmith’s advanced debugging and orchestration features** will be indispensable. Additionally, if you’re working on **cross-platform model deployments** — say, running models on-prem and in the cloud simultaneously — Langsmith offers better orchestration and monitoring tools to handle the complexity.', 'In LLM application development, LangChain and LangSmith have become central tools for building and managing large language model-powered solutions. This article compares LangChain and LangSmith, focusing on their core features, integration options, and value for developers in the LLM application space. LangChain is an open-source framework that helps developers create LLM applications efficiently. LangSmith provides tools to debug, monitor, and improve LLM-powered agents, and offers a managed cloud service with a web UI. | LLM Evaluation | Minimal built-in support; developers typically create custom logic or use external tools. | Key Features | Modular chains and sequences, prompt templates, agent framework, data connectors, wide model support, community integrations. LangChain provides building blocks for LLM applications, while LangSmith offers observability and evaluation.', 'If you’re responsible for ensuring your AI models work in production, or you need to frequently debug and monitor your pipelines, Langsmith is your go-to tool. In short, while **Langchain** excels at managing and scaling model workflows, **Langsmith** is designed for those times when you need deep visibility and control over large, complex AI systems in production. But if you’re managing a **complex AI pipeline** with multiple models that need debugging and orchestrating, Langsmith’s capabilities become essential. If you’re debugging complex AI models or managing large-scale workflows with multiple moving parts, **Langsmith’s advanced debugging and orchestration features** will be indispensable. Additionally, if you’re working on **cross-platform model deployments** — say, running models on-prem and in the cloud simultaneously — Langsmith offers better orchestration and monitoring tools to handle the complexity.', 'In LLM application development, LangChain and LangSmith have become central tools for building and managing large language model-powered solutions. This article compares LangChain and LangSmith, focusing on their core features, integration options, and value for developers in the LLM application space. LangChain is an open-source framework that helps developers create LLM applications efficiently. LangSmith provides tools to debug, monitor, and improve LLM-powered agents, and offers a managed cloud service with a web UI. | LLM Evaluation | Minimal built-in support; developers typically create custom logic or use external tools. | Key Features | Modular chains and sequences, prompt templates, agent framework, data connectors, wide model support, community integrations. LangChain provides building blocks for LLM applications, while LangSmith offers observability and evaluation.', 'If you’re responsible for ensuring your AI models work in production, or you need to frequently debug and monitor your pipelines, Langsmith is your go-to tool. In short, while **Langchain** excels at managing and scaling model workflows, **Langsmith** is designed for those times when you need deep visibility and control over large, complex AI systems in production. But if you’re managing a **complex AI pipeline** with multiple models that need debugging and orchestrating, Langsmith’s capabilities become essential. If you’re debugging complex AI models or managing large-scale workflows with multiple moving parts, **Langsmith’s advanced debugging and orchestration features** will be indispensable. Additionally, if you’re working on **cross-platform model deployments** — say, running models on-prem and in the cloud simultaneously — Langsmith offers better orchestration and monitoring tools to handle the complexity.', "LangSmith steps in to give you the tools you need to debug and monitor your models at scale, ensuring everything is running as expected in your AI system. You might think of LangSmith as LangChain's counterpart, but it takes things further by focusing on managing, debugging, and orchestrating AI and ML models. LangSmith steps in to give you the tools you need to debug and monitor your models at scale, ensuring everything is running as expected in your AI system. In short, while LangChain excels at managing and scaling model workflows, LangSmith is designed for when you need deep visibility and control over large, complex AI systems in production. If you're debugging complex AI models or managing large-scale workflows with multiple moving parts, LangSmith's advanced debugging and orchestration features will be indispensable."]}}
# {'generate': {'draft': "**Title: A Comparative Analysis of Langchain and Langsmith in AI Model Management**\n\nI. Introduction\nLangchain and Langsmith are two prominent tools in the realm of AI model management, each serving distinct purposes in the development and deployment of AI systems. While Langchain focuses on managing and scaling model workflows efficiently, Langsmith is tailored for providing deep visibility and control over complex AI systems in production. This essay delves into the disparities between Langchain and Langsmith to elucidate their unique functionalities and applications.\n\nII. Langchain\nLangchain is an open-source framework designed to streamline the creation of large language model (LLM) applications. It offers modular chains and sequences, prompt templates, an agent framework, data connectors, wide model support, and community integrations. Developers leverage Langchain's building blocks to construct LLM applications efficiently. However, Langchain lacks built-in support for LLM evaluation, often necessitating the use of custom logic or external tools.\n\nIII. Langsmith\nIn contrast, Langsmith provides tools for debugging, monitoring, and enhancing LLM-powered agents. It offers advanced debugging and orchestration features crucial for managing complex AI pipelines with multiple moving parts. Additionally, Langsmith presents a managed cloud service with a web UI, enhancing observability and evaluation capabilities for developers working on large-scale AI systems.\n\nIV. Comparison between Langchain and Langsmith\nA. Technology Stack: Langchain focuses on providing building blocks for LLM applications, while Langsmith emphasizes observability and evaluation tools.\nB. Scalability: Langchain excels in managing and scaling model workflows, whereas Langsmith offers advanced debugging and orchestration features for complex AI systems.\nC. Security: Both Langchain and Langsmith prioritize security; however, Langsmith's monitoring tools enhance security by providing deep visibility into AI systems.\nD. Performance: Langchain enhances performance through efficient workflow management, while Langsmith's debugging features optimize model performance.\nE. Adoption and Popularity: Langchain is favored for its efficiency in LLM application development, while Langsmith gains popularity for its advanced debugging and monitoring capabilities.\n\nV. Conclusion\nIn conclusion, understanding the distinctions between Langchain and Langsmith is paramount for developers navigating the landscape of AI model management. While Langchain streamlines the development of LLM applications, Langsmith offers indispensable tools for debugging and monitoring complex AI systems. By recognizing the strengths and weaknesses of each tool, developers can make informed decisions to optimize their AI workflows effectively. Embracing the unique functionalities of Langchain and Langsmith paves the way for enhanced AI model management and deployment practices in the future.", 'revision_number': 2}}
# {'reflect': {'critique': "**Critique:**\n\nThe essay provides a clear and structured comparison between Langchain and Langsmith in the context of AI model management. The introduction effectively sets the stage for the discussion, outlining the purpose of the essay. The subsequent sections delve into the functionalities of each tool, highlighting their strengths and differences. The conclusion effectively summarizes the key points discussed in the essay.\n\n**Recommendations:**\n\n1. **Depth and Detail:** While the essay provides a good overview of Langchain and Langsmith, consider delving deeper into specific features, functionalities, and use cases of each tool. Providing more detailed examples or case studies could help illustrate the practical applications of Langchain and Langsmith in AI model management.\n\n2. **Expansion on Limitations:** It would be beneficial to include a section discussing the limitations or challenges associated with both Langchain and Langsmith. This could provide a more comprehensive understanding for readers evaluating these tools for their own projects.\n\n3. **Real-world Examples:** Incorporating real-world examples or scenarios where Langchain and Langsmith have been successfully utilized could enhance the essay's credibility and practical relevance.\n\n4. **Comparative Analysis:** While the essay does compare the two tools across different aspects, consider providing a more nuanced analysis by exploring how Langchain and Langsmith complement each other or can be used in conjunction for comprehensive AI model management.\n\n5. **Recommendations for Developers:** Offer specific recommendations or guidelines for developers on when to choose Langchain over Langsmith or vice versa based on project requirements, team expertise, scalability needs, etc.\n\n6. **Length:** Consider expanding the essay to provide a more comprehensive analysis, possibly by including a section on future trends or developments in AI model management tools.\n\nOverall, the essay is well-structured and informative, but incorporating the above recommendations could further enrich the content and provide readers with a more in-depth understanding of Langchain and Langsmith in AI model management."}}

# {'research_critique': {'content': ['If you’re responsible for ensuring your AI models work in production, or you need to frequently debug and monitor your pipelines, Langsmith is your go-to tool. In short, while **Langchain** excels at managing and scaling model workflows, **Langsmith** is designed for those times when you need deep visibility and control over large, complex AI systems in production. But if you’re managing a **complex AI pipeline** with multiple models that need debugging and orchestrating, Langsmith’s capabilities become essential. If you’re debugging complex AI models or managing large-scale workflows with multiple moving parts, **Langsmith’s advanced debugging and orchestration features** will be indispensable. Additionally, if you’re working on **cross-platform model deployments** — say, running models on-prem and in the cloud simultaneously — Langsmith offers better orchestration and monitoring tools to handle the complexity.', 'In LLM application development, LangChain and LangSmith have become central tools for building and managing large language model-powered solutions. This article compares LangChain and LangSmith, focusing on their core features, integration options, and value for developers in the LLM application space. LangChain is an open-source framework that helps developers create LLM applications efficiently. LangSmith provides tools to debug, monitor, and improve LLM-powered agents, and offers a managed cloud service with a web UI. | LLM Evaluation | Minimal built-in support; developers typically create custom logic or use external tools. | Key Features | Modular chains and sequences, prompt templates, agent framework, data connectors, wide model support, community integrations. LangChain provides building blocks for LLM applications, while LangSmith offers observability and evaluation.', 'If you’re responsible for ensuring your AI models work in production, or you need to frequently debug and monitor your pipelines, Langsmith is your go-to tool. In short, while **Langchain** excels at managing and scaling model workflows, **Langsmith** is designed for those times when you need deep visibility and control over large, complex AI systems in production. But if you’re managing a **complex AI pipeline** with multiple models that need debugging and orchestrating, Langsmith’s capabilities become essential. If you’re debugging complex AI models or managing large-scale workflows with multiple moving parts, **Langsmith’s advanced debugging and orchestration features** will be indispensable. Additionally, if you’re working on **cross-platform model deployments** — say, running models on-prem and in the cloud simultaneously — Langsmith offers better orchestration and monitoring tools to handle the complexity.', 'In LLM application development, LangChain and LangSmith have become central tools for building and managing large language model-powered solutions. This article compares LangChain and LangSmith, focusing on their core features, integration options, and value for developers in the LLM application space. LangChain is an open-source framework that helps developers create LLM applications efficiently. LangSmith provides tools to debug, monitor, and improve LLM-powered agents, and offers a managed cloud service with a web UI. | LLM Evaluation | Minimal built-in support; developers typically create custom logic or use external tools. | Key Features | Modular chains and sequences, prompt templates, agent framework, data connectors, wide model support, community integrations. LangChain provides building blocks for LLM applications, while LangSmith offers observability and evaluation.', 'If you’re responsible for ensuring your AI models work in production, or you need to frequently debug and monitor your pipelines, Langsmith is your go-to tool. In short, while **Langchain** excels at managing and scaling model workflows, **Langsmith** is designed for those times when you need deep visibility and control over large, complex AI systems in production. But if you’re managing a **complex AI pipeline** with multiple models that need debugging and orchestrating, Langsmith’s capabilities become essential. If you’re debugging complex AI models or managing large-scale workflows with multiple moving parts, **Langsmith’s advanced debugging and orchestration features** will be indispensable. Additionally, if you’re working on **cross-platform model deployments** — say, running models on-prem and in the cloud simultaneously — Langsmith offers better orchestration and monitoring tools to handle the complexity.', "LangSmith steps in to give you the tools you need to debug and monitor your models at scale, ensuring everything is running as expected in your AI system. You might think of LangSmith as LangChain's counterpart, but it takes things further by focusing on managing, debugging, and orchestrating AI and ML models. LangSmith steps in to give you the tools you need to debug and monitor your models at scale, ensuring everything is running as expected in your AI system. In short, while LangChain excels at managing and scaling model workflows, LangSmith is designed for when you need deep visibility and control over large, complex AI systems in production. If you're debugging complex AI models or managing large-scale workflows with multiple moving parts, LangSmith's advanced debugging and orchestration features will be indispensable.", "[![Image 3](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68e8e27023585643370a6471_icons.svg) LangChain Quick start agents with any model provider](https://www.langchain.com/langchain)[![Image 4](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68e8e270417338c7f027082d_d035ce400e48f9bc4dd0578d0e3e3211_icons-1.svg) LangGraph Build custom agents with low-level control](https://www.langchain.com/langgraph)[![Image 5](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68f20863b71dbae1af829979_DeepAgents.svg) Deep Agents New Use planning, memory, and sub-agents for complex, long-running tasks](https://github.com/langchain-ai/deepagents) [![Image 6](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68e8e270df09334914882b88_Frame%209.svg) Observability Debug and monitor in-depth traces](https://www.langchain.com/langsmith/observability)[![Image 7](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68e8e270f9e8de1d368764a8_Frame%20206.svg) Evaluation Iterate on prompts and models](https://www.langchain.com/langsmith/evaluation)[![Image 8](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68e8e2709eef27fc61465416_Frame%20100039.svg) Deployment Ship and scale agents in production](https://www.langchain.com/langsmith/deployment) [![Image 67](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/680a284548300b0261543d2e_logo_Replit.svg)![Image 68](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/681c7597a78988d463e5d100_Group%2070.svg)](https://blog.langchain.dev/customers-replit/) [![Image 69](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/680a1dfda253f7223b13bac8_d9e260826e5b7426f8f02e0eee665d8b_logo_rakuten.svg)![Image 70](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/681c7597a78988d463e5d100_Group%2070.svg)](https://blog.langchain.dev/customers-rakuten/) [![Image 71](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6819fde80c00e748f53c881f_logo_klarna.svg)![Image 72](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/681c7597a78988d463e5d100_Group%2070.svg)](https://blog.langchain.dev/customers-klarna/) [![Image 73](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6819fe5973e7714382bddd35_logo_morningstar.svg)![Image 74](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/681c7597a78988d463e5d100_Group%2070.svg)](https://blog.langchain.dev/morningstar-intelligence-engine-puts-personalized-investment-insights-at-analysts-fingertips/) [![Image 75](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6819fe7f44e8a32e29c579f7_logo_lovable.svg)![Image 76](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/681c7597a78988d463e5d100_Group%2070.svg)](https://blog.langchain.dev/customers-lovable/) [![Image 77](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6819fda7d5be8df12d501368_The_Home_Depot-Logo.wine%201.svg)](https://www.langchain.com/#) [![Image 78](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/680a2845c7235e92f485a254_Klarna_Logo_black%201.svg) Financial Services Klarna's AI assistant reduced customer query resolution time by 80%, powered by LangSmith and LangGraph.](https://blog.langchain.dev/customers-klarna/)[![Image 79](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68e8de0dceca1502358e8e8f_logo_Elastic.svg) B2B SaaS Elastic’s AI security assistant, built with LangSmith and LangGraph, cut alert response times for 20,000+ customers.](https://blog.langchain.com/langchain-partners-with-elastic-to-launch-the-elastic-ai-assistant/)[![Image 80](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68e8de0d42e1a2f38a3d182d_logo_Replit.svg) AI/ML Replit's AI Agent serves 30+ million developers. ![Image 81](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/65c6a38f9c53ec71f5fc73de_langchain-word.svg)", 'import jsonfrom langchain_core.prompts import ChatPromptTemplatefrom langchain_openai import ChatOpenAIdef get_current_weather(location, unit="fahrenheit"):  weather_info = {      "location": location,      "temperature": "72",      "unit": unit,      "forecast":["sunny","windy"]  }  return json.dumps(weather_info)# define functionfunction = [    {        "name":"get_current_weather",        "description": "get the current weeather in a given location",        "parameters":{            "type":"object",            "properties": {                "location":{                    "type":"string",                    "description":"The city and state name for which you want to know weather, e.g. San Francisco, CA"                },                "unit":{                    "type":"string",                    "enum":["celsius","fahrenheit"]                }            },            "required":["location"],        }    }]# description is really important as it is passed to the LLM and based on it LLM will figure out if this function is to be used or not.prompt = ChatPromptTemplate.from_messages(    [        ("human", "{input}")    ])model = ChatOpenAI(    model="gpt-4",    temperature=0,    max_tokens=None,    timeout=None,    max_retries=2,    api_key=YOUR_API_KEY,  # if you prefer to pass api key in directly instaed of using env vars).bind(functions=function)prompt = ChatPromptTemplate.from_messages(    [        ("human", "{input}")    ])final_prompt = prompt.format_message(input="What\'s the weather like in Boston?")res = model.invoke(final_prompt)print(res) import from import from import def get_current_weatherlocation, unit="fahrenheit" "fahrenheit" "location" "temperature" "72" "unit" "forecast" "sunny" "windy" return', 'LangSmith and AutoGen tackle different but complementary challenges in AI application development: LangSmith excels at making complex language model (LLM) workflows transparent and testable, whereas AutoGen enables the orchestration of multiple AI “agents” to collaborate on tasks. LangSmith is a service offered by the LangChain team that provides end-to-end observability, tracing, and evaluation for LLM applications. Gathering token usage and cost data at the trace level helps teams monitor expenses and optimize prompts based on actual performance (LangSmith OpenTelemetry). LangSmith excels at rigorous monitoring, testing, and optimization of language model workflows, ideal for teams emphasizing precise observability. Choose LangSmith for robust oversight of AI performance, or AutoGen for flexibility and collaborative agent orchestration.', 'Langsmith offers the solution: an all-in-one platform for debugging, testing, evaluating and monitoring LLMs. With tools for data set creation and an interactive playground for optimizing inputs, Langsmith ensures that developers maintain an overview and control at all times. Langsmith offers decisive advantages for developers and companies that rely on language models: * **Wide range of applications:** Suitable for developers who test, optimize and monitor language models, as well as for companies that rely on powerful AI models. Langsmith is primarily focused on testing, monitoring and optimizing language models and supports developers in continuously improving the performance and reliability of their models. Langsmith offers a comprehensive solution for developers who want to efficiently test, monitor and optimize their language models.', '[![Image 3](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68e8e27023585643370a6471_icons.svg) LangChain Quick start agents with any model provider](https://www.langchain.com/langchain)[![Image 4](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68e8e270417338c7f027082d_d035ce400e48f9bc4dd0578d0e3e3211_icons-1.svg) LangGraph Build custom agents with low-level control](https://www.langchain.com/langgraph)[![Image 5](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68f20863b71dbae1af829979_DeepAgents.svg) Deep Agents New Use planning, memory, and sub-agents for complex, long-running tasks](https://github.com/langchain-ai/deepagents) [![Image 6](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68e8e270df09334914882b88_Frame%209.svg) Observability Debug and monitor in-depth traces](https://www.langchain.com/langsmith/observability)[![Image 7](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68e8e270f9e8de1d368764a8_Frame%20206.svg) Evaluation Iterate on prompts and models](https://www.langchain.com/langsmith/evaluation)[![Image 8](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/68e8e2709eef27fc61465416_Frame%20100039.svg) Deployment Ship and scale agents in production](https://www.langchain.com/langsmith/deployment) ![Image 10](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6811d18a7cef47c38c0eeb48_C._H._Robinson_logo%201.svg) ![Image 11](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/680b77461846a7cf254d8391_Klarna_Logo_black%201.svg) ![Image 12](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6811d1aa251143166667aec3_logo_Rakuten.svg) ![Image 13](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6811d1d39b2c6c806093f171_GitLab_logo_(2)%201.svg) ![Image 14](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/680b77fc1381ca4ebea292b0_logo_Replit.svg) ![Image 15](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6811d1c1c55df212370b53fd_logo_Elastic.svg) ![Image 17](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/681b568070e9f341ed73b877_logo_Cisco.svg) ![Image 19](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6811d18a7cef47c38c0eeb48_C._H._Robinson_logo%201.svg) ![Image 20](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/680b77461846a7cf254d8391_Klarna_Logo_black%201.svg) ![Image 21](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6811d1aa251143166667aec3_logo_Rakuten.svg) ![Image 22](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6811d1d39b2c6c806093f171_GitLab_logo_(2)%201.svg) ![Image 23](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/680b77fc1381ca4ebea292b0_logo_Replit.svg) ![Image 24](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6811d1c1c55df212370b53fd_logo_Elastic.svg) ![Image 26](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/681b568070e9f341ed73b877_logo_Cisco.svg) [](https://www.langchain.com/customers#)![Image 27](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/683ebed2db7596a00aeb6883_Frame%20441.webp) ![Image 28](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/683ed8583aab2220a0584112_logo_morningstar.svg) [](https://www.langchain.com/customers#)![Image 29](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/683ed5b3c305e1f0810ee165_Zrzut%20ekranu%202025-05-29%20o%2008.39.51%201.webp) [](https://www.langchain.com/customers#)![Image 31](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/683ed6ec5f21e86c8f67b24e_Zrzut%20ekranu%202025-05-29%20o%2008.40.28%201.webp) ![Image 32](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/683ed87d3aab2220a0585c54_logo_Rakuten.svg) [](https://www.langchain.com/customers#)![Image 33](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/683ed8cdabbd877010c381e7_Zrzut%20ekranu%202025-05-29%20o%2008.42.08%201.webp) ![Image 34](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/683ed8a197c2372e9514cfc8_logo_modern%20treasury.svg) [](https://www.langchain.com/customers#)![Image 35](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6847439da2d2ab267d523496_Screenshot%202025-06-09%20at%201.22.35%E2%80%AFPM.png) ![Image 36](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/68474302ae6163d917d3450b_Pigment%20logo%20svg.svg) [](https://www.langchain.com/customers#)![Image 37](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/684867863ea0b77fe80b1fd7_Screenshot%202025-06-10%20at%2010.12.29%E2%80%AFAM.png) ![Image 38](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/683ed94b51c824411f773561_city-of-hope-logo-vector%201.svg) ![Image 39](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/68403bcb15045b3eb73fa481_logo_Elastic.svg) ![Image 40](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/68403f1e7aafbecebd57a732_logo_Trellix.svg) ![Image 41](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/68403f761830d136e87c2757_logo_replit.svg) ![Image 42](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/68403fd6b7af2641fb80d176_logo_chrobinson.svg) ![Image 43](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6840405d0c7f0d019851a934_logo_Elastic.svg) ![Image 44](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/684040c76e0334123089b2ea_logo_podium.svg) ![Image 45](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/684041be7034e54f87e91948_logo_vizient.svg) ![Image 46](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6840420bbd8fd31bd9231f03_logo_appfolio.svg) ![Image 47](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6840424d80dcf0badfc89841_logo_unify.svg) ![Image 48](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/68404288fb8eec2e1f087004_logo_lovable.svg) ![Image 49](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/684042c46a5fa9ce2365b8e7_logo_vodafone.svg) ![Image 50](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/684296b214bd0067789cc95b_logo_dun.svg) ![Image 51](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/65c6a38f9c53ec71f5fc73de_langchain-word.svg)', '[Newsletter Curated insights on AI, Cloud & System Design](https://www.educative.io/newsletter)[Blog For developers, By developers](https://www.educative.io/blog)[Guides Step-by-step tutorials to master real-world tech skills](https://www.educative.io/guides)[Free Cheatsheets Download handy guides for tech topics](https://www.educative.io/cheatsheets)[Games Sharpen your skills with daily challenges](https://www.educative.io/games)[Compilers Execute code in an interactive environment](https://www.educative.io/compilers) [Preview](https://www.educative.io/courses/langchain-llm) 12 ways developers are using LangChain[#](https://www.educative.io/blog/langchain-usecases#12-ways-developers-are-using-LangChain) 1. Intelligent chatbots with memory[#](https://www.educative.io/blog/langchain-usecases#1-Intelligent-chatbots-with-memory) [Preview](https://www.educative.io/courses/build-your-own-chatbot-in-python) 2. Q&A over private documents and databases[#](https://www.educative.io/blog/langchain-usecases#2-QampA-over-private-documents-and-databases) 3. Code generation and debugging assistants[#](https://www.educative.io/blog/langchain-usecases#3-Code-generation-and-debugging-assistants) 4. Automated research and analysis bots[#](https://www.educative.io/blog/langchain-usecases#4-Automated-research-and-analysis-bots) 5. Workflow automation for productivity[#](https://www.educative.io/blog/langchain-usecases#5-Workflow-automation-for-productivity) 6. Legal document review and summarization[#](https://www.educative.io/blog/langchain-usecases#6-Legal-document-review-and-summarization) 7. Custom AI tutors and interactive learning apps[#](https://www.educative.io/blog/langchain-usecases#7-Custom-AI-tutors-and-interactive-learning-apps) 8. Multi-step report generation[#](https://www.educative.io/blog/langchain-usecases#8-Multi-step-report-generation) 9. Multi-agent collaboration environments[#](https://www.educative.io/blog/langchain-usecases#9-Multi-agent-collaboration-environments) Voice-to-insight apps and transcript analysis[#](https://www.educative.io/blog/langchain-usecases#10-Voice-to-insight-apps-and-transcript-analysis) Personalized e-commerce and product recommendation assistants[#](https://www.educative.io/blog/langchain-usecases#11-Personalized-e-commerce-and-product-recommendation-assistants) Real-time data monitoring and anomaly detection[#](https://www.educative.io/blog/langchain-usecases#12-Real-time-data-monitoring-and-anomaly-detection) Final word[#](https://www.educative.io/blog/langchain-usecases#Final-word)']}}

# {'generate': {'draft': "**Title: Exploring the Contrasts Between Langchain and Langsmith**\n\nI. Introduction\nLangchain and Langsmith are two essential tools in the realm of AI application development. While both serve crucial roles, they cater to distinct needs and functionalities. This essay delves into the disparities between Langchain and Langsmith to provide a comprehensive understanding of their unique characteristics.\n\nII. Langchain\nLangchain is an open-source framework designed to facilitate the efficient creation of Language Model (LLM) applications. It offers modular chains and sequences, prompt templates, an agent framework, data connectors, wide model support, and community integrations. Langchain serves as a foundational tool for developers to build LLM applications effectively, albeit with minimal built-in support.\n\nIII. Langsmith\nIn contrast, Langsmith focuses on providing tools for debugging, monitoring, and enhancing LLM-powered agents. It offers advanced features for observability and evaluation, making it indispensable for managing large-scale workflows with multiple moving parts. Langsmith also provides a managed cloud service with a user-friendly web UI for enhanced accessibility.\n\nIV. Comparison between Langchain and Langsmith\nA. Technology Stack: Langchain emphasizes building blocks for LLM applications, while Langsmith prioritizes observability and evaluation tools.\nB. Scalability: Langchain excels in managing and scaling model workflows, whereas Langsmith offers advanced debugging and orchestration features for complex AI systems.\nC. Security: Both Langchain and Langsmith prioritize security, but Langsmith's focus on monitoring and evaluation enhances security measures.\nD. Performance: Langsmith's advanced debugging capabilities contribute to optimizing performance, while Langchain's modular approach aids in efficient application development.\nE. Adoption and Popularity: Langsmith's comprehensive monitoring and debugging features have garnered popularity among developers, while Langchain's open-source framework appeals to those seeking customizable solutions.\n\nV. Conclusion\nIn conclusion, understanding the distinctions between Langchain and Langsmith is crucial for developers and organizations working with AI applications. While Langchain offers foundational building blocks for LLM applications, Langsmith provides advanced tools for monitoring and debugging complex AI systems. Both tools play vital roles in the AI development landscape, and recognizing their unique strengths is essential for leveraging them effectively in various scenarios.", 'revision_number': 3}}


#### Creating a GUI for the essay writer
import warnings
warnings.filterwarnings("ignore")

from helper import ewriter, writer_gui
MultiAgent = ewriter()
app = writer_gui(MultiAgent.graph)
app.launch()














































#########################################################################
#########################################################################
### C8 ###
#########################################################################
#########################################################################
    # Assistente Autonomo - Base #
import os
from dotenv import load_dotenv
_ = load_dotenv()
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model
from prompts import triage_system_prompt, triage_user_prompt
from langchain_core.tools import tool
from prompts import agent_system_prompt
from langgraph.prebuilt import create_react_agent
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from IPython.display import Image, display

##### Router Agente
#### Setup General
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

prompt_instructions = {
    "triage_rules": { # Se defininen en 'triage_system_prompt'
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}

email = { # Example incoming email
    "from": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "body": """
Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}
#### Modelo Pydantic 
    #Ensures that router agent will have and output dict with
            #"reasining" and "classication" keys
class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
        # Reasoning behind why LLM made the decision it chose
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )   #  Los 'ignore se definen abajo en 'triage_system_prompt'
#### Creando Router agente
llm = init_chat_model("openai:gpt-4o-mini")
llm_router = llm.with_structured_output(Router)






##### Diferentes formas de ingenieria de prompt
#### Prompts GENERALES
print(triage_system_prompt)
#Output:
# < Role >
# You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
# </ Role >

# < Background >
# {user_profile_background}. 
# </ Background >

# < Instructions >

# {name} gets lots of emails. Your job is to categorize each email into one of three categories:

# 1. IGNORE - Emails that are not worth responding to or tracking
# 2. NOTIFY - Important information that {name} should know about but doesn't require a response
# 3. RESPOND - Emails that need a direct response from {name}

# Classify the below email into one of these categories.

# </ Instructions >

# < Rules >
# Emails that are not worth responding to:
# {triage_no}

# There are also other things that {name} should know about, but don't require an email response. For these, you should notify {name} (using the `notify` response). Examples of this include:
# {triage_notify}

# Emails that are worth responding to:
# {triage_email}
# </ Rules >

# < Few shot examples >
# {examples}
# </ Few shot examples >
print(triage_user_prompt)
#Output:
# Please determine how to handle the below email thread:

# From: {author}
# To: {to}
# Subject: {subject}
# {email_thread}
print(agent_system_prompt)
#output:
# < Role >
# You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
# </ Role >

# < Tools >
# You have access to the following tools to help manage {name}'s communications and schedule:

# 1. write_email(to, subject, content) - Send emails to specified recipients
# 2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings
# 3. check_calendar_availability(day) - Check available time slots for a given day
# </ Tools >

# < Instructions >
# {instructions}
# </ Instructions >
#### Transformacion con '.format': de general a especifico
system_prompt = triage_system_prompt.format(
    full_name=profile["full_name"],
    name=profile["name"],
    examples=None,
    user_profile_background=profile["user_profile_background"],
    triage_no=prompt_instructions["triage_rules"]["ignore"],
    triage_notify=prompt_instructions["triage_rules"]["notify"],
    triage_email=prompt_instructions["triage_rules"]["respond"],
)
user_prompt = triage_user_prompt.format(
    author=email["from"],
    to=email["to"],
    subject=email["subject"],
    email_thread=email["body"],
)
#### Transformacion con funciones: de general a especifico
def create_prompt(state):
    return [
        {
            "role": "system", 
            "content": agent_system_prompt.format(
                instructions=prompt_instructions["agent_instructions"],
                **profile
                )
        }
    ] + state['messages']
#### Invocacion
result = llm_router.invoke(
    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
)
print(type(result))
    #output: <class '__main__.Router'>
print(result)











##### Pre-built ReAct agente
#### Creando herramientas
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    # Placeholder response - in real app would check calendar and schedule
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"
#### Creando pre-built ReAct
tools=[write_email, schedule_meeting, check_calendar_availability]
agent = create_react_agent(
    "openai:gpt-4o",
    tools=tools,
    prompt=create_prompt,
)
print(dir(agent))
#### INvoke main state-agent and explore
response = agent.invoke(
    {"messages": [{
        "role": "user", 
        "content": "what is my availability for tuesday?"
    }]}
)

response["messages"][-1].pretty_print()
#Output:
# ================================== Ai Message ==================================

# You have the following available time slots on Tuesday: 

# - 9:00 AM
# - 2:00 PM
# - 4:00 PM

# If you need to schedule a meeting or an appointment, please let me know how I can assist you further!















##### MAS-grafo
#### Estado del agente
class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]
#### Node para router agente
def triage_router(state: State) -> Command[
    Literal["response_agent", "__end__"]
]:
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples=None
    )
    user_prompt = triage_user_prompt.format(
        author=author, 
        to=to, 
        subject=subject, 
        email_thread=email_thread
    )
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ]
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update = None
        goto = END
    elif result.classification == "notify":
        # If real life, this would do something else
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)
#### Creando MAS - grafo
email_agent = StateGraph(State)
email_agent = email_agent.add_node(triage_router)
email_agent = email_agent.add_node("response_agent", agent)
email_agent = email_agent.add_edge(START, "triage_router")
email_agent = email_agent.compile()
#### Visualizando grafo
display(Image(email_agent.get_graph(xray=True).draw_mermaid_png()))
#### Comparando 'StateGraph'
### Uninitialized
print(dir(StateGraph))
### Initialzied
print(dir(email_agent))
#### Invocacion de MAS
email_input = {
    "author": "Marketing Team <marketing@amazingdeals.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "🔥 EXCLUSIVE OFFER: Limited Time Discount on Developer Tools! 🔥",
    "email_thread": """Dear Valued Developer,

Don't miss out on this INCREDIBLE opportunity! 

🚀 For a LIMITED TIME ONLY, get 80% OFF on our Premium Developer Suite! 

✨ FEATURES:
- Revolutionary AI-powered code completion
- Cloud-based development environment
- 24/7 customer support
- And much more!

💰 Regular Price: $999/month
🎉 YOUR SPECIAL PRICE: Just $199/month!

🕒 Hurry! This offer expires in:
24 HOURS ONLY!

Click here to claim your discount: https://amazingdeals.com/special-offer

Best regards,
Marketing Team
---
To unsubscribe, click here
""",
}

response = email_agent.invoke({"email_input": email_input})
    #Output:
    # 🚫 Classification: IGNORE - This email can be safely ignored


email_input = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}

response = email_agent.invoke({"email_input": email_input})
    #output:
    # 📧 Classification: RESPOND - This email requires a response

for m in response["messages"]:
    m.pretty_print()
#Otuput:
# ================================ Human Message =================================

# Respond to the email {'author': 'Alice Smith <alice.smith@company.com>', 'to': 'John Doe <john.doe@company.com>', 'subject': 'Quick question about API documentation', 'email_thread': "Hi John,\n\nI was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?\n\nSpecifically, I'm looking at:\n- /auth/refresh\n- /auth/validate\n\nThanks!\nAlice"}
# ================================== Ai Message ==================================
# Tool Calls:
#   write_email (call_a8HKveq3cxlp90hT1UwXrk4g)
#  Call ID: call_a8HKveq3cxlp90hT1UwXrk4g
#   Args:
#     to: alice.smith@company.com
#     subject: Re: Quick question about API documentation
#     content: Hi Alice,

# Thank you for bringing this to my attention. I'm looking into the API documentation for the new authentication service. It appears that the endpoints /auth/refresh and /auth/validate may have been unintentionally left out. Let me verify this with the development team.

# I'll get back to you with the confirmation and any necessary updates to the documentation by tomorrow.

# Thank you for your diligence!

# Best regards,

# John Doe
# ================================= Tool Message =================================
# Name: write_email

# Email sent to alice.smith@company.com with subject 'Re: Quick question about API documentation'
# ================================== Ai Message ==================================

# I've sent a response to Alice, clarifying the situation about the missing API endpoints and assuring her that you're looking into it with the development team. You promised to get back with a confirmation and any updates by tomorrow. Let me know if you need any more help with this matter!

























#####################################################################
#####################################################################
    # Asistente Autonomo - Mem Semantica #
import os
from dotenv import load_dotenv
_ = load_dotenv()
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from IPython.display import Image, display
from langgraph.graph import add_messages



##### Creando in-mem persistence
#### Creacion 
### Largo-plazo
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)
### Corto-plazo
snapshot_store = InMemorySaver()
#### LG mem. wrappers
manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "email_assistant", 
        "{langgraph_user_id}",
        "collection"
    )
)
search_memory_tool = create_search_memory_tool(
    namespace=(
        "email_assistant",
        "{langgraph_user_id}",
        "collection"
    )
)
#### Explorando mem. wrappers
### Manage Mem
print(manage_memory_tool.name)
    #output: manage_memeory

print(manage_memory_tool.description)
# otuput:
# Create, update, or delete persistent MEMORIES to persist across conversations.
# Include the MEMORY ID when updating or deleting a MEMORY. Omit when creating a new MEMORY - it will be created for you.
# Proactively call this tool when you:

# 1. Identify a new USER preference.
# 2. Receive an explicit USER request to remember something or otherwise alter your behavior.
# 3. Are working and want to record important context.
# 4. Identify that an existing MEMORY is incorrect or outdated.

print(manage_memory_tool.args)
#output:
# {'content': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
#   'default': None,
#   'title': 'Content'},
#  'action': {'default': 'create',
#   'enum': ['create', 'update', 'delete'],
#   'title': 'Action',
#   'type': 'string'},
#  'id': {'anyOf': [{'format': 'uuid', 'type': 'string'}, {'type': 'null'}],
#   'default': None,
#   'title': 'Id'}}

### Search mem
print(search_memory_tool.name)
    #Otuput: search_memory

print(search_memory_tool.description)
#Output:
#'Search your long-term memories for information relevant to your current context.'

print(search_memory_tool.args)
#otuput:
# {'query': {'title': 'Query', 'type': 'string'},
#  'limit': {'default': 10, 'title': 'Limit', 'type': 'integer'},
#  'offset': {'default': 0, 'title': 'Offset', 'type': 'integer'},
#  'filter': {'anyOf': [{'type': 'object'}, {'type': 'null'}],
#   'default': None,
#   'title': 'Filter'}}










##### ReAct con in-mem persistence
#### Prompt
agent_system_prompt_memory = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Tools >
You have access to the following tools to help manage {name}'s communications and schedule:

1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings
3. check_calendar_availability(day) - Check available time slots for a given day
4. manage_memory - Store any relevant information about contacts, actions, discussion, etc. in memory for future reference
5. search_memory - Search for any relevant information that may have been stored in memory
</ Tools >

< Instructions >
{instructions}
</ Instructions >
"""

def create_prompt(state):
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt_instructions["agent_instructions"], 
                **profile
            )
        }
    ] + state['messages']
#### Creando ReAct agente
tools= [
    write_email, 
    schedule_meeting,
    check_calendar_availability,
    manage_memory_tool,
    search_memory_tool
]
response_agent = create_react_agent(
    "openai:gpt-4o-mini",
    tools=tools,
    prompt=create_prompt,
    # Use this to ensure the store is passed to the agent 
    store=store,
    checkpointer=snapshot_store
)



















##### Invocando conversacion con in-mem persistence
#### Creando thread
config = {"configurable": {"langgraph_user_id": "lance", "thread_id":"1"}}
    # OJO: la razon que necesitamos 'langgraph_user_id' es 
            # por la forma en que se creo las herramientas de memoria
        # En especial su 'namespace' donde guardan la informacion
    # Thread will have both:
        #"langgraph_user_id" for long-term mem. (persistence - VectorStore)
        # "thread_id" for short-term mem. (temporal - StateSnapshot)
#### Verificando 'manage_memory'
response1 = response_agent.invoke(
    {"messages": [{"role": "user", "content": "Jim is my friend"}]},
    config=config
)

for m in response1["messages"]:
    m.pretty_print()
#output:
# ================================ Human Message =================================

# Jim is my friend
# ================================== Ai Message ==================================
# Tool Calls:
#   manage_memory (call_DuJNWNlYi1qJfgi2vIRLxTLJ)
#  Call ID: call_DuJNWNlYi1qJfgi2vIRLxTLJ
#   Args:
#     content: Jim is John's friend.
#     action: create
# ================================= Tool Message =================================
# Name: manage_memory

# created memory a6b9e4c5-2db7-49c9-98f7-a7eaa2534442
# ================================== Ai Message ==================================

# I've noted that Jim is your friend. Let me know if there's anything else you'd like to add or manage!
#### Verificando 'search_memory'
response2 = response_agent.invoke(
    {"messages": [{"role": "user", "content": "who is jim?"}]},
    config=config
)

for m in response2["messages"]:
    m.pretty_print()
#Output:
# ================================ Human Message =================================

# Jim is my friend
# ================================== Ai Message ==================================
# Tool Calls:
#   manage_memory (call_DuJNWNlYi1qJfgi2vIRLxTLJ)
#  Call ID: call_DuJNWNlYi1qJfgi2vIRLxTLJ
#   Args:
#     content: Jim is John's friend.
#     action: create
# ================================= Tool Message =================================
# Name: manage_memory

# created memory a6b9e4c5-2db7-49c9-98f7-a7eaa2534442
# ================================== Ai Message ==================================

# I've noted that Jim is your friend. Let me know if there's anything else you'd like to add or manage!
# ================================ Human Message =================================

# who is jim?
# ================================== Ai Message ==================================
# Tool Calls:
#   search_memory (call_p5NHrQqBU0UzXqNU1XvylOTM)
#  Call ID: call_p5NHrQqBU0UzXqNU1XvylOTM
#   Args:
#     query: Jim
# ================================= Tool Message =================================
# Name: search_memory

# [{"namespace": ["email_assistant", "lance", "collection"], "key": "a6b9e4c5-2db7-49c9-98f7-a7eaa2534442", "value": {"content": "Jim is John's friend."}, "created_at": "2025-10-30T18:40:06.053646+00:00", "updated_at": "2025-10-30T18:40:06.053653+00:00", "score": 0.4343276126880873}]
# ================================== Ai Message ==================================

# Jim is your friend. If you need more specific details or context about Jim, feel free to let me know!











##### Explorando mem: corto- y largo- plazo
#### "StateSnapshot" mem corto-plazo
state = response_agent.get_state(config)
print(state)
for msg in state.values['messages']:
    print(type(msg))
    print(msg)
    print('\n\n')
#### Mem largo-plazo
store.list_namespaces()
    #output: [('email_assistant', 'lance', 'collection')]

store.search(('email_assistant', 'lance', 'collection'))
    #Output:
    #[Item(namespace=['email_assistant', 'lance', 'collection'], key='a6b9e4c5-2db7-49c9-98f7-a7eaa2534442', value={'content': "Jim is John's friend."}, created_at='2025-10-30T18:40:06.053646+00:00', updated_at='2025-10-30T18:40:06.053653+00:00', score=None)]

store.search(('email_assistant', 'lance', 'collection'), query="jim")
    #Output:
    # [Item(namespace=['email_assistant', 'lance', 'collection'], key='a6b9e4c5-2db7-49c9-98f7-a7eaa2534442', value={'content': "Jim is John's friend."}, created_at='2025-10-30T18:40:06.053646+00:00', updated_at='2025-10-30T18:40:06.053653+00:00', score=0.553310811429239)]
        #Note: last attr: "score" = 0.5533108...









##### Creando MAS-grafo con in-mem persistence ReAct
    #OJO: es importante epxlorar los persistent mem de
            # assitente y ReAct agente, aparte
        # El ReAct esta dentro del assitente, cual es la relacion
                # que se manifsta entre nested agentes
#### Estado
class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]


### Create node fcnality for router simple-agent
def triage_router(state: State) -> Command[
    Literal["response_agent", "__end__"]
]:
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples=None
    )
    user_prompt = triage_user_prompt.format(
        author=author, 
        to=to, 
        subject=subject, 
        email_thread=email_thread
    )
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ]
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update = None
        goto = END
    elif result.classification == "notify":
        # If real life, this would do something else
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)

### Assembling email assistnat
email_agent = StateGraph(State)
email_agent = email_agent.add_node(triage_router)
email_agent = email_agent.add_node("response_agent", response_agent)
email_agent = email_agent.add_edge(START, "triage_router")
email_agent1 = email_agent.compile(store=store, checkpointer=snapshot_store)

### Visuale email assistant graph
display(Image(email_agent1.get_graph(xray=True).draw_mermaid_png()))



































#########################################################################
#########################################################################
    # Semantic + Episodic Mem #
import os
from dotenv import load_dotenv
_ = load_dotenv()
import uuid
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model
from prompts import triage_user_prompt
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from IPython.display import Image, display
from langgraph.graph import add_messages
from langchain_core.tools import tool
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent




##### Creando train data para episodic mem.
#### Input
email1 = { # SAme as "email" above
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

    I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

    Specifically, I'm looking at:
    - /auth/refresh
    - /auth/validate

    Thanks!
    Alice"""
}

email2 = {
    "author": "Sarah Chen <sarah.chen@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Update: Backend API Changes Deployed to Staging",
    "email_thread": """Hi John,

    Just wanted to let you know that I've deployed the new authentication endpoints we discussed to the staging environment. Key changes include:

    - Implemented JWT refresh token rotation
    - Added rate limiting for login attempts
    - Updated API documentation with new endpoints

    All tests are passing and the changes are ready for review. You can test it out at staging-api.company.com/auth/*

    No immediate action needed from your side - just keeping you in the loop since this affects the systems you're working on.

    Best regards,
    Sarah
    """
}
#### Label
data1 = {
    "email": email1,
    "label": "respond"
}

data2 = {
    "email": email2,
    "label": "ignore"
}
#### Almacenando ejemplos
store.put(
    ("email_assistant", "lance", "examples"), 
    str(uuid.uuid4()), 
    data1
)

store.put(
    ("email_assistant", "lance", "examples"), 
    str(uuid.uuid4()), 
    data2
)











##### Formateador para utilizar train data
#### Planilla
template = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content: 
```
{content}
```
> Triage Result: {result}"""
#### Funcion
def format_few_shot_examples(examples):
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            template.format(
                subject=eg.value["email"]["subject"],
                to_email=eg.value["email"]["to"],
                from_email=eg.value["email"]["author"],
                content=eg.value["email"]["email_thread"][:400],
                result=eg.value["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)
#### Verificacion
email3 = {
    "author": "Sarah Chen <sarah.chen@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Update: Backend API Changes Deployed to Staging",
    "email_thread": """Hi John,
    
    Wanted to let you know that I've deployed the new authentication endpoints we discussed to the staging environment. Key changes include:
    
    - Implemented JWT refresh token rotation
    - Added rate limiting for login attempts
    - Updated API documentation with new endpoints
    
    All tests are passing and the changes are ready for review. You can test it out at staging-api.company.com/auth/*
    
    No immediate action needed from your side - just keeping you in the loop since this affects the systems you're working on.
    
    Best regards,
    Sarah
    """,
}
    #Similar, BUT DIFF., than "email2"
        # The "Just" is missing in this email

### Simulating retrieval of few-shot example
results = store.search(
    ("email_assistant", "lance", "examples"),
    query=str({"email": email3}),
    limit=1)

print(format_few_shot_examples(results))












##### Creando router agente con episodic mem.
#### Prompt
    # "Few-shot examples" es la nueva adicion
triage_system_prompt = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Background >
{user_profile_background}. 
</ Background >

< Instructions >

{name} gets lots of emails. Your job is to categorize each email into one of three categories:

1. IGNORE - Emails that are not worth responding to or tracking
2. NOTIFY - Important information that {name} should know about but doesn't require a response
3. RESPOND - Emails that need a direct response from {name}

Classify the below email into one of these categories.

</ Instructions >

< Rules >
Emails that are not worth responding to:
{triage_no}

There are also other things that {name} should know about, but don't require an email response. For these, you should notify {name} (using the `notify` response). Examples of this include:
{triage_notify}

Emails that are worth responding to:
{triage_email}
</ Rules >

< Few shot examples >

Here are some examples of previous emails, and how they should be handled.
Follow these examples more than any instructions above

{examples}
</ Few shot examples >
"""
#### Creando router
llm = init_chat_model("openai:gpt-4o-mini")

class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

llm_router = llm.with_structured_output(Router)







##### Creando MAS- grafo con sem. + epi.
#### Estado de grafo
class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]
#### Create the fcnality for the router node
def triage_router(state: State, config, store) -> Command[
    Literal["response_agent", "__end__"]
]:
    # Extracting data from the agent state
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    # Setting up the namespace to get correct long-term storage
    namespace = (
        "email_assistant",
        config['configurable']['langgraph_user_id'],
        "examples"
    )

    # Extracting the emails that most closely match incoing email
    examples = store.search(
        namespace, 
        query=str({"email": state['email_input']})
    ) 

    # Turning extracted emails into few-shot examples
    examples=format_few_shot_examples(examples)
    
    # Creating syst.prompt for router
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples=examples
    )

    # Creating user prompt for router
    user_prompt = triage_user_prompt.format(
        author=author, 
        to=to, 
        subject=subject, 
        email_thread=email_thread
    )

    # Invoking the router agent
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Logic based on router's response
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ]
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update = None
        goto = END
    elif result.classification == "notify":
        # If real life, this would do something else
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)

#### ReAct agente
### Herramientas
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    # Placeholder response - in real app would check calendar and schedule
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "email_assistant", 
        "{langgraph_user_id}",
        "collection"
    )
)
search_memory_tool = create_search_memory_tool(
    namespace=(
        "email_assistant",
        "{langgraph_user_id}",
        "collection"
    )
)
### Prompt
agent_system_prompt_memory = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Tools >
You have access to the following tools to help manage {name}'s communications and schedule:

1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings
3. check_calendar_availability(day) - Check available time slots for a given day
4. manage_memory - Store any relevant information about contacts, actions, discussion, etc. in memory for future reference
5. search_memory - Search for any relevant information that may have been stored in memory
</ Tools >

< Instructions >
{instructions}
</ Instructions >
"""

def create_prompt(state):
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt_instructions["agent_instructions"], 
                **profile
            )
        }
    ] + state['messages']
### Creando ReAct
tools= [
    write_email, 
    schedule_meeting,
    check_calendar_availability,
    manage_memory_tool,
    search_memory_tool
]
response_agent = create_react_agent(
    "openai:gpt-4o",
    tools=tools,
    prompt=create_prompt,
    # Use this to ensure the store is passed to the agent 
    store=store,
    # checkpointer=checkptr
        #Won't use bc it's not the focus of this section
)
#### Creando MAS- grafo
#### Creating email assistant MAS-agent
email_agent = StateGraph(State)
email_agent = email_agent.add_node(triage_router)
email_agent = email_agent.add_node("response_agent", response_agent)
email_agent = email_agent.add_edge(START, "triage_router")
email_agent = email_agent.compile(
    store=store,
    # checkpointer=checkptr
        #Omitted bc it's not the focus of this section
)























#####################################################################
#####################################################################
    # Sem. + Epi. + Procedural #
import os
from dotenv import load_dotenv
_ = load_dotenv()
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model
from prompts import triage_user_prompt
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from IPython.display import Image, display

from langgraph.graph import add_messages
from langchain_core.tools import tool
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent
from langmem import create_multi_prompt_optimizer
import json



##### Semantic + Episodic
#### Setup
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}

email = {
    "from": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "body": """
Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}
#### Semantic: long-term y short-term 
### Long-term
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"} #embed model
)
### Short-term
checkptr = InMemorySaver()
### ReAct agent
## Tools
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    # Placeholder response - in real app would check calendar and schedule
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "email_assistant", 
        "{langgraph_user_id}",
        "collection"
    )
)

search_memory_tool = create_search_memory_tool(
    namespace=(
        "email_assistant",
        "{langgraph_user_id}",
        "collection"
    )
)
## Prompt
agent_system_prompt_memory = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Tools >
You have access to the following tools to help manage {name}'s communications and schedule:

1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings
3. check_calendar_availability(day) - Check available time slots for a given day
4. manage_memory - Store any relevant information about contacts, actions, discussion, etc. in memory for future reference
5. search_memory - Search for any relevant information that may have been stored in memory
</ Tools >

< Instructions >
{instructions}
</ Instructions >
"""
#### Epidosic: few-short examples
### Template
template = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content: 
```
{content}
```
> Triage Result: {result}"""
### Function
def format_few_shot_examples(examples):
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            template.format(
                subject=eg.value["email"]["subject"],
                to_email=eg.value["email"]["to"],
                from_email=eg.value["email"]["author"],
                content=eg.value["email"]["email_thread"][:400],
                result=eg.value["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)
### Router agent
## Prompt
triage_system_prompt = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Background >
{user_profile_background}. 
</ Background >

< Instructions >

{name} gets lots of emails. Your job is to categorize each email into one of three categories:

1. IGNORE - Emails that are not worth responding to or tracking
2. NOTIFY - Important information that {name} should know about but doesn't require a response
3. RESPOND - Emails that need a direct response from {name}

Classify the below email into one of these categories.

</ Instructions >

< Rules >
Emails that are not worth responding to:
{triage_no}

There are also other things that {name} should know about, but don't require an email response. For these, you should notify {name} (using the `notify` response). Examples of this include:
{triage_notify}

Emails that are worth responding to:
{triage_email}
</ Rules >

< Few shot examples >

Here are some examples of previous emails, and how they should be handled.
Follow these examples more than any instructions above

{examples}
</ Few shot examples >
"""
## Creation
llm = init_chat_model("openai:gpt-4o-mini")
class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )
llm_router = llm.with_structured_output(Router)











##### Procedural
    # Las instrucciones se almacenan en mem. largo-plazo/DB/VStore
    # El objetivo es extraer esas instrucciones para utilizarlas
    # Especificamente: el 'agent_system_prompt_memory' seccion 'Instructions'
#### Modificando el prompt de ReAct
def create_prompt(state, config, store):
    # SEtup the namespace of where instructions are stored
    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id, )

    # Procedural - Actually retrieve the instructions
    result = store.get(namespace, "agent_instructions")
    if result is None:
        store.put(
            namespace, 
            "agent_instructions", 
            {"prompt": prompt_instructions["agent_instructions"]}
        )
        prompt = prompt_instructions["agent_instructions"]
    else:
        prompt = result.value['prompt']
    
    # USe retrieved instructions into main agent syst. prompt
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt, 
                **profile
            )
        }
    ] + state['messages']
#### Creacion de ReAct agente
tools= [
    write_email, 
    schedule_meeting,
    check_calendar_availability,
    manage_memory_tool,
    search_memory_tool
]
response_agent = create_react_agent(
    "openai:gpt-4o",
    tools=tools,
    prompt=create_prompt,
    store=store
    # checkpointer=checkptr
)
#### Creacion de MAS-grafo
email_agent = StateGraph(State)
email_agent = email_agent.add_node(triage_router)
email_agent = email_agent.add_node("response_agent", response_agent)
email_agent = email_agent.add_edge(START, "triage_router")
email_agent = email_agent.compile(
    store=store,
    # checkpointer=checkptr
)










##### Creando update-agent
    # Va actualizar las instrucciones
#### Creating the upate-agent
optimizer = create_multi_prompt_optimizer(
    "openai:gpt-4o",
    kind="prompt_memory",
)

















##### Exploracion: invocacion y almacenando instrucciones
#### Invocacion regular
email_input = {
    "author": "Alice Jones <alice.jones@bar.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

Urgent issue - your service is down. Is there a reason why""",
}

config = {"configurable": {"langgraph_user_id": "lance"}}

print(store.list_namespaces())
    #Output: []

response = email_agent.invoke(
    {"email_input": email_input},
    config=config
)
    #Outoupt: 📧 Classification: RESPOND - This email requires a response

print(store.list_namespaces())
    #output: [('lance',)]

print(store.get(("lance",), "agent_instructions").value['prompt'])
print('\n\n')
print(store.get(("lance",), "triage_respond").value['prompt'])
print('\n\n')
print(store.get(("lance",), "triage_ignore").value['prompt'])
print('\n\n')
print(store.get(("lance",), "triage_notify").value['prompt'])
#output
# Use these tools when appropriate to help manage John's tasks efficiently.



# Direct questions from team members, meeting requests, critical bug reports



# Marketing newsletters, spam emails, mass company announcements



# Team member out sick, build system notifications, project status updates
for m in response["messages"]:
    m.pretty_print()












##### Actualizando las insrtucciones
#### Ejemplo 1
### Gathering the other agent's execution and appending user feedback
conversations = [
    (
        response['messages'],
        "Always sign your emails `John Doe`"
    )
]

### Creating the update-prompts for update-agent
prompts = [
    {
        "name": "main_agent",
        "prompt": store.get(("lance",), "agent_instructions").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on how the agent should write emails or schedule events"
        
    },
    {
        "name": "triage-ignore", 
        "prompt": store.get(("lance",), "triage_ignore").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be ignored"

    },
    {
        "name": "triage-notify", 
        "prompt": store.get(("lance",), "triage_notify").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails the user should be notified of"

    },
    {
        "name": "triage-respond", 
        "prompt": store.get(("lance",), "triage_respond").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be responded to"

    },
]
### Invoking the update-agent
updated = optimizer.invoke(
    {"trajectories": conversations, "prompts": prompts}
)
### Explorando actualizaciones
print(updated)
print(json.dumps(updated, indent=4))
### Updating instructions with update-agent output
for i, updated_prompt in enumerate(updated):
    old_prompt = prompts[i]
    if updated_prompt['prompt'] != old_prompt['prompt']:
        name = old_prompt['name']
        print(f"updated {name}")
        if name == "main_agent":
            store.put(
                ("lance",),
                "agent_instructions",
                {"prompt":updated_prompt['prompt']}
            )
        else:
            #raise ValueError
            print(f"Encountered {name}, implement the remaining stores!")
### Confirming instructions were updated
print(store.get(("lance",), "agent_instructions").value['prompt'])
### Retry email sample with updated instructions
response = email_agent.invoke(
    {"email_input": email_input}, 
    config=config
)
    #output: 📧 Classification: RESPOND - This email requires a response

for m in response["messages"]:
    m.pretty_print()

#### Ejemplo 2
email_input = {
    "author": "Alice Jones <alice.jones@bar.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

Urgent issue - your service is down. Is there a reason why""",
}

response = email_agent.invoke(
    {"email_input": email_input},
    config=config
)
    #outout: 📧 Classification: RESPOND - This email requires a response

conversations = [
    (
        response['messages'],
        "Ignore any emails from Alice Jones"
    )
]

prompts = [
    {
        "name": "main_agent",
        "prompt": store.get(("lance",), "agent_instructions").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on how the agent should write emails or schedule events"
        
    },
    {
        "name": "triage-ignore", 
        "prompt": store.get(("lance",), "triage_ignore").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be ignored"

    },
    {
        "name": "triage-notify", 
        "prompt": store.get(("lance",), "triage_notify").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails the user should be notified of"

    },
    {
        "name": "triage-respond", 
        "prompt": store.get(("lance",), "triage_respond").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be responded to"

    },
]

updated = optimizer.invoke(
    {"trajectories": conversations, "prompts": prompts}
)

for i, updated_prompt in enumerate(updated):
    old_prompt = prompts[i]
    if updated_prompt['prompt'] != old_prompt['prompt']:
        name = old_prompt['name']
        print(f"updated {name}")
        if name == "main_agent":
            store.put(
                ("lance",),
                "agent_instructions",
                {"prompt":updated_prompt['prompt']}
            )
        if name == "triage-ignore":
            store.put(
                ("lance",),
                "triage_ignore",
                {"prompt":updated_prompt['prompt']}
            )
        else:
            #raise ValueError
            print(f"Encountered {name}, implement the remaining stores!")

response = email_agent.invoke(
    {"email_input": email_input},
    config=config
)
    #Output: 🚫 Classification: IGNORE - This email can be safely ignored

store.get(("lance",), "triage_ignore").value['prompt']









