"""
Este archivo servira como la base para construir agentes en general

Recalque:
json.dumps() -> (JSON-)str

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