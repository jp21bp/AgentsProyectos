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
### C6 ###
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