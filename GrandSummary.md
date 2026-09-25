# Grand Summary of Agents Courses

This file will accumulate all the important, code-independent concepts across all the courses.

It'll mainly consist of concepts and ideas that can be applied to the design and creation of MAS.

# Table of Contents
## LangGraph Courses
### Course 6: Functions, Tools, and Agents using LangChain
* [Creating fallbacks](#creating-fallbacks)
* [Tagging and extraction definitions](#tagging-and-extraction-definitions)
* [Real world example - tagging an article/blog](#real-world-example---tagging-an-articleblog)
* [REal world example - extracting academic papers mentioned in a blog](#real-world-example---extracting-academic-papers-mentioned-in-a-blog)
* [Create a function that flatters a list of lists into a single, 1D list](#create-a-function-that-flatters-a-list-of-lists-into-a-single-1d-list)
* [Tools and Routing definitons:](#tools-and-routing-definitons)
* [Pydantic details:](#pydantic-details)
* [Open API Specs](#open-api-specs)
* [Analysis on LLMs and tool/fcn usage:](#analysis-on-llms-and-toolfcn-usage)
* [Chain that AUTOMATICALLY performs function call](#chain-that-automatically-performs-function-call)
* [There are 8 general steps to create basic OpenAI fcn call agent:](#there-are-8-general-steps-to-create-basic-openai-fcn-call-agent)
* [Incorporating chat-history basic short-term memory usage](#incorporating-chat-history-basic-short-term-memory-usage)
* [There are 7 steps to create an action-memory chain/agent (using a loop):](#there-are-7-steps-to-create-an-action-memory-chainagent-using-a-loop)
* [Enabling long-term memory](#enabling-long-term-memory)


### Course 7: AI Agents in LangGraph
* [Key design patterns of agentic workflow](#key-design-patterns-of-agentic-workflow)
* [ReAct Details:](#react-details)
* [ReAct agent with few shot examples](#react-agent-with-few-shot-examples)
* [Understanding "Annotated" type hint, 'operator' library, and "AnyMessage" type:](#understanding-annotated-type-hint-operator-library-and-anymessage-type)
* [5 steps in an agentic search:](#5-steps-in-an-agentic-search)
* [Scraping content from a given URL](#scraping-content-from-a-given-url)
* [Difference between regular search and agentic search](#difference-between-regular-search-and-agentic-search)
* [Defns: persistence and streaming](#defns-persistence-and-streaming)
* [Cerating (in-memory) persistence](#cerating-in-memory-persistence)
* [Streaming tokens, async methods, and async checkpointers - details:](#streaming-tokens-async-methods-and-async-checkpointers---details)
* [HITL changes to agent state and its attributes](#hitl-changes-to-agent-state-and-its-attributes)
* [Incorporating HITL in the agent class with interrupts - details:](#incorporating-hitl-in-the-agent-class-with-interrupts---details)
* [Debugging: creating mock LLM responses by targeting specific "Message"s in a snapshot ](#debugging-creating-mock-llm-responses-by-targeting-specific-messages-in-a-snapshot)
* [Node functionality's order details:](#node-functionalitys-order-details)
* [Steps to create an essay writer:](#steps-to-create-an-essay-writer)
* [Essay writer details](#essay-writer-details)
* [Helpful LangChain/ LangGraph resources:](#helpful-langchain-langgraph-resources)
* [Agent flows/architectures supported by LG:](#agent-flowsarchitectures-supported-by-lg)


### Course 8: Long Term Agentic Memory
* [Simple-agent vs state-agent vs MAS-agent - IMPORTANT:](#simple-agent-vs-state-agent-vs-mas-agent---important)
* [Concepts to consider when adding mem. to agents:](#concepts-to-consider-when-adding-mem-to-agents)
* [3 types of memories:](#3-types-of-memories)
* [There are 2 mechanisms for updating/saving mem.](#there-are-2-mechanisms-for-updatingsaving-mem)
* [Steps to create an email assistant agent](#steps-to-create-an-email-assistant-agent)
* [Guided Qs when adding long-term mem.:](#guided-qs-when-adding-long-term-mem)
* [Important info on 3 types of prompt - general vs ambigious vs specific prompts:](#important-info-on-3-types-of-prompt---general-vs-ambigious-vs-specific-prompts)
* [Exploring diff. types of mem. - short term vs long term:](#exploring-diff-types-of-mem---short-term-vs-long-term)
* [Rule of thumb for short-term and long-term mem.](#rule-of-thumb-for-short-term-and-long-term-mem)
* [Gameplan to intergrate semantic mem.](#gameplan-to-intergrate-semantic-mem)
* [Exploring nested dynamic b/w "email_assistant" and "response_agent"](#exploring-nested-dynamic-bw-email_assistant-and-response_agent)
* [Note on in-hot-path update mechanism:](#note-on-in-hot-path-update-mechanism)
* [Episodic mem. overview:](#episodic-mem-overview)
* [Integrating episodic mem. to agents:](#integrating-episodic-mem-to-agents)
* [Storing few-shots in long-term mem. ](#storing-few-shots-in-long-term-mem)
* [Storing the coupled few-shot exs in long-term mem. - important format:](#storing-the-coupled-few-shot-exs-in-long-term-mem---important-format)
* [Understanding the long-term storage's namespace and interaction with "langmem"](#understanding-the-long-term-storages-namespace-and-interaction-with-langmem)
* [Relationship b/w namespace and agent behavior](#relationship-bw-namespace-and-agent-behavior)
* [Relationship b/w few-shot exs and namespaces - details:](#relationship-bw-few-shot-exs-and-namespaces---details)
* [Note on background-update mechanism through diff process:](#note-on-background-update-mechanism-through-diff-process)
* [Procedural mem overview:](#procedural-mem-overview)
* [Background-update mechanism using a diff. agent -details:](#background-update-mechanism-using-a-diff-agent--details)
* [Implementing procedural mem. gameplan:](#implementing-procedural-mem-gameplan)
* [Update agent details:](#update-agent-details)
* [Invoking an update agent:](#invoking-an-update-agent)


## CrewAI Courses
### Course 9: Multi-AI Agent Systems
* [Multiple strings vs triple quote docstring:](#multiple-strings-vs-triple-quote-docstring)
* [MAS overview:](#mas-overview)
* [6 elements that make a great agent - IMPORTANT](#6-elements-that-make-a-great-agent---important)
* [3 types of memory for CrewAI:](#3-types-of-memory-for-crewai)
* [MAS important dynamics:](#mas-important-dynamics)
* [Relationship b/w real life manager and MAS - guided Qs:](#relationship-bw-real-life-manager-and-mas---guided-qs)
 * IMPORTANT Steps to creating great agents - consider the:
* [3 key elements of a great tool:](#3-key-elements-of-a-great-tool)
* [Manager framework when building agents - guided Qs:](#manager-framework-when-building-agents---guided-qs)
* [3 key elements in a great task:](#3-key-elements-in-a-great-task)
* [3 types of agent collaboration PROCESSES:](#3-types-of-agent-collaboration-processes)
* [Agent collaboration and "crewai.Progress" details](#agent-collaboration-and-crewaiprogress-details)


### Course 10: Practical Multi-AI Agents w/ Advanced Use Cases
* [General tasks agents can do:](#general-tasks-agents-can-do)
* [YAML files:](#yaml-files)
* [MAS and tools](#mas-and-tools)
* [Creating complex MASs](#creating-complex-mass)
* [Explicitly creating custom collab. orchestration:](#explicitly-creating-custom-collab-orchestration)
* [2 main components that impact MAS's performance:](#2-main-components-that-impact-mass-performance)
* [Quantifying MAS's performance](#quantifying-mass-performance)
* [Agents and MAS challenges:](#agents-and-mas-challenges)





## Course 11: Building Code Agents w/ smolagents
* [Definitions of diff types of agents:](#definitions-of-diff-types-of-agents)
* [5 levels of AI agency:](#5-levels-of-ai-agency)
* [Benefits of code agent actions over tool-calling agent actions:](#benefits-of-code-agent-actions-over-tool-calling-agent-actions)
* [Note on dataframe:](#note-on-dataframe)
* [Secure code execution overview:](#secure-code-execution-overview)
* [Custom, safeguard executor needs to be IMPORTED](#custom-safeguard-executor-needs-to-be-imported)
* [There are 3 main rules that are built-in safeguards](#there-are-3-main-rules-that-are-built-in-safeguards)
* [Sandbox details:](#sandbox-details)
* [Smolagent allows you to use one of the 2 following options:](#smolagent-allows-you-to-use-one-of-the-2-following-options)
* [Steps to run agent in a sandbox:](#steps-to-run-agent-in-a-sandbox)
* [You can setup tracing in a local and remote manner, using "register()"](#you-can-setup-tracing-in-a-local-and-remote-manner-using-register)
* [Diff. ways to monitor agents:](#diff-ways-to-monitor-agents)
* [Notes on creating tools for code agents:](#notes-on-creating-tools-for-code-agents)
* [Steps to get a smolagent's state history (in dataframe format)](#steps-to-get-a-smolagents-state-history-in-dataframe-format)
* [Multiple snapshot that use same tool for same request - details](#multiple-snapshot-that-use-same-tool-for-same-request---details)
* [Steps to evaluating a code agent](#steps-to-evaluating-a-code-agent)
* [smolagents and fixing errors:](#smolagents-and-fixing-errors)
* [BEnefits of MAS:](#benefits-of-mas)



## Course 12: Model Context Protocol w/ Anthropic
* [MCP details:](#mcp-details)
* [MCPS vs tool/API calling:](#mcps-vs-toolapi-calling)
* [MCP client-server architecture:](#mcp-client-server-architecture)
* [General terminology:](#general-terminology)
* [MCPSs expose the following:](#mcpss-expose-the-following)
* [Overview in creating tools, resources, and prompts in MCP:](#overview-in-creating-tools-resources-and-prompts-in-mcp)
* [Steps in the communication lifecycle b/w MCPC and MCPS:](#steps-in-the-communication-lifecycle-bw-mcpc-and-mcps)
* [MCP Transports (MCPT) details:](#mcp-transports-mcpt-details)
* [Details of the 3 common, built-in MCPTs:](#details-of-the-3-common-built-in-mcpts)
* [In reference to tools, there are 2 main MCPC requests to the MCPS:](#in-reference-to-tools-there-are-2-main-mcpc-requests-to-the-mcps)
* [There are 2 ways to create a MCPS:](#there-are-2-ways-to-create-a-mcps)
* [Steps to setup environment to test MCPS- working with "uv":](#steps-to-setup-environment-to-test-mcps--working-with-uv)
* [MCPInspector details:](#mcpinspector-details)
* [Steps to use MCPInspector:](#steps-to-use-mcpinspector)
* [Steps to connect MCPC with MCPS:](#steps-to-connect-mcpc-with-mcps)
* [Relationship b/w MCPC and LLM-app file:](#relationship-bw-mcpc-and-llm-app-file)
* [MCPC details:](#mcp-details)
* [Steps to run MCPC from terminal:](#steps-to-run-mcpc-from-terminal)
* [Remote MCPS details:](#remote-mcps-details)
* [Connecting to multiple MCPS - overview](#connecting-to-multiple-mcps---overview)
* ["uv" vs "uvx" vs "npx":](#uv-vs-uvx-vs-npx)
* [Sample JSON file that lists ALL MCPSs you want to connect to:](#sample-json-file-that-lists-all-mcpss-you-want-to-connect-to)
* [Resources in MCPSs - details:](#resources-in-mcpss---details)
* [Requests associated with resources and prompts:](#requests-associated-with-resources-and-prompts)
* [Local MCPS vs Remote MCPS - code changes:](#local-mcps-vs-remote-mcps---code-changes)
* [Steps to deploy MCPS using render.com; how to start git repo:](#steps-to-deploy-mcps-using-rendercom-how-to-start-git-repo)
* [Updating MCPC to work with remote MCPSs:](#updating-mcpc-to-work-with-remote-mcpss)



## Course 13: Agent Communication Protocol
* [ACP details](#acp-details)
* [ACP benefits:](#acp-benefits)
* [Steps to build ACP compliant agents:](#steps-to-build-acp-compliant-agents)
* [4 patterns made possible by ACP](#4-patterns-made-possible-by-acp)
* [AI tech stack (from bottom layer to top layer):](#ai-tech-stack-from-bottom-layer-to-top-layer)
* [Steps to setup ACP communication:](#steps-to-setup-acp-communication)
* [Steps in an ACP agent's lifecycle:](#steps-in-an-acp-agents-lifecycle)
* [RAG overview:](#rag-overview)
* [Needed libraries to convert code to ACPS:](#needed-libraries-to-convert-code-to-acps)
* [ACPS details](#acps-details)
* ["AsynchGenerator\[RunYield,RunYieldResume\]" details](#asynchgeneratorrunyieldrunyieldresume-details)
* [Steps to run ACPS locally](#steps-to-run-acps-locally)
* ["Message" and "MessagePart" details:](#message-and-messagepart-details)
* [Steps to start a project with "uv"](#steps-to-start-a-project-with-uv)
* [ACPS pre-reqs to running an ACPC:](#acps-pre-reqs-to-running-an-acpc)
* [Steps to run an ACPC:](#steps-to-run-an-acpc)
* [Relationship b/w ACPC, ACPS, and multiple agents:](#relationship-bw-acpc-acps-and-multiple-agents)
* [Reasons to create diff ACPSs:](#reasons-to-create-diff-acpss)
* [ACPS common libraries imported:](#acps-common-libraries-imported)
* [GEneral communication b/w ACPC and ACPS of diff frameworks - IMPORTANT:](#general-communication-bw-acpc-and-acps-of-diff-frameworks---important)
* [Sequentiall calling overview:](#sequentiall-calling-overview)
* [Sequential calling and ACPC:](#sequential-calling-and-acpc)
* [Steps to perform sequential agent chaining from ACPC's main code:](#steps-to-perform-sequential-agent-chaining-from-acpcs-main-code)
* [Passing in MULTIPLE prompts to SAME agent in one ACPC call:](#passing-in-multiple-prompts-to-same-agent-in-one-acpc-call)
* [Hierarchical chaining overview:](#hierarchical-chaining-overview)
* [Libraries needed for hierarchical chaining - "AgentCollection" and "ACPCallingAgent":](#libraries-needed-for-hierarchical-chaining---agentcollection-and-acpcallingagent)
* [Steps to perform hierarchical workflows:](#steps-to-perform-hierarchical-workflows)
* [Integrating MCP overview:](#integrating-mcp-overview)
* [Libraries needed to integrate MCP into ACP:](#libraries-needed-to-integrate-mcp-into-acp)
* ["smolagents.ToolCollection" details:](#smolagentstoolcollection-details)



## Course 17: Pydantic for LLM Workflows
* [Pydantic model basics:](#pydantic-model-basics)
* [Steps to integrate Pydantic models w/ tool-calling:](#steps-to-integrate-pydantic-models-w-tool-calling)
* [PYdantic basics -FULL OF GOOD INFO:](#pydantic-basics--full-of-good-info)
* [Relationship b/w JSON and Python:](#relationship-bw-json-and-python)
* [Pydantic code info - IMPORTANT:](#pydantic-code-info---important)
* [Steps to implement TRADITIONAL LLM-output validation:](#steps-to-implement-traditional-llm-output-validation)
* [Fcns to analyze COMPLEX data types - IMPORTANT:](#fcns-to-analyze-complex-data-types---important)
* [Pydantic models and tools](#pydantic-models-and-tools)
* [Validating complex fields/attrs:](#validating-complex-fieldsattrs)





# Detailed Notes
## LangGraph Courses
### Course 6: Functions, Tools, and Agents using LangChain

#### Creating fallbacks
* Used to fallback to a backup chain when the main chain fails
* There can be a LIST of different backup chains, where backup order follows list order
 
#### Tagging and extraction definitions
* Tagging = create a structured DESCRIPTION from unstructured text
* Extraction = extracting specific ENTITIES from unstructured text
 
#### Real world example - tagging an article/blog
* Consists of loading a blog post and tagging info from a subset of text
 
#### Real world example - extracting academic papers mentioned in a blog
 
#### Create a function that flatters a list of lists into a single, 1D list
 
#### Tools and Routing definitons:
* Tools = Fcns/services LLM can use to extend its capabilities
* Routing = selecting from multiple possible tools
 
#### Pydantic details:
* Pydantic classes create more explicit struct
  - USed by LLM to determine what input should be
* Specifically, the pydantic class is used to add descriptions to the tool's params
  - But the tool's logic still needs to be implemented separately from the pydantic
 
#### Open API Specs
* Open API specs = API specifications for their inputs and outputs
* There are regular user-fcns that we want to interact with BUT are exposed behind APIs
  - Thus, there is a need for ease of conversion 
 
#### Analysis on LLMs and tool/fcn usage:
* There are 2 outputs for an LLM on making a tool/fcn decision:
  1. LLM decides to NOT use ANY tools/fcn
      - Main interest: value of "content" in the "AIMessage" output
  2. LLM decides to use a fcn (doesn't matter which one)
      - Main interest: choosen tool and its corresponding "arguments"
      - For "arguments", convert from JSON blob into readible dictionary
        * Done with output parser components
 
#### Chain that AUTOMATICALLY performs function call
  * Recall: up to this point the chain would provide which function call to use
    - But actually calling the function needed to be done manually
  * With "AgentActionMessageLog.tool" and ".tool_input" this can be automatize
  * This requires creating a router that performs different logic based on whether:
    - It encounters "AgentActionMessageLog" or "AgentFinish"

#### There are 8 general steps to create basic OpenAI fcn call agent:
1. Determine the SPECIFIC task you want to solve
2. Create a fcn to solve that SPECIFIC task
3. Turn those fcns into tools
4. Turn those tools into OpenAi fcns
5. Bind those OpenAI fcns into a model
6. Create a prompt template (for more flexibility)
7. Choose appropriate output parser (based on SPECIFIC task)
8. Create a chain (which will be the agent)
 
#### Incorporating chat-history basic short-term memory usage
* Note: this will ONLY store the action history of CURRENT invocation
  - I.e., future invocations will NOT remember previous invocations' actions
    * To enable this, use long-term memory, which is explored further below
* AT this point, the LLM is only aware of it's current actions on current query
  - But it does NOT remember past actions on the same query
* In order to integrate past action awareness, it will require a modifying the chain prompt
  - Specifically, the prompt will need to pass back in ALL the action history of:
    * The tools that were called
    * The results of calling the tool
    * The LLM's response to having the tool's result fed in as context
    * The user's queries
  - This information will be passed using the "Messages" data type found in LC
    * Some "Message" types: "HumanMessage", "AIMessage", "ToolMessage", ...
  - This history of actions will be stored in "MessagesPlaceholder"
    * This is a prompt setup parameter
    * NOte, this same param will be used to set up long-term mem, further below
  - NOTE: THIS SET UP HAS A LOT OF FLEXIBILITY
    * I.e., it can be molded and folded depending on the specific task at hand 
* In order to properly set up the memory, a loop will need to be implemented
  - This way the user can keep chatting with the LLM in a seemesless way
* Passing in previous action history into future LLM actions of SAME query
  - Requires passing in "(result1, observation)" tuple
    * This encompasses the history of the first LLM call
  - Recall: "AgentActionsMesasgeLog" has ".message_log" 
    * ".message_log" = a LIST of msgs that show how we ARRIVED at current action
  - "format_to_openai_functions" will be required 
    * Used to convert "(result1, observation)" tuple into open_ai json format
    * REmember, whenever passing in details to LLM invocation, use proper formatting
* Printout of "(result1, observation)" tuple converted to OpenAI JSON format
  - NOte: this printout has TWO "Message" types:
    * "AIMessage" = the function call LLM decided to use
    * "FunctionMessage" = the "observation" converted to "Message" type format
    * This implies that OpenAI JSON format REUQIRES using the "Messages" types 
   
#### There are 7 steps to create an action-memory chain/agent (using a loop):
1. Create the chain
    - Will need to be created before this loop part/fcn
2. Initialize an empty list
    - Will be the mem. for the action history
3. Invoke chain using the action history list (step 2) and CURRENT user query
    - Action history list will need to be converted to OpenAI JSON format
4. Check whether chain reached "AgentFinish"
5. Choose appropriate tool to use
    - I.e., check the tool choose/outputted by the LLM
6. Run the tool using the LLM's chosen inputs for the tool
7. Append the tool's results to the action history list
    - Loop step 3 through 7
      * Loop will stop once "AgentFinish" is reached  
   
#### Enabling long-term memory
* Requires the use of ANOTHER "MessagesPlaceholder"
  - This one will be used to store CHAT history of ALL queries
  - Recall: the other was used to store ACTION history on CURRENT query
  - Therefore we'll need to update prompt setup again  
   
   
   
   
   
   
   
   
   
### Course 7: AI Agents in LangGraph

#### Key design patterns of agentic workflow
 - Planning: thinking through the STEPS to take (ex: doing an outline)
 - Tool use: being AWARE of avaialble tools and using them
 - REflection: ITERATIVELY IMPROVING results
   i.) Possibly using MULTIPLE LLMs to make useful suggestions
 - Multi-agent communication:
   i.) EACH LLM will have a UNIQUE prompt/role in the multi-agent system
   ii.) Establish communication between these DIFF. LLMs
 - Memory: tracking the progress and results over multiple steps
 - System capabilities: ex- search, tools, human-in-the-loop, persistence
   i.) Many of these capabilites are implemented OUTSIDE the LLM
   ii.) These are usually FRAMEWORK DEPENDENT
   
#### ReAct Details:
 - PAper: Arxiv number 2210.03629
 - Pattern details:
   i.) LLM THINKS about what to do
   ii.) LLM DECIDES an ACTION to take
   iii.) ACtion is EXECUTED in an environment
   iv.) And OBSERVATION is returned from the executed action
   v.) With that observation, repeat steps 1-4
 - Bootlegged diagram:
   [Reasoning Traces] (<- thought ->) [LLM] (-action->)(<-observations-) [ Environment]

#### ReAct agent with few shot examples
 - functions used in ReAct prompt
 - Similar to basic agents from Course 1
   i.) Agent will give back fcn to use
   ii.) But fcn and observation will need to be processed manually
 - Agent will need to be RE INITIALIZED to clear previous msgs

#### Understanding "Annotated" type hint, 'operator' library, and "AnyMessage" type:
 - "AnyMessage" = a "Message" type that is umbrella for ALL "Message" types
 - "opeator" = details HOW the "Messages" will be added (ex: append, overwrite,...)
 - "Annotated" = a type hint that allows for detailing metadata
   i.) Syntax: "Annotated[Type, meta1, meta2,...]"
   ii.) "Type" = the actual type being investigated (ex: string, int, float, ...)
   iii.) metadata = extra details ABOUT the "Type" (ex: "greater than 10", "negative",...)
   iv.) Ex: "Annotated[int, "must be positive", {"min_value":1}]"

#### 5 steps in an agentic search:
 1.) Agents needs to DECIDE whether to use a tool or not
   -) I.e., does it have the information in its training data or not?
 2.) Agent analyzes Q and breaks it down into sub-Qs (if needed)
 3.) For EACH sub-Q, the search tool will have to find the BEST SOURCE, choosing from multiple integrations
 4.) From the produced sources, the search tool extracts on the RELEVANT sub-Q info
   -) BAsic implementation: chunk the avaiable sources and run a vectorbase search
     i.) The vector search will retrieve the top-K chunks
 5.) Search tool scores the results and filters out the LESS RELEVANT info

#### Scraping content from a given URL
 - Good for getting started with web scraping
 - returns a raw form of HTML

#### Difference between regular search and agentic search
 - Regular: results are catered towards HUMAN needs
 - Agentic: results are catered towards AGENTIC needs
   i.) Specifically, agents NEED STRUCTURED FORMAT, like JSON

#### Defns: persistence and streaming
 - Persistence: KEeping around the state of an agent at a PARTICULAR pt. in time
   i.) Lets you go BACK to that state and RESUME from that state
   ii.) Needed for LONG running applications and using human-in-the-loop
   iii.) Allows for continuing conversations from PREVIOUS invocations
 - Streaming: emit a LIST of SIGNALS that give details about what's currently happening
   i.) Useful for long-running apps to know what is happening
   ii.) Can be done based on "Message" types or on tokens
   iii.) Requires the use of: {"configurable": {"thread_id": "1"}}
     a.) Requires the USE of persistence memory through "checkpointer"

#### Cerating (in-memory) persistence
 - Since it's in-memory, memory WONT persist after the process terminates
 - Agents can be connected to external DBs to create true persistence
   i.) Avaiable DB languages: SQLite, Redis, Postgres, etc.
 - Needed in order to USE thead configs
   i.) Which is used to keep track of conversations

#### Streaming tokens, async methods, and async checkpointers - details:
 - Streaming tokens requires the use of "astream_events" method
   i.) This method is available on ALL LC and LG OBJECTS
 - Asynch methods requires the use of async checkpointers

#### HITL changes to agent state and its attributes
 - With HITL, there might be times that human disagrees with agent decision
  i.) In this case we will want to REPLACE the existing agent state
    a.) The replacement will highlight human preference overriding agent preference
  ii.) But, previously we've had agent state attributes be updated through APPENDINGS
    a.) If we want to replace, then appending will need to be changed to overwriting
  iii.) HOwever, this overwriting might need to happen in the middle of a "Message" log
    a.) I.e., we might want to keep some "Messages" and not overwrite everthing
    b.) Moreover, we might want to replace a specific "Message" in middle of a log
  iv.) To solve this issue, we'll need to create a CUSTOM replacer function

#### Incorporating HITL in the agent class with interrupts - details:
 - Recall: an agent progresses automatically until it reaches "AgentFinsih"
 - But, if we want HITL, then we'll need to add interrupts to slow this automatic progress
 - There are 2 interrupts: "interrupt_before["node1",...]" and "interrupt_after["node1",...]"
   i.) "nodeX" will need to be replaced with the node name as it appears on the graph
   ii.) These interrupts are parameters to "graph.compile"

#### Debugging: creating mock LLM responses by targeting specific "Message"s in a snapshot 
 - Useful for debugging different parts of the agent and LLM
   i.) Debugging how an LLM responds to a random user query
    a.) Can reveal a lot of details of the strength of the prompt
   ii.) Debugging the path an agent takes given a previous movement/"Message"
    a.) Recall: agent movements across the graph are captured by "Message"s
    b.) Therefore if we put in different "Messages", we can see strength of graph
    c.) Can reveal if graph works as intended
      * Including diff. fcns used INSIDE the graph itself

#### Node functionality's order details:
 - The logic inside the node fcnality is executed before the snapshot is updated
   i.) The snapshot is only updated once the return is reached
 - In this case, the graph has a counter that increases after each step
   i.) the node logic has 2 components:
     a.) Print out the current counter number
     b.) Perform the snapshot update (through the return)
 - The print happens before the return
   i.) Thus the print will have the counter BEFORE it is updated
 - I.e., the counter that gets printed WON'T be the value that shows up in the snapshot
   i.) Since the print happens before the update is done
 - In general: logic done inside the node fcnality happens before snapshot is updated

#### Steps to create an essay writer:
 1.) Plan
  - Generate a plan for the essay
  - Happens once upfront
 2.) Research the plan
  - Base on step 1 (plan)
  - Involves calling search tools to retrive documents
 3.) GEnerate
  - Generates essay following plan and using retrieved docs
  - Conditional edge: reflect OR finish
 4.) REfelction
  - Generate a CRITIQUE of the current essay
 5.) Research the critique
  - Do another search (tool call) and retrieve more docs
 6.) Generate (again)
  - Write the esay based on the critique and new info
  - LOOP: until conditional edge reaches finish

#### Essay writer details
 - Will need many diff prompts to execute each step's fcnality
 - BUt, SAME MODEL will be used throughout agent
  i.) Diff prompts are fed to the SAME model at diff. nodes

#### Helpful LangChain/ LangGraph resources:
 - LC documenation page
 - LangSmith
 - LC Github repo
 - Prompt Hub

#### Agent flows/architectures supported by LG:
 - Multi-agent architecture
  i.) github.com/langchain-ai/langgraph/tree/main/examples/multi-agent
 - Supervisor agent architecture
 - Flow Engineering
  i.) Source: AlphaCodium research paper
 - Plan and execute paradigm
 - Language agent tree search
  i.) github.com/langchain-ai/langgraph/blob/main/examples/lats/lats.ipynb

  
   
   
   
   
   
   
### Course 8: Long Term Agentic Memory

#### Simple-agent vs state-agent vs MAS-agent - IMPORTANT:
 - This is essentially a review of most important concepts in past 2 courses
 - Note: these are PERSONAL terminologies to help guide personal work
 - The use of "agentic" objects requires the use of LLMs
   i.) I.e, not convolutional models, nor feed-forwards models, etc.
   ii.) LLMs provide the REASONING agents require to execute tasks
   iii.) BUT non-LLM models can still be used with agents to extend capabilities
 - Recall: Pydantic models are used to create a STRUCTURED OUTPUT
 - Definitions:
   i.) Simple-agent: a LLM agent that does not have an agent state
     a.) In general, it doesn't have mem. capabilities
       * But this can be faked by appending results to a list
     b.) There are 2 ways to create a simple-agent:
       * Using a LC chain
         - Ex: "agent_chain = preprocess | prompt | model | outputParser"
           i.) See C6VI (Conversational agent), 4 pages in
         - Recall: using tools with chains
           i.) The tools must be defined OUTSIDE the chain
           ii.) (Optional) the tool must have an associated Pydantic model
           iii.) The tool must be converted to OpenAI format
           iv.) After conversion, tool must be bounded to model
             a.) "model = ChatOpenAI.bind(functions=tool_fcns)"
               * See C6VI (Convesational agent), 1 page in
         - Recall: pydantic models can ALSO be directly used as tools/fcns
           i.) Through the conversion fcn "convert_pydantic_to_openai_function"
             a.) See C6IV (Tagging and Extraction), 1 pg in
           ii.) Once converted to OpenAI format, they can be bounded to model
             a.) "model.bind(functions=taggin_fcn"
         - Note, models can ALSO be bounded with pydantic models
           i.) See C7VIII (Essay Writer), 4 pgs in
             a.) "model.with_structured_output(pydantic_model)"
           ii.) This only formats the FINAL output of the agent to fit pydantic's struct.
         - Mem. (for single query) and automatization can be faked by using a for loop
           i.) See C6VI (Conversational agent), 4 pgs in, "run_agent"
           ii.) "intermediate_steps=[]" = stores agent's executions as it answers query
           iii.) "tool.run()" = automatically runs tools 
              a.) Results are then turned into OpenAI format (under "preprocess")
                * Conversion is needed so that these results can be fed back into LLM
       * Using a python "class"
         - This gives more flexibility to the anatomy of the simple-agent
         - See C7II (Building Agent from scratch), 1 page in, see "Agent"
         - It still requires the use of:
           i.) Prompts (system and user)
           ii.) Tools 
             a.) Although not explicit in the code example, it can be done
             b.) Under "execute" method, the model can be bounded with tools
                 * In the same way that it was explained above
           iii.) Keeping track of short-term mem through fake mem.
             a.) Much in the same way as "intermediate_steps" above
                 * Except here it's "self.messages"
           iv.) A loop for continuous interaction with user
             a.) See C7II, 4 pgs in, look at "query" fcn
     c.) Note: simple-agents do NOT have mem. of PREV. INVOCATIONS
       * I.e, they don't have long-term memory
       * They only have short-term mem. on the CURR. query in the CURR. invocation
         - Which is technically faked, as mentioned above
           i.) "intermediate_steps=[]", "self.messages"
  ii.) State-agent: LLM agent with an associated agent state
    a.) This state can keep track of diff. attributes across DIFF. INVOCATIONS
       * These history of these attr. changes is in "StateSnapshot" history
         - "StateSnapshot" also contains MORE valuable information
       * These attributes MUST:
         - Be defined in the "AgentState" class
         - Specify how UPDATES will show up on these attr. changes
           i.) Ex: "count: Annotated[int, operator.add]"
              a.) See C7VII (Extra practice), 0 pgs in
           ii.) Ex: "messages: Annotated[List[BaseMessage], reduce_messages]"
              a.) See C7 VI(Human in the loop), 1 pg in
    b.) The state must be used when creating the main "Agent" class
       * See C7III (LG COmponents), 4 pgs in
       * Usually under "graph = StateGraph(AgentState)"
       * Requires setting up the graph structure as well
         - Node and coditionals are usually defined inside python class itself, too
       * Tools are also used when creating the state-agent "Agent"
         - BUT these tools must be defined OUTSIDE the "Agent" class
         - Still requires using "@tool()" decorator
         - Can be used with pydantic models through "@tool(args_schema=py_model)"
           i.) As was done in the previous course
    c.) In order to activate history-storage of "StateSnapshot", you need:
       * To activate long-term mem. by using "checkpointer" param. of agent
        - Avaialable types of mem. found in:
         i.) "langgraph.checkpoint.memory"
         ii.) "langgraph.checkpoint.sqlite"
       * Pass in a thread configuration to the invocation
        - This config contains "thread_id" and "thread_ts"
         i.) "thread_id": identifies the specific conversation you want
           a.) Diff. convos have differenent "thread_id"
         ii.) "thread_ts": identifies the specific "StateSnapshot" of that convo.
   iii.) MAS-agent = creating a system that uses multiple agents
     a.) Each agent may or may not have it's own agent state
     b.) A proper defn is difficult to pin bc of the variety of architectures posible
     c.) Supervisor architecture = a supervisor agent delegates tasks to sub-agents
        * The supervisor is usually the only one with an agent state
        * The sub-agents usually consists of simple-agents
     d.) Peer-coordinated architecture = all agents work in coordination with each other
        * This can consist of a mix of simple-agents and state-agents
          - Analogy: state-agents = employees and simple-agent=printer
          - Each employee has a role/task, but they can all use the printer 
        * There might also be a general "fake" agent who's only task is to hold the state
          - I.e, maybe each state-agent has a task to complete
            i.) They use their personal state to complete the task
          - Once the task is complete, they upload that version to the "fake" agent state
            i.) IT's possible this "fake" agent's state can be accessed by all agents
            ii.) USeful when needing the results of one task to start on another task  
 - When to use:
   i.) Simple-agent: use on simple tasks that don't require remembering anything
      a) Ex: classification of an input text
        * The classification categories can be detailed in the agent's prompts
      b.) Good for tasks that DON'T need to remeber details of prev. invocations
   ii.) State-agent: use on more complex tasks that require remember details
      a.) Remebering details such as:
        * Tool results, prev. conversations, current step in an execution, etc.
      b.) These details will be stored in the agent state
   iii.) MAS-agent: use on even more complex tasks that require the use of many agents
      a) Specifically, tasks that would require the use of diff. skills
        * Such as a personal email assistant
          - Needs to: know which emails to prioritize, how to respond, who they talk to,..
      b.) Recall: diff. architectures are best suited for diff. tasks
      c.) Bear in mind the use of state bc it'll take a tool on mem. storage SPACE/SIZE

#### Concepts to consider when adding mem. to agents:
 - Figure out what info to store in long-term mem.
 - When needing long-term info, figure out what info needs to be retrieved
 - Figure out when to update the stored info

#### 3 types of memories:
 1.) Semantic: facts, like birthdays for a calendar agent
    - Human: things I learned in school
    - Agent: facts about a user
 2.) Episodic: experiences that can help an agent remember how to do tasks
    - Human: things i DID in school
    - Agent: past agent actions
 3.) Procedural: rules for an agent to follow
    - HUman: instincts/motor skills
    - Agent: agent syst. prompt

#### There are 2 mechanisms for updating/saving mem.
 1.) Hot path: agent updates mems. as it's responding
    - Pro: only 1 agent; con: single agents MUST respond and update
    - Ex: user msg-> update mem.-> respond to user-> user msg-> update mem. ->...
 2.) BAckground: mem. updates happen in the background/separate process
    - Pro: separate agent is solely in charge of updating; con: multiple agents
    - Ex: User msg-> response-> user msg-> response->...-> *After 1 hr* update mem

#### Steps to create an email assistant agent
 1.) Add semantic mem.
    - Agent will needs:
      i.) Tools for WRITING to mem.
         a.) Encompasses semantic mem. PROFILE
         b.) Meeting preferences, response priorities, VIP contacts, etc.
      ii.) Tools for READING fro mem.
         a.) Encompasses semantic mem. COLLECTION
         b.) Facts, past interactions, etc.
    - Tools will be used with "hot path" mech.
 2.) Add episodic mem.
    - Will be placed in triage LLM
      i.) Will filter out which emails to reply to
    - Will be added in the form of few-shot examples
      i.) Examples will be placed in triage LLM's prompt
      ii.) Success exs: urgent client requests, VIP meet requests, etc.
      iii.) Failure exs: spam, newsletter, low-prio emails, etc.
    - Will be updated with "background" mech using separate PROCESS
 3.) Add procedural mem. 
    - Create PARTS of prompts that contin INSTRUCTS on how to use tools/triage
    - Will be updated with "background" mech using separate AGENT

#### Guided Qs when adding long-term mem.:
 - Semantic: Does it need to know facts about ppl and things?
 - Episodic: Will a few shot exs. guide its actions better?
 - Procedural: does it need to learn better instructions?
 - It all DEPENDS on the APP you're building

#### Important info on 3 types of prompt - general vs ambigious vs specific prompts:
 - There is slight, but important, diff. b/w the 3
  i.) Note: prompt has general instructions for router agent and main agent
  ii.) Note: this prompt will NOT be used DIRECTLY in either agent
     a.) Each agent will have a separate prompts that are diff. from this general prompt
     b.) BUT each's agent's prompt will PULL PARTS from this GENERAL prompt
  iii.) In essence, this general prompt gives details SPECIFIC to our task/case
  iv.) Each agent will pull from pre-existing, library prompt
     a.) Since these are library prompts, they are ambiguous (for anyone to use)
     b.) Thus, we'll use bits of this GENERAL prompt into those prompts
        * Turns ambigious prompts into specific prompts by using bits from general prompt

#### Exploring diff. types of mem. - short term vs long term:
 - Good source: https://docs.langchain.com/oss/python/langgraph/memory
 - Prev., we have used short-term mem. through in-mem persistence enabling
  i.) Create "short_mem = InMemorySaver()"
  ii.) Pass this short term mem. into "checkpointer" arg of agent intiiatlization
  iii.) This allows for play/modification of "StateSnapshots"
  iv.) Tracks ongoing conversations WITHIN A SESSION
  v.) Maintains message "StateSnapshot" history WITHIN A SESSION
 - Long-term mem. is diff. than short-tmer
 - Long-term mem details:
  i.) Create "long_mem = InMemoryStore()"
  ii.) Pass this long-term mem into "store" arg of agent initalization
  iii.) Stores data ACROSS DIFF. SESSIONS
  iv.) Long-term mem. is shared ACROSS CONVERSATIONAL THREADS
  v.) Mem.s are scoped to a custom namespace
     a.) NOT JUST within a SINGLE thread id
  vi.) Can be recalled at ANYTIME in ANY THREAD
 - NOte: semantic, procedural, and episodic mem all require LONG-TERM mem.
 - With semantic mem., long-term mem. is managed/search THROUGH TOOLS
  i.) While short-term mem is updated AUTOMATICALLY, REGARDLESS of tool used

#### Rule of thumb for short-term and long-term mem.
 - Generally, you want diff. agents to have diff. short-term storage
  i.) That way you can track each agent's movements INDIVIDUALLY
  ii.) If you use the same short-term storage, then they will add on top of each other
    a.) This will make it hard to see which agent executed which tool
  iii.) Alternatively, you can use diff "thread_ids" for diff agents
    a.) But this can also get messy with storage
    b.) GEnreally, diff "thread_ids" are for diff convos on the SAME agent
 - But, you MAY want to have the same long-term storage for all agents
  i.) Since diff. agents may acquire diff. semantic/procedural/episodic mem at diff times
  ii.) Diff. agents may gather user knowledge at diff times
    a.) But you want this info to be avaialable to all agents
    b.) You want agents to use all users's info that's avaialble
       * Even if that agent didn't acquire that specific info
 - NOTE: this dynamic MIGHT CHANGES with NESTED AGENTS
  i.) See below "Exploring nested dynamic b/w "email_assistant" and "response_agent""
  ii.) Essentially, if they share the same short-term mem, then they SHARE SAME HISTORY
  iii.) Diff. short-term storages will prevent this overlap
  iv.) BUT they can still share the same long-term storage
     a.) That way they can access the same updated info found in that storage

#### Gameplan to intergrate semantic mem.
 - Recall: semantic mem. involves remembering facts about a user
 - There are 2 LLMS being used
  i.) Even though there are 3 agents
    a.) Recall: email-assistant MAS-agent doesn't itself have a LLM
       * It simply uses the LLMs from the router and main agents
 - Therefore one of these 2 LLMs will be working with semantic mem. directly
  i.) Router's role is to simply to classify an email
  ii.) Main agent's role actually manages communication between emails
 - Thus main agent will need to work directly with semantic mem.
  i.) We'll enable this by giving the main agent additional tools
    a.) Specifically, 2 tools: 1 to manage mem. and other to search that mem.
       * This implies that we'll need to setup in-mem. persistence
 - After we give these 2 tools to the main state-agent, then semantic mem. is enabled
  i.) The email assistant will have semantic mem. abilities

#### Exploring nested dynamic b/w "email_assistant" and "response_agent"
 - Exploring long-term after "response4" invocation
  i.) Here we can see that the long-term mem hasn't changed after this invocation
  ii.) This makes sense bc the invocation did NOT call the "manage_memory_tool"
     a.) Evident through the snapshot history in line 1243
 - Exploring "response_agent" snapshot history after "response4" invocation
  i.) We can see that this history is SAME as "email_assistant" history in line 1180
  ii.) This happens for 2 reasons:
     a.) Both agents are using the SAME "config" setup
        * I.e., they have the same "thread_id"
     b.) Both agents are using the SAME short-term mem storage
        * Namely, "snapshot_store"
  iii.) What's interesting here is the nested dynamic
     a.) It makes sense that "response_agent" would have ALL invocation history
        * I.e., invocations 1-4
        * Since "response_agent" is nested INSIDE "email_assistant"
     b.) BUT, it's not intuitive that "email_assistant" would have ALL invocation history
        * Since "email_assistant" technically onlt invoked 2 times
         - Namely, "response3" and "response4"
        * And also bc "email_assistant" is not nested inside "response_agent"
     c.) This reveals that when agents use the same short-term mem. storage, then:
        * ALL invocation history gets dumped into the same storage spot
        * Any agent accessing that storage will get ALL of info in it
         - Regardless if the info was dumped in there by them or by another agent
     d.) Although not in code, the dyanmic in (c.) above was VERIFIED by doing:
        * Running "response1" and "response2" through "response_agent"
        * Checking out the "response_agent" state history
        * Setting up "email_assistant" with same short-term mem storage
        * Running "print(email_agent.get_state(config))" B4 "email_agent" did any invokes
         - This results in the SAME state history as "response_agent"
           i.) EVEN THOUGH "email_agent" did NOT do any invocations
 - NOTE: it is NOT ncessary to compile BOTH agents with "checkpointer"
  i.) I.e, the INNER agent does NOT need a "checkpointer"
  ii.) Putting a checkptr on the outer agent GUARANTEES:
     a.) That outter agent will have snapshot history
     b.) I.e., outter agent will have short-term mem. enabled
  iii.) BUT, snapshot history WONT be enabled for inner agent
     a.) EVEN THOUGH inner agent is working with outer agent checkptr
  iv.) This is helpful when you have limited storage you're working with
  v.) Also helpful if you don't care about inner agent snapshot history

#### Note on in-hot-path update mechanism:
 - The update mechanism from this section is "hot-path"
 - We see this bc the agent itself is doing the updating of the information
  i.) The agent is also in charge of executing its task
 - Ex: "response_agent.invoke("kim is my friend")"
  i.) Agent will process response through the LLM
  ii.) Agent will use the "manage_memory_tool" to write info to long-term storage
  iii.) Agent will then send a final response to the user
 - THUS, to enable "hot-path" update, simply give the agent the following tools:
  i.) "manage_memory_tool": writing info into long-term storage
  ii.) "search_memory_tool": searching info inside long-term storage

#### Episodic mem. overview:
 - Episodic mem. will be added through few-shot examples
 - These few-shot examples will be used to SIMULATE past agent actions
 - Recall: few-shot examples should have successed and failures
 - Few-shot will be implemented by:
  i.) Creating some sample emails
  ii.) Associating a label to those emails
  iii.) Ssaving the pair (sample email, label) into long-term mem.

#### Integrating episodic mem. to agents:
 - Recall: episodic mem. will be done by simulating past actions through few-shots
 - These past actions will resemble how an agent SHOULD CLASSIFY emails
 - The few-shots method will give (sample email, label) examples to guide agent
 - This episodic mem. relates to the CLASSIFICATION of emails
  i.) Thus, this episodic mem will be directly integrated with the ROUTER agent
 - REcall: router simple-agent was in charge of classifying emails
 - Speficially, these few-shot exs. will be fed to router's SYST. PROMPT
 - REcall: the way email assistant uses the router is by creating a node for it
  i.) The node wraps the router in additional fcnality and logic
  ii.) Thus, to integrate episodic mem., add the few-shot setup in this node fcnality
 - The few-shot examples will be stored in the LONG-TERM mem
  i.) Thus, short-term mem isn't needed
    a.) Unless you want to work with "StateSnapshot"

#### Storing few-shots in long-term mem. 
 - You will need to create a SPECIFIC data model which you will format the examples
  i.) Ex:"data = {"email": {"author": ..., "to":..., ...}, "label": "respond"}
 - NOTE: THIS CAN BE DONE WITH A PYDANTIC MODEL
 - This will make it easier when RETRIEVING the few-shot examples to work with them
  i.) If they all follow the same data model, then you can create an easy template
    a.) The template it used to format the few-shot exs. into the syst. prompt for router

#### Storing the coupled few-shot exs in long-term mem. - important format:
 - line 1971
 - Requirements: long-term storage namespace, data ID, data itself
  i.) IN this case, the long-term namespace is explained below
 - NOTE: the namespace is DIFF. for the few-shot examples
  i.) Namely: "(email_assistnat, {langgraph_user_id}, example)"
  ii.) For manage and search tools, it was the following:
    a.) "(email_assistant, {langgraph_user_id}, collection)"
      * See lines 740 and 747

#### Understanding the long-term storage's namespace and interaction with "langmem"
 - Recall: long-term mem. is created with "InMemoryStore"
 - BUT intearcting with this long-term mem REQUIRES specific tools
  i.) Specifically, "manage_memory_tool" and "search_memory_tool"
 - These tools were created from "langmem"
 - When creating these tools, the namespace assigned to them was 
  i.) "namespace=("email_assistant", "{langgraph_user_id}", "collection")"
 - I.e., there was space created in the long-term mem. storage such that:
  i.) The NAME of that SPECIFIC space is "namespace"
  ii.) Accessing that SPECIFIC space is done through "namespace"
 - This allows for other parts of the long-term mem. storage to be untouched
  i.) And those parts can be used for OTHER projects/agents
  ii.) Those other parts can be associated with other projects through a DIFF namespace
 - I.e, the namespace is SPECIFIC to this project and the agents using it
 - The namespace helps organize the long-term storage space
  i.) This is EXACTLTY what's being done when using separate namespaces
    a.) Few-shot namespace != manage/search tool namespace
    b.) See bullet before this section

#### Relationship b/w namespace and agent behavior
 - REcall: episodic mem. CHANGES an agent's behavior
  i.) Agent learns from past actions, which shape future actions
  ii.) These past actions are implemented in the form of few-shot exs
 - Recall: few-shot exs. are stored in a specific namespace in the long-term storage
  i.) If exs are stored in diff. names spaces, then they will impact the agent behavior
 - Specficially, an agent using diff. namespaces will:
  i.) Be using diff. few shot exs.
  ii.) Thus, will have a diff. behavior/classification on user input

#### Relationship b/w few-shot exs and namespaces - details:
 - Few-shot exs are SCOPED to INDIVIDUAL namespaces
  i.) I.e., few-shot exs DONT carry over b/w diff namespaces, even on SAME AGENT
 - If you do few-shot exs on a spcific namespace and then change the namespace, then:
  i.) It will affect the agent behavior
 - Few-shot exs. are pulled in at RUNTIME
  i.) Since few-shots are pulled in INSIDE the node's fcnality
  ii.) Thus every invocation changes few-shots used => changes syst. prompt used

#### Note on background-update mechanism through diff process:
 - The updates in this section are done by a separate process
 - Specifically, they are done by "store.put"
  i.) This is called on separate lines/process than the agent
   a.) Agent is focused on calling invocations
 - Thus, to enable "backgroud" update using a diff. process, simply:
  i.) Find a process/API that can update info into the long-term storage
    a.) Ex: "storage.put"

#### Procedural mem overview:
 - This is the mem. involved with INSTRUCTIONS on how to do things
 - The integration of this type of mem. will involve syst. prompts
  i.) It will affects BOTH syst. prompts: from router and main agent
    a.) Recall: email assistnat has a state but not an LLM (=> has no assoc. syst. prompt)
 - Recall: there are 3 types of prompts (personal terminology)
  i.) General prompt: the "prompt_instructions" in the code
    a.) This contains details that are specific to our task/application
    b.) Specifically, contains SPECIFIC INSTRUCTIONS that are used by other agents
  ii.) Ambiguous prompt: pre-exisitng prompts pulled in from libraries
    a.) These are ambiguous so they can be used by diff. applications
    b.) Ex: "from prompts import triage_system_prompt, triage_user_prompt"
  iii.) Specific prompts: the integration of general prompt into ambiguous prompt
    a.) It allows these ambiguous prompts to change to fit our specific application/task
 - Procdural mem. will be reflected through AUTOMATIC UPDATES of the specifc syst. prompts
  i.) BUT, these will be MANUAL changes to the general "prompt_instructions" prompt
 - Since general prompt will change, we DONT want direct reference to "prompt_instruction"
  i.) Instead, we'll used long-term storage as the MIDDLE PERSON
  ii.) We'll save general prompt to storage and ambiguous prompts will PULL from storage

#### Background-update mechanism using a diff. agent -details:
 - Recall: our gameplan is to create a middle person between general and ambig. prompt
 - This separation is NECESSARY to do background-update with diff. agent
 - That way:
  i.) Router and main agent focus on: mem.storage -> specific prompt
    a.) They'll also be in charge of PUTTING the general prompt into long-term storage
       * This is triggered if the general prompt isn't already in the storage
  ii.) Update agent focuses on: updates -> mem.storage
 - Router and main agent will incorporate pulling from mem. storage as follows:
  i.) Router pulling will occur in it's node fcnality, used by email assistant
  ii.) Main pulling will occur in it's "create_prompt" fcn
    a.) Recall: this fcn is passed into "create_react_agent" to initialize main agent
    b.) This fcn is in charge of the syst. prompt for the main agent
 - If updates happen in b/w invocations, then post-update invocations will reflect them

#### Implementing procedural mem. gameplan:
 - We'll need to change the dyanmics of the agents to allow for backgrd-update agent
 - Router agent's node fcnality will be changed as follows:
  i.) Search for general prompt instructions in long-term mem. storage
    a.) Will search the namespace where general prompt instruction info is stored
  ii.) Store general prompt  instructions into long-term mem. storage
    a.) Triggers if general prompt instructions isn't in storage
 - Main agent's dynamics will change as follows:
  i.) "create_prompt": pull general prompt instruction info from long-term storage
   a.) Similar to above, it will search the namespace where general prompt info is stored
  ii.) IT'll also store general prompt instructions into long-term mem. storage
   a.) Triggers if general prompt instruciton isn't in storage
 - Create a background-update agent
  i.) Will be a pre-existing agent from "langmem'
    a.) "langmem.create_multi-prompt-optimizer"
  ii.) More info on separate bullet pt below
 - Note: email agent itself won't DIRECTLY pull from long-term mem. storage
  i.) It'll INDIRECTLY do that pull through the sub-agents inside of it

#### Update agent details:
 - It'll be created using "langmem.create_multi_prompt_optimizer"
  i.) Github pg: https://github.com/langchain-ai/langmem/blob/main/src/langmem/prompts/optimization.py
 - It doesn't seem to have an agent state, so it's a simple agent
 - IT will update instructions based on user FEEDBACK
 - NOte: users DON'T give feedback for SPECIFIC instructions
  i.) They give feedback on the OUTPUT of an agent
  ii.) They rarely know the specific prompts being used 
 - Upon this overall feedback, update agent must DETERMINE which instructions to update
  i.) Specifically, which parts/sections of the general prompt (line 2480) to update
 - Since update agent uses a LLM, it will require some form of update-prompts
  i.) These update-prompts will instruct LLM on HOW updates should occur
  ii.) There should be a 1-1 b/w update-prompts and instruction sections
    a.) Ex: the general prompt (line 2480) has 4 sections =>  4 update-prompts
 - EACH update-prompt will have the following placeholders: 
  i.) "name": name of the instruction section being worked on
  ii.) "prompt": the current instruction stored in long-term mem.
  iii.) "update_instructions": how the updated instructions should look like
     a.) Expected format for update agent's output, which will be an updated instruction
  iv.) "when_to_update": guide for when agent should update this specific section

#### Invoking an update agent:
 - Recall: update agent applies user feedback to another agent's execution/results
 - Thus, update agent will NEED that agent's execution as part of its context
  i.) Needed in order to see what the user feedback is referring to
 - Then, with relevant context and feedback, the update agent must determine:
  i.) How to apply that feedback, to create a new updated instruction for other agent
    a.) Requires knowing what is the agent, to find out what instructions they use
  ii.) Where to apply that feedback, where to store the new updated instruction
    a.) Requires knowing where the old instruction was stored at
 - Thus, to invoke an update-agent you need:
  i.) A copy of the other agent's execution and results
    a.) This can be done by getting their latest "StateSnapshot"
       * Which reveals EVERYTHING about a specific invocation through the "MEssages"
        -) USer input, LLM thoughts, tools used, etc.
  ii.) The user's feedback itself
    a.) This can be appended with the "StateSnapshot"
  iii.) THe update-prompts, 1-1 correspondence with the number of instruction section
    a.) See above bullet for more info




## CrewAI Courses
### Course 9: Multi-AI Agent Systems

#### Multiple strings vs triple quote docstring:
 - Multiple string ex: 
  i.) "sentence 1"
       "sentence 2"
 - TRiple quote docstring: 
  i.) """ sentence 1
          sentence 2
      """
 - Using multiple strings can avoid whitespace and newline characters
  i.) Makes it better formatted to be passed into LLM

#### MAS overview:
 - Breaks down COMPLEX tasks into SUBTASKS that can be executed by agents
  i.) Each agent has a particular, defined ROLE
 - You can CUSTOMIZE each agent to execute/solve a sub-task efficiently
 - Each agent can run a diff. LLM

#### 6 elements that make a great agent - IMPORTANT
 1.) Role playing
    - MAkes a HUGE difference in agent responses
    - Gives agent CONTEXT on how to style outputs
    - Being meticulous on role-playing will lead to BETTER results
 2.) Focus
    - Mixing up too much context/info can lead to loss of important info
     i.) LEads to MORE HALLUCINATIONS
    - Agents should be FOCUSED on:
     i.) The amount of TOOLS they use
     ii.) The amount of context they receive 
     iii.) The thing/goal/task they're trying to achieve
    - DONT rely on only ONE agent to do EVERYTHING
     i.) Multiple agents work better together
    - You want to focus on:
     i.) Narrowly defined tasks
     ii.) Specific agent ROLES & OBJECTIVES
     iii.) Limited set of tools assigned to ONE agent
 3.) Tools
    - You could potential OVERLOAD agent when giving it TOO MANY tools
     i.) Too many tools => hard time choosing ONE TOOL
    - Smaller models should be given LESS TOOLS to work with
    - Be INTENTIONAL & SELECTIVE w/ toolsyou provide your agents
    - Only provide KEY TOOLS that are NECESSARY to complete the goal/task
 4.) Cooperation
    - Ability to BOUNCE IDEAS b/w diff agents produces BETTER RESULTS
     i.) Allows for easy integration of feedback
       a.) Feedback can be other agents' outputs
     ii.) Allows for easier time in picking an agent for a SPECIFIC TASK
 5.) Guardrails 
    - REcall: AI produces NON-DETERMINISTIC, PROBABILISTIC outputs
    - Guardrails prevent agent from derailing and helps them stay on track
    - Most guardrails are implemented at the FRAMEWORK LEVEL
     i.) => guardrails are FRAMEWORK DEPENDENT
    - Guardrails are ESPECIALLY important when building CUSTOM TOOLS
     i.) Prevents HALLUCINATIONS and ensures reliable results are produced
 6.) Memory
    - The MOST IMPORTANT element
    - Enables the ability to:
     i.) RECOLLECT past actions
     ii.) LEARN from the past actions
     iii.) APPLY the learning to FUTURE EXECUTIONS
    - Helps inform NEW deicison and NEW executions
    Diff. frameworks offer diff. types of mem. and diff. implementation processes

#### 3 types of memory for CrewAI:
 1.) Short-term mem.
    - Lives ONLY DURING the crew's EXECUTION of a TASK
    - Upon EVERY KICKOFF, this mem. starts from BLANK
    - AS diff. agents accomplish diff tasks, they STORE their LEARNINGS in this mem.
    - This mem. is SHARED across ALL AGENTS of the CREW
    - Allows sharing knowledge, activities, and learnings with OTHER AGENTS
    - Helps share CONTEXT during the crew's execution
 2.) Long-term mem.
    - Lives even AFTER the crew FINISHES EXECUTION
    - Mem. is stored in a DB
    - Allows agents to learn from PREV. EXECUTIONS
     i.) Produces better and more reliable outcomes
    - Can be used in ANY FUTURE TASK
 3.) Entity mem.
    - Short-lived
    - Only used DURING EXECUTION
    - Stores info about SUBJECT/ENTITES being discussed

#### MAS important dynamics:
 - In ANY syst., you usually want a FINAL AGENT that acts like a quality-check agent
  i.) They would do the FINAL QUALITY CHECK on other agents' results
 - If the quality-check agent finds something wrong, they DONT have to fix it themselves
  i.) IN fact, it might be WORSE if they try to fix it themselves
    a.) Since they probably don't have the proper tools to fix the error
  ii.) It's best to DELEGATE the error to some other agent with proper tools
    a.) Delegation decision is left for agent's LLM to decide
       * LLM's delegation decision is made at RUNTIME => little control over it
 - Allowing delegations STRENGTHENS agent's COLLABORATION
  i.) Yields better quality responses

#### Relationship b/w real life manager and MAS - guided Qs:
 - THere's a HIGH CORRELATION b/w good managers and great MASs
 - Guided Qs:
  i.) What is the GOAL you're trying to accomplish?
  ii.) What is the PROCESS you're trying to get your agents/employees to FOLLOW?
  iii.) What are the PPL/AGENTS you need to HIRE to get the JOB DONE?
 - Analyzing roles:
  i.) Okay roles: researcher, writer, financial analyst
  ii.) BEtter roles: HR Research specialist, Senior copywriter, FINRA approved analyst

 * IMPORTANT Steps to creating great agents - consider the:
  1.) Overarching SINGLE GOAL you're trying to accomplish
  2.) PROCESS NECESSARY to achieve that goal
  3.) KEYWORDS/COMPONENTS that make up that PROCESS
  4.) ROLES NEEDED in EACH COMPONENT
  5.) INDIVIDUAL GOALS EACH ROLE has within their CORRESPONDING COMPONENT
  6.) ADDITIONAL BACKSTORY each role NEEDS in order to EXCEL in their role

#### 3 key elements of a great tool:
 1.) Versatile
    - Accept DIFF. kind of REQUESTS
     i.) Recall: tools are the CONNECTION b/w AI apps and the external world
       a.) AI apps have fuzzy inputs and outputs
       b.) The external world can have strong-typed inputs and outputs
     ii.) Fuzzy data: text, instructions, mix of words and numbers, etc.
     iii.) Strong-typed data: JSON, key-value pairs, categories, etc.
    - Needs to handle DIFF. INPUTS that an LLM might throw at the tool
    - Tool MUST be able to CCONVERT b/w fuzzy and strong-typed data
 2.) Fault-tolerant
    - Tools sometimes break and throw exceptions
    - YOu must treat these exceptions in order to make the tool fault-tolerant
     i.) Allow the tool to FAIL GRACEFULLY and try again (self-heal)
    - Have exception errors be sent BACK to the agents as INPUTS to the agent
    - Being fault-tolerant prevents agents from STOPPING the EXECUTION
 3.) Caching
    - You want to store the returns from tool-calls (ex: APIs) in a CACHE
     i.) Helps avoid making UNECCESSARY REQUESTS
       a.) If you already have what you need, you don't need ot make the trip again
    - Having a cache layer that prevents unnecessary requests leads to more optimal agents
    - Ideally, you want SMART-AGENT caching
     i.) Ex: sps 2 diff agents need to use the SAME TOOL with the SAME ARGS
       a.) Thus, have them access the SAME CACHE layer
          * DO this BEFORE having them decide whether they need to make request or not
    - Benefits of caching:
     i.) Precents unnecessary requests
     ii.) Stays w.in/ RATE LIMITS allows for an API call
       a.) Rate limit = num of requests per second
     iii.) Saves time
        a.) Retrieving cache is FASTER than API calls
    - Caching is SUPER IMPORTANT when scaling

#### Manager framework when building agents - guided Qs:
 - What is the GOAL you're trying to ACCOMPLISH?
 - What is the PROCESS on how to achieve that GOAL?
 - Who are the PPL/ROLES you would hire?
 - How will you DELEGATE WORK to those employees?
 - How will you communicate the WORK PROCESS EXPECTATIONS?
 - How will you communicate the WORK RESULT EXPECTATIONS?
 - Which PROCESSES & TASKS do you expect the individuals in your team to do?
 
#### 3 key elements in a great task:
 1.) Description
    - CLEAR description of the tasks
 2.) Expected output
    - Set a CLEAR and CONCISE EXPECTATION
 3.) Any COMBINATION of the following, depending on task:
    - Set context
    - Set a callback
    - Override agent tools with specific task tools
    - Force human input before end of task
    - Execute asynchronously
    - Output as a pydantic, JSON, or file
    - Run in parallel

#### 3 types of agent collaboration PROCESSES:
 1.) Sequential
    - Pros
     i.) Waits for prev.agent to complete tasks before beginning
       a.) This can happen even w/ parallel execution
    - Cons
     i.) The INITIAL CONTEXT slowly fades as task gets passed from one agent to another
 2.) Hierarchical
    - Pros:
     i.) The manager agent ALWAYS REMEBERS the INITIAL GOAL
     ii.) Automatic delegation of work to other agents
     iii.) Manager agent reviews the results of the agents it's in charge of
     iv.) Managet agent can ask for further improvements, if necessary
 3.) Parallel/ Asynchronous
    - Execute certain tasks/agents in parallel
    - CAn also have other tasks wait for ALL parallel tasks to finish

#### Agent collaboration and "crewai.Progress" details
 - Diff. types of collaboration processes have diff. consequences
 - Recall: manager framework
  i.) You want to assemble agents in a way that will achieve the BEST RESULTS
 - Diff. collab. types can be set with "crewai.Progress" class
  i.) This sets diff. ways for agents to work together
 - This "Progress" class is passed into "Crew.progress" param. when initializing crew
  i.) Will ALSO require passing an LLM/agent to "Crew.manager_llm"
    a.) This will set up the manager agent
 - Manager agent will delegate work to ALL the diff agents
  i.) Sub-agent results will go BACK to manager agent
  ii.) Manager agent coordinates ALL the results
 - Final results will show manager agent's decision making
 - EACH sub-agent RESULT TEACHES manager agent something NEW
 - When building systs., make sure to create something that CREATES VALUE



### Course 10: Practical Multi-AI Agents w/ Advanced Use Cases

#### General tasks agents can do:
 - Pulling data out of existing systs
  i.) Ex: from DBs
 - Research
  i.) Ex: On the internet, other docs, other existing systs.
 - Analysis
  i.) Ex: Comparing data, extracting specific data, inferring new data
 - Summarization
  i.) Ex: extract learnings, plot charts, build executive summaries
 - REporting
  i.) Ex: as PDF, as JSON, as markdown
 - Push data into existing systs.

#### YAML files:
 - They can be used to store info on agents and tasks
 - They can also contain placeholders that'll be filled by other parts of the syst.

#### MAS and tools
 - In ALL MAS you need to either push/pull info from internal/external systs.
 - There are diff. moments in the crew's EXECUTION where you'll need this interaction
 - Sometimes you want TOOLS to interact w/ internal/external systs.
  i.) External systs: apps, cloud services, etc
  ii.) Internal systs.: DNs, internal apps, etc.
  iii.) External interactions: search web, check calendar, reply email
  iv.) Internal interactions: RAG, SQL query, trigger side effect, etc.

#### Creating complex MASs
 - Complex MAS can be created by connecting MULTIPLE CREWS
  i.) EACH crew will need its OWN associated YAML files/configs
 - Connecting diff. crews can be done through conditional logic
  i.) Will detail when to execute a crew and when to pass info into anothercrew
 - You can mix and match crews that have DIFF collab. orchestrations

#### Explicitly creating custom collab. orchestration:
 - Tasks that DONT depend on each other can be done in PARALLEL
  i.) Done through "Task.async_execution=True"
 - You can also pass in task's outputs as inputs to another task
  i.) Done through "Task.context"
  ii.) This is how you can CHAIN agent outputs and CONTROL COLLABORATION
 - You can also use flow to help customize orchestration
 - Flow = crewAI feat. that lets you execute REGULAR fcn before/during/after an execcution

#### 2 main components that impact MAS's performance:
 1.) Speed
    - Smaller models are CHEAPER and FASTER to run
     i.) Since they require smaller server
 2.) Quality
    - Bigger models usually have more capable reasoning
     i.) Can also better adhere to formats
    - DEPENDS on the KIND of TASK you're trying to accomplish
     i.) If you're working on non-complex tasks, then smaller model might do great quality

#### Quantifying MAS's performance
 - Whatever you're optimizing for on individual tasks, keep the SAME aim for ALL TASKs
  i.) COnsistency is KEY
  ii.) Make sure you're ALWAYS getting the SAME speed/quality as you go
 - On individual tasks, you can COMPARE the task results w/ the task's "expected_output"
  i.) Allows you to RANK tasks based on results
 - Look at MAS successes and failures
  i.) Will reveal PATTERNS that led to that success/failure
  ii.) Also reveals common use cases by clients

#### Agents and MAS challenges:
 - Agents: EAsy and hard parts
  i.) EAsy: creating the agent
  ii.) Hard: driving STANDARDS around the agents you're creating
    a.) How do you get to RE-USE agents across diff. use-cases?
    b.) How do you create a CONSISTENT foundation & pipeline for code generation?
       * BUT only change the BARE MINIMUM when changing b/w proprietary langs.
    c.) Crating development/standardization patterns
 - MAS deployment challenges:
  i.) Core infrastructure problems
  ii.) Token limits, rate throttling, etc.
  iii.) Change mgmt: getting ppl used to WORKING w/ agents
 - GENERAL TIP: start simple, then increase complexity as you go
  i.) Crawl, walk, run
  ii.) As you increase complexity, ANALYZE the GAPS that EXIST in the SIMPLER SYSTS.







## Course 11: Building Code Agents w/ smolagents

#### Definitions of diff types of agents:
 - Tool-calling agent: use LLM to write out MULTIPLE fcn calls SEQUENTIALLY
 i.) Completes a complex seq of tasks
 - Coding agent: agents that will write code for you to execute
 - Code agents: consolidates ALL fcn calls into a SINGLE block of snippet code
 i.) Writes code to perform a SEQ. of actions
 ii.) Executed in sandbox for safety reasons

#### 5 levels of AI agency:
 1.) Simple processor: LLM output has no impact on program flow
    - Ex: process_llm_output(llm_reponse)
 2.) Router: LLM output determines BASIC control flow
    - Ex: if llm_decision(): path_a(); else: path_b()
 3.) Tool call: LLM output determine FCN EXECUTION
    - Ex: run_fcn(llm_chosen_tool, llm_chosen_args)
 4.) Multi-step agent: LLM output controls iteratons and prog. continuation
    - Ex: while llm_should_continue(): execute_next_step()
 5.) Multi-agent: one agentic workflow can start another agentic workflow
    - Ex: if llm_trigger(): execute_agent()

#### Benefits of code agent actions over tool-calling agent actions:
 - Easier to chain/parallelize actions
 - Assign variables for reuse
 - Manipulate non-text elements
 - Building own tools through fcns
 - Succint execution
  i.) Whereas tool-calling agent requires calling EACH tool individually and separately
    a.) EACH new step for the SAME TASK => you have more latency, costs, and error-chances

#### Note on dataframe:
 - You'll need to feed in a dicts. with diff. keys
 - The dict. keys are the COLUMN names
 - Each key will be assoc. with a LIST of diff. values
  i.) The LENGTH of the list will be the NUMBER of rows
  ii.) EAch key's list MUST be of same length
    a.) If not, it'll be assumed that the missing values are "NaN" 
       * This assumption can be configurd in the datagframe setup
 - The ordering of the lists also matter
  i.) I.e., all list's 1st elem. will be associated with the first row
    a.) All list's Nth elem. will be associated with the Nth row
 - When a fcn is applied to entire DF, it can either be applied to COLS. or ROWS
  i.) Regardless, it would be applied to EACH (col./row) in an INDIVIDUAL MANNER
  ii.) The fcn is applied to entire DF through "dataframe.apply(fcn, axis=)"
     a.) The "axis" param. dictates whether it would be applied to cols./rows
     b.) REcall: "axis=0"=> apply to cols.; "axis=1"=> apply to rows
 - The use of "axis" param. dictates whether a fcn is applied to the cols. or the rows
  i.) "axis=0": default param., applied fcn to EACH col.
  ii.) "axis=1": applied fcn to EACH row
 - Recall dictionary formatting of dataframe:
  i.) Keys = columns => keys = 0th axis
  ii.) values = rows => values = 1st axis

#### Secure code execution overview:
 - Agents executing python code creates a RISK of arbritrary python execution
 - There are ways that malicious code can be executed:
  i.) LLM errors: LLMs can execute malicious code when trying to help out
  ii.) Supply chain attack: you accidentally run a malicious LLM
  iii.) Prompt injection: while web browsing, agent can come across malicious website
     a.) This website might give the agent harmful/malicious instructions
 - Neg. effects of malicious code/prompt from an agent:
  i.) Harm file system
  ii.) Steal data
  iii.) Abuse resources
  iv.) COmpromise network
  v.) INstall malware/backdoor to be use in a LATER attack
 - PYthon executor safeguard details:
  i.) Custom python interpreter built in by HF
  ii.) Code agent execution is NOT performed by VANILLA python interpreter
  iii.) Custom interpreter works by loading the AST (Abstract Syntax Tree)
      a.) Custom interpreter executes code operation-by-operation
  iv.) Non-implemented behaviors/code FAILS
  v.) By default: imports are DISALLOWED
     a.) UNLESS they've been EXPLICITLY added to an authorization list
     b.) This includes innocuous modules
  vi.) If code agent executes infinite loop, it'll eventually throw an error
 - Sandbox environments are used ON TOP of custom python interpreters
  i.) Adds in more safety guardrails

#### Custom, safeguard executor needs to be IMPORTED
 - pip install git+https://github.com/huggingface/smolagents.git

#### There are 3 main rules that are built-in safeguards
 1.) Any UNDEFINED COMMAND is IGNORED
 2.) Imports are SECURE
 3.) INfinite loops are PREVENTED - there is a count limit

#### Sandbox details:
 - Remote sandboxes are the best way to secure LLM-generated code execution
 - Only AFTER agent generates code SNIPPETS should you send them to sanbox for execution
 - This s'ltn is LIMITED if you want to do MULTI-AGENT runs
  i.) Since code is NOT entirely ported on the sandbox, you CANT run MAS on it
 - Export EVERYTHING on your sandbox: code, API keys, tools, etc.
  i.) Enables MASs in remote sandboxes, but FORCES you to export sentitive API keys

#### Smolagent allows you to use one of the 2 following options:
 1.) LOCAL Docker containers
 2.) E2B sandboxes
  i.) Requires E2B API key
  ii.) MAS execution is possible, but required CUSTOM setup

#### Steps to run agent in a sandbox:
 1.) Get the E2B API key
 2.) Create customs tool to use in the sandbox
 3.) Create code agent
   -) Pass in: custom tool, executor type, and e2b api key

#### You can setup tracing in a local and remote manner, using "register()"
 - Local endpt: "get_phoneix_endpoint()" + "v1/traces"
 - Remote endpt: "os.getenv("DLAI_LOCAL_URL").format(port='6006')" + "v1/traces"
 - Tracing records ANY smolagent RELATED CODE that was EXECUTED
  i.) Record lateny, token usage, inputs, outputs, etc.
 - Usually goes along with "SmolagentsInstrumentor"

#### Diff. ways to monitor agents:
 - Check whetehr the APPRORIATE tool was called
  i.) Given a list of test inputs
 - CHeck whether any of the answers were hallucinations
 - Use LLM as a judge to rate the agent trace
  i.) Comparing the request and the expected answer

#### Notes on creating tools for code agents:
 - These tools MUST be related to the probelm at hand
 - Recall: safeguard 1 - undefined commands are ignored
 - The code agent uses these tools as a STARTING point
  i.) It can then create additional tools that build off these base tools
 - BUT these base tools MUST be created by the developer

#### Steps to get a smolagent's state history (in dataframe format)
 1.) Create a project name (line 734)
 2.) Setup a tracer (line 741)
 3.) Setup a "SmolagentsInstrumentor()" (line 746)
 4.) Create the code agent (line 842)
 5.) Send the project name into a phoenix client (line 1374)
    - This returns a dataframe
 6.) Check out the phoneix DATAFRAME (line 1375)
    - Will contain the following keys/cols:
      i.) "context.span_id" = id for the current snapshot
      ii.) "name" = name of the snapshot
      iii.) "span_kind" = what the snapshot consisted of
      iv.) "parent_id" = id of the parent snapshot
      v.) "start_time" = time when snapshot was executed
      vi.) And A LOT MORE

#### Multiple snapshot that use same tool for same request - details
 - From these line we can see that one request can involve multiple snapshots
  i.) This is normal, since diff. tool executions might be needed to solve the request
 - What's weird about this though is that some of those snapshots use the SAME TOOL
  i.) I.e., there are duplications of the same tool-calling for the same request
 - line 1564: this is why we need "set", to erase these duplications
 - BUt, i'm still not sure why there are repetitions of the same tool call for same reques

#### Steps to evaluating a code agent
 1.) Setup tracer and agent
 2.) Create an object that has sample inputs and the CORRECT tools to use with them
 3.) Configure tracer dataframe to hold the data you want to evaluate
 4.) Creating a bool fcn to see whether agent execution matches expected execution
 5.) Running bool fcn on (2) and (3) above
 6.) Printing out the final evaluation

#### smolagents and fixing errors:
 - Errors get apprended to the agent's mem
 - The agent attempts to fix its own mistake
 - IF a single agent REPEATEDLY FAILS to accomplish the task on multiple iterations, then:
  i.) Break down the task into sub-tasks
  ii.) Assign a diff. agent to EACH sub-task

#### BEnefits of MAS:
 - Soecialize EACH agent
  i.) By the tools/models chosen, you can SPECIALIZE an agent's CORE TASKS
 - Separate mems. b/w DIFF sub-tasks
  i.) REduces the counts of input tokens at each step => reduces latency and costs




## Course 12: Model Context Protocol w/ Anthropic

#### MCP details:
 - Defn: STANDARDIZES how LLMs acces tools, data, prompts from EXTERNAL sources
 - It's an open protocol
 - Based on client-server architecture
  i.) MCP Client (MCPC): hosted INSIDE the AI app itself
  ii.) MCP Server (MCPS): exposes tools, resources, and prompt templates
 - There are 2 ways to launch MCPS:
  1.) Locally, as a subprocess LAUNCHED by the client
  2.) Remotely, as an independent process
 - MCP is MODEL-AGNOSTIC
 - Can handle API calls and tool executions
 - Can be built with Human-In-The-Loop (HITL) features
  i.) Based on the INTERFACE that the HOST DEVELOPS
   a.) MCPS simply sends back info, which is CONTEXT for an AI app
 - Uses natural language to talk to MCPS resources W.O./ having to write the logic ourself
 - MCPS can be used with ANYTHING that is MCP compatible
  i.) Thus, MCPS are REUSABLE across DIFF. APPLICATIONS
 - SDKs that power MCP are written across many diff. programming langs.

#### MCPS vs tool/API calling:
 - MCPSs provide tool schemas and fcns ALREADY DEFINED for you
 - If you want to directly call an API, you'll be AUTHORING those on your own
 - MCPSs are like wrappers on top of an API
 
#### MCP client-server architecture:
 - MCPCs have a 1-1 connection with MCPSs
 - MCPC and MCPS COMMUNICATE through MESSAGES defined by the MCP itself
 - MCPC lives INSIDE HOSTS
 - Hosts are RESPONSIBLE for storing and maintaining ALL of the MCPCs
  i.) As well as the connections with MCPSs
  
#### General terminology:
 - Host = LLM app that wants to access data through MCP
  i.) ExL Claude desktop, IDEs, AI agents, etc.
 - MCPS = LIGHTWEIGHT programs that EACH expose specific capabilities
 - MCPC = maintain a 1-1 connection with MCPS; lives inside the host app
 
#### MCPSs expose the following:
 - Tools = fcns and tools that can be INVOKED by the MCPC
  i.) Meant for data that might require a POST request, modification, etc.
  ii.) Ex: retrieve/ search, send a msg, update DB records, etc.
 - Resources = READ-ONLY data/context exposed by the MCPS
  i.) App can use these resources W.O./ having to bring it into context
  ii.) Ex: Files, DB records, API responses, etc.
 - Prompt templates = pre-defined templates for prompt interactions
  i.) REmoves the burden of prompt engineering from the user
  ii.) Templates live INSIDE the MCPS
  ii.) Ex: Document Q&A, transcript summary, output as JSON, etc.

#### Overview in creating tools, resources, and prompts in MCP:
 - Tools
  i.) Decorate a fcn with "@mcp.tool()" decorator
  ii.) Provide a docstring to fcn to elaborate on tool's fcnality
 - Resources
  i.) Done by SPECIFYING a URI (similar to URL) where MCPC can find the data
   a.) Similar to GET request handlers in a typical HTTP Server
  ii.) Decorate fcn with "@mcp.resource()" and fill in necessary params.
    a.) Fcn will RETURN desired DATA when resource is accessed
  iii.) Can return ANY TYPE of data: string, JSON, binary, etc.
     a.) You can set the "@mcp.resource(mime_type=...)"
        * This gives the MCPC a HINT as to what data the server is returning
  iv.) There are 2 ways to create MCPS resources:
    a.) Direct code: fcn doesn't have additional params; fcn is hard coded
    b.) Templated code: fcn has additional params, which are used in the fcn logic
  v.) MCPS simply sends data BACK to MCPC and app CHOOSE/DECIDE whether to use data or not
 - Prompts
  i.) Created using "@mcp.prompt()" deocrator on a fcn
   a.) Reqires "name" and "description" params. in decorator
   b.) Fcn returns a LIST of msgs/text
  ii.) Defines a SET of User and Assistant msgs that can be used by MCPC
  iii.) Prompts should be HIGH-QUALITY and WELL-TESTED
  
#### Steps in the communication lifecycle b/w MCPC and MCPS:
 1.) Initialization
    - MCPC sends initialization request
    - Then, MCPS sends intitialization response
    - Finally, MCPC sens initialized notification as confirmation
 2.) Msg exchange
    - The following exchanges are allowed:
     i.) MCPC sends request to MCPS -> MCPS sends response
     ii.) MCPS sends request to MCPS -> MCPC sends resposne
    - Notificaiton can be send back and forth b/w MCPS/C
 3.) Termination
    - Connections are terminated

#### MCP Transports (MCPT) details:
 - Handles underlying MECHANICS of HOW msgs are sent and received b/w client and server
 - There are built-in MCPTs, but you can also MAKE YOUR OWN MCPT
 - There are 2 common built-in MCPTs:
  i.) For LOCAL MCPS: stdio (standard input-output)
  ii.) For REMOTE MCPS: there are 2 popular options:
     a.) HTTP+SSE (Server Sent Events), protocol version 11/5/2024
     b.) Streamable HTTP, protocol version 3/26/2025
 - Diff. between both remote options:
  i.) HTTP+SSE: you MUST open a STATEFUL CONNECTION
    a.) It MUST maintain a backend port that is OPEN
    b.) Does NOT WORK for STATELESS deployments
  ii.) Streamable HTTP: allows for STATEFUL and STATELESS CONNECTIONS
    a.) BUT not yet supported across all SDKs

#### Details of the 3 common, built-in MCPTs:
 1.) Stdio Transport
    - USed when MCPS is running locally
    - MCPC launches the MCPS as a SUBPROCESS
    - MCPS reads and writes alongside MCPC w/ stdio
    - MCPC (<-- stdio transport -->) Local MCPS
    - Steps in this process:
     i.) MCPC launches the MCPS as a SUBPROCESS and initiates conversation
     ii.) MCPC writes to MCPS stdin stream
     iii.) MCPS writes to MCPC stdout stream
     iv.) MCPC closes stdin and terminates subprocess
 2.) HTTP+SSE
    - Client and server do NOT CLOSE CONNECTIONS b/w diff requests
    - Since connections are STATEFUL, then:
     i.) Data can be shared, sent, and remembered
    - MCPS can send back EVENTS and MSGS to the MCPC
    - MCPC (POST HTTP -->) (<-- SSE) Remote MCPS
 3.) Streamable HTTP
    - SUpports stateful (HTTP+SSE) and Stateless (HTTP) connections
    - For SCALING, it's efficient to have servers that are ephemeral/stateless
     i.) I.e., EACH CONNECTION and REQUEST is DIFF and NOT REMEMBERED
    - In future, this will be the STANDARD for remote MCPS
    - MCPC (POST HTTP -->) (<-- HTTP or SSE) Remote MCPS
    - Uses HTTP's GET and POST request extensively
    - Steps in this procedure:
     i.) MCPC sends "POST/mcp" to intialize request
     ii.) MCPS sends and initialized reponse
     iii.) Optional (for STATEFUL activation): MCPC send "GET/mcp" w./ "Accept:text/event"
         a.) MCPS can now send msgs to MCPC
     iv.) MCPC sends "POST/mcp" with request
     v.) MCPS sends HTTPReponse object
     vi.) Optional (for STATELESS): MCPC sends "DELETE/mcp" to terminate sesssion
        a.) MCPS terminates session

#### In reference to tools, there are 2 main MCPC requests to the MCPS:
 1.) "ListToolsRequest" = list all the tools available in MCPS
 2.) "CallToolRequest" = Execute a particular tool with args given by MCPC

#### There are 2 ways to create a MCPS:
 1.) Low-level implementation
  i.) YOu directly define and handle various types of requests
    a.) Ex: "ListToolsRequest", "CallToolRequest", etc.
  ii.) Allows you to CUSTOMIZE EVERY ASPECT of the MCPS
 2.) High-level implementation
  i.) Uses "FastMCP"
  ii.) "FastMCP" = high-level interface
     a.) MAkes building MCPS faster and simpler
  iii.) YOu simply focus on defining the tools as functions
     a.) "FastMCP" handles ALL protocol details

#### Steps to setup environment to test MCPS- working with "uv":
 1.) Open terminal
 2.) CD into folder where MPCS is at
 3.) Start working with MCP through package manager
    - Use package manager "uv" instead of "pip"
    - You'll also need "node.js" to replicate locally
 4.) Initialize the project folder: "$uv init"
 5.) Create virt. environment: "$ uv venv"
 6.) Activate virt environment: "$source .venv/bin/activate"
 7.) Install the necessary dependencies
    - MCP dependencies: "$uv add mcp"
    - Arxiv dependencies: "$uv ad arxiv"
 8.) Use MCP INspector to test MCPS
    - Tests MCPS in browser

#### MCPInspector details:
 - It's a browser-based environment to explore the MCPS tools, resources, prompts, etc.
 - IT creates a SANDBOX to play with MCPS W.O./ needing to build a MCPC/host
 - Can also run tools W.O./ needing to set tests through PYthon code and w.o./ LLM
  i.) Simply tests tool's fcnality

#### Steps to use MCPInspector:
 1.) Open terminal, CD MCPS folder, activate virt. environment, install dependencies
    - Essentially, ensure the steps in above bullet are complete
 2.) Run: "$npx @modelcontextprotocol/inspector uv run <MCPSPYthonFile>"
    - "npx" @modelcontextprotocol" = pulls in the cmd to start MCPS
      i.) That way MCPS does NOT have to be installed locally
    - "uv run <MCPSPythonFile>" = actually runs the MCPS
    - At the end, a browser link will show up
 3.) Go to browser and past in link
 4.) Fill in the sidebar configurations accordingly
    - Transport = "STDIO"
    - Command = "uv"
    - Arguments = 'run <MCPSPYthonFile>'
 5.) Click "connect" and you'll be connected to MCPS

#### Steps to connect MCPC with MCPS:
 1.) IMport necessary libraries
 2.) Establish the MCPS and the params necessary that we want to connect to
    - Will need to include terminal code (prev. bullet) for runninf the MCPS
     i.) Namely "uv run <MCPSPyhtonFile>"
     ii.) Will let MCPC know HOW to START the MCPS
 3.) Establish the connetion and launch the MCPS as a subprocess
    - YOu DONT want this step to BLOCK the execution of the rest of the host code
     i.) Thus, make us of "async" and "await" Python keywords
    - Setup a context manager to do the following:
     i.) PAss in the params. from our MCPS
     ii.) Establish a connection as a SUBPROCESS
    - ONce MCPS is connected, you'll get access to "read" and "write" streams
     i.) REcall, these are used for communication b/w MCPC and MCPS
    - Pass the "read" and "write" streams to a HIGHER level class, aka "ClientSession"
     i.) Gains access to an underlying CONNECTION that allows us to use fcnality for:
       a.) Listing tools
       b.) Initializing connections
       c.) etc.
    - Initialize session (& handshake) from the "ClientSession"
     i.) Afterwards, connection is COMPLETE
     
#### Relationship b/w MCPC and LLM-app file:
 - Recall: in the first code we created the simple-agent chatbot
 - The fcns inside this app will be WRAPPED INSIDE a diff MCPC python file
  i.) Specifically, the fcns will be WRAPPED INSIDE a "MCP_ChatBot" class
     a.) "MCP_ChatBot" class will be inside the MCPC file
     b.) This "MCP_ChatBot" will also contain fcns that will connect to MCPS
     
#### MCPC details:
 - MCPC python file MUST be separate than the MCPS file
 - MCPC's job is to QUERY for tools and pass them to the LLM
  i.) If tool needs to be used, MCPS does the tool invocation
 - Upon llm-app initialization, there is NOT a current session => no tool available
  i.) This changes once a connection is established through "ClientSession"
  ii.) Established session is NEEDED in order to go to MCPS and execute tools
 - MCPC LAUNCHES the MCPS => you DONT NEED 2 terminals
  i.) Recall: MCPC launches the MCPS as a SUB-PROCESS
 
#### Steps to run MCPC from terminal:
 1.) Open terminal, CD into MCPC folder, and activate virtual environment
 2.) Install necessary dependencies
    - "$uv add anthropic python-dotenv nest_asyncio"
    - "nest_asyncio": allows diff OSes to work PROPERLY with python event loop
 3.) Run the MCPC: "$uv run <MCPCPYthonFile>"
    - Once the cmd runs, the following will occur:
     i.) Connect to MCPS
     ii.) Make use of tools defined in MCPS
     iii.) Pass through tools to target LLM in MCPC file
     iv.) Create a terminal interface to talk with LLM


#### Remote MCPS details:
 - ANY DATA SOURCE you can imagine connecting to probably already has a MCPS
 - If MCPS in Python => use "uvx" or "npx" to download dependencies and establish connect
  i.) If MCPS NOT in PYhton, it'll probably use a non-uv command
 - ALL remote MCPS have installation instructions
 - Each MCPS has a bit of configuration required
 - To interact with remote MCPSs, create a small JSON file
  i.) MCPC will read JSON file and figure out commands for interactions
  ii.) BUT MCPC code will need to be updated so that it reads from JSON file
  iii.) JSON file will configure HOW we want to CONNECT to EACH individual MCPS
  
#### Connecting to multiple MCPS - overview
 - REcall:EACH MCPC has a 1:1 connection with a MCPS
  i.) One connection == one session
 - Host will need to keep a list of:
  i.) ALL tools (from ALL MCPS)
  ii.) The session corresponding to EACH tool
    a.) Will require a MAP b/w tools and "ClientSession"
 - Multiple connections will require multiple context managers 
  i.) Will require "AsyncExitStack()" to manage all connections
    a.) This is a STACK, as the name says
 - "AsyncExitStack" will manage:
  i.) Our connections for reading and writing
    a.) I.e, the "read" and "write" STREAMS for ALL connections
  ii.) The ENTIRE "ClientSession" connection to EACH session
 - In order to close any connection, use the session's corresponding CONTEXT MANAGER
  i.) Context manager will be found IN the "AsynExitStack"
  ii.) Call ".aclose()" on the targeted connection
  
#### "uv" vs "uvx" vs "npx":
 - "uvx" and "npx" allow users to run PYthon packages WITHOUT installing them
  i.) i.e., this is useful for accessing REMOTE MCPS
 - "uv" is used for python files that are LOCAL
  i.) USeful for accessing LOCAL MCPS
  
#### Sample JSON file that lists ALL MCPSs you want to connect to:
 - List of MCPSs:
  i.) "fetch": provide the "fetch" tool
    a.) Fetches a URL from the internet and EXTRACTS its content as MARKDOWN
  ii.) "filesystem": provides several tools for interacting with files and dirs
    a.) These files/dirs will be within a dir. that YOU SPECIFY
  iii.) "research": the custom MCPS created in second code
    a.) Searches for research papers in arxiv
 - Note the use of "uv", "uvx", and "npx"
  i.) "uvx" and "npx" will REACH the remote MCPSs
    a.) THey will use the URI found under "args"

#### Resources in MCPSs - details:
 - Resources are READ-ONLY data that MCPS can expose to LLM app
 - They are similar to GET endpts in REST API
  i.) They provide data
  ii.) BUT shouldn't perform significant computation or have side-effects
 - Ex: resource can be a LIST of folders in a dir or CONTENT of a file in a folder
 - They are created with "@mcp.resource(uri=.., mime_type=...)" decorator
 - URIs usually have the following scheme:
  i.) "something://xyz/xcv"
 - There are 2 types of URIs:
  i.) Static: URI is hard coded and doesn't change
   a.) Ex: "papers://folders"
  ii.) Dynamic: URI can change based on params. fed to it
    a.) Ex: "papers://{topic}"

#### Requests associated with resources and prompts:
 - "ListResourceRequest": list of all resources MCPS can provide
 - "ListPromptsRequest": list of all prompts MCPS can provide
 - "PromptGetRequest": actually retrieves the prompt

#### Local MCPS vs Remote MCPS - code changes:
 - The code for local and remore MCPS are ALMOST IDENTICAL
 - There are only 2 changes required to the SAME code:
  i.) Add a port to "FastMCP" param.
    a.) "mcp = FastMCP("research", port=8001)"
  ii.) Add a "transport" param. to "mcp.run"
    a.) "mcp.run(transport='sse')"
    b.) This is the type of the MCPT you're planning to use
    c.) See "MCP Transports (MCPT) details" bullet for more details
 - You can test the MCPS using a MCPInspector

#### Steps to deploy MCPS using render.com; how to start git repo:
 1.) Open terminal and cd to remote MCPS project folder
 2.) Intialize an empty git repo: "$git init"
 3.) Ensure that ".venv" folder is NOT included when adding/committing
    - "$ echo ".venv" > .gitignore"
    - Verify with: "$git status"
 4.) Convert "uv" dependencies into "pip" dependencies
    - Render does NOT currently SUPPORT "uv"
    - Run "$uv pip compile pyproject.toml > requirements.txt"
     i.) Recall: "pyproject.toml" is where "uv" dependencies are at
     ii.) This code turns dependencies into pip-compatible and then store them in txt file
 5.) Ensure you have the RIGHT version of PYthon for Render to use
    - Run: "$echo "python-3.11.11">runtime.txt"
 6.) Verify text files have been successfully added
    - "$git status"
 7.) Add and commit changes to git repo
    - "$git add; git commit -m "ready for deployment""
 8.) Create a Github repo
    - Needed for Render to pick up and load project files
    - Can be done on github browser
 9.) Upload local project to newly created github repo
    - "$git remote add origin <githubURL>"
     i.) URL found on github browser
 10.) Verify successul upload
     - "$git remote -v"
 11.) PUsh project to github repo
    - "$git push origin main"
    - AFter this, you should see changes in github repo
 12.) Go to "render.com"
     - Login/signup, click "dashboard", click "deploy web service"
 13.) Choose the target github repo and use "Start command"
     - Under "Start Command" input "python <MCPSPythonFileName>"
 14.) Make sure to choose FREE plan, then deploy
     - It will provide you with base URL
 15.) Verify successful deployment
     - Browser: "remote-research.onrender.com" => not Found 404
       i.) This is EXPECTED, since there is not front end to this MCPS
     - Browser: "remote-research.onrender.com/sse" => returns session ID
       i.) This CONFIRMS successful deployment
     - You can also ping to the base URL given above

#### Updating MCPC to work with remote MCPSs:
 - You'll need to import the appropriate MCPT library
  i.) SSE: "from mcp.client.sse import sse_client"
  ii.) Streamable HTTP: "from mcp.client.streamable_http import streamablehttp_client"
 - These new libraries will need to be integrated when into MCPS connection
  i.) specifically, when you're CREATING "ClientSession"
  ii.) "await self.exit_stack.enter_async_context(sse_client(url=...))"
  iii.) "await self.exit_stack.enter_async_context(streamablehttp_client(url=...))"
  iv.) These replace: "...enter_async_context(stdio_client(server_params))" goes


## Course 13: Agent Communication Protocol

#### ACP details
 - It's an open protocol that standardizes communication b/w agents
 - Provides unified interface through which agents can collaborate
  i.) REGARDLESS of their frameworks
 - Makes it easy to replace an agent w/ a NEW VERSION
  i.) WO/ needing to REFACTOR the ENTIRE system
 - Based on client-server architecture
  i.) HOst an agent (of ANY framework ) inside an ACPS
  ii.) ACPC connects to agents by sending requests to ACPS
    a.) ACPC MUST connect to ACPS
  iii.) Client is responsible for intiating communication
    a.) Server responds to this request
 - ACP agent lifecycle:
  i.) configuration -> activation -> discovery -> execution
 - ACP provides a UNIFIED REST interface
  i.) ACPC communicate with ACPS agent through REST requests
 - NOTE: ACPC can be an AGENT or a PROCESS
 - ACPC discovers agents using the ACPS endpts
 - Standardization makes for easy integration of agents 
  i.) Even into existing MASs
 - Uses HTTP to connect standalone agents
 - Supports human-in-the-loop collaboration
 - ACPCs or ACPSs can be: Ai agent, human, or a microservice
 - DEsigned with PRODUCTION-GRADE environment in mind
  i.) Prioritizes security, scalability, and observability for reliable performance
 
#### ACP benefits:
 - Chain ACP-compliant agents in linear and hierarchical workflows
  i.) Use ROUTER AGENT to delegate tasks to SPECIALIZES agents
 - Import ACP-compliant agents into a registry to make them easy to discover and share
 
#### Steps to build ACP compliant agents:
 1.) WRap the agents in an ACPS
 2.) Launch ACPS to ACTIVATE the agents
 3.) Make agents discoverable by ACPCs 
   - Enables easi integration within MASs
   
#### 4 patterns made possible by ACP
 1.) Dynamic updating
    - An agent can be swapped for another ACP agent
    - Makes syst. truly interoperable
    - Can easily swap old agents with new agents
    - Can be used to easily TEST diff. agent combinations
     i.) Used to identify which combo gives your syst. the best performance
 2.) Specialized agents can work as a TEAM
    - Pre-configured agents can collaborate
     i.) WO/ needing to build one MONOLITHIC multi-agent architecture
    - Instead of building one GIANT agent, ACP allows teams of specialized agents
     i.) These specialized agents can collaborate dynamically
     ii.) Agents can handoff tasks to one another
 3.) Across company workflows
    - Agents that represent diff. systs/parts of the company can collaborate
    - Ex: a customer submits complaint about a delayed shipment
     i.) Cust. support agent can RECOGNIZE it needs assistance of inventory agent
       a.) Cust. supp. agent send request to inventory agent using ACP
       b.) EACH agent stays FOCUSED in its DOMAIN, but COLLABORATES with ACP
 4.) Inter-organizational workflows
    - Agents belonging to diff. organizations can collab on inter-company tasks
    - ACP enables SECURE collaboration

#### AI tech stack (from bottom layer to top layer):
 1.) Semiconductors (ex: Nvidia GPUs)
 2.) Cloud infrastructure (ex: AWS)
 3.) Foundational models (ex: GPT, GEmini, BERT, etc.)
 4.) Agentic orchestration (ex: CrewAI, ACP, LangGraph, HuggingFace, etc.)
 5.) Applications (ex: ANY AI application)

#### Steps to setup ACP communication:
 1.) ACPC discovers available ACPSs and then initiates a request
 2.) ACPC sends REST request to ACPS
 3.) ACPS, which WRAPS an agent, manages the request
    - Returns a response back to the client over REST

#### Steps in an ACP agent's lifecycle:
 1.) Configuration
    - Define agent's basic identity and capabilities
    - Can specify: name, description, metadata
    - Makes agent discoverable and usuable within the ACP ecosyst.
 2.) Discovery
    - Online discovery: happens when ACPSs are already running
     i.) Can be accessed through their API endpts
    - Offline discovery: happens at a higher level (ex: agent catalog/registry)
     i.) Agent detail is embedded in the agent package
       a.) Allows a user/syst. to discover agents WO/ requiring them to be running
    - Agents can be browsed, selected, and spawned when needed
 3.) Deploy
    - Deploy to activate and share agents with others
    - NECESSARY in order to ACTIVATE agent and enable it's ONLINE discovery
    - Deployment can be done with:
     i.) Built-in SDK server
     ii.) External server
    - Makes ACPS agents available for ACPC to work with
 4.) Execution
    - There are 3 execution modes:
     i.) Synchronous: ACPC waits until ACPS agent completes its run 
       a.) ACPC then returns a final result
     ii.) Asynchronous: ACPC does NOT wait for ACPS agent to respond
       a.) ACPC can carry on w/ other taks in the background
     iii.) Streaming: ACPS establishes a SSE (Server Sent Event) connection
       a.) PRovides real time updates as agent generates results
    - Regardless of execution mode, each run progresses through states
     i.) States, such as: in-progress, awaiting, termination (completed or failed)

#### RAG overview:
 - It chunks and vectorizes your data
 - PRocess:
  i.) Have a vector DB
  ii.) Query vector DB
  iii.) Bring back context
  iv.) Pass context to LLM
  v.) GEnerate output
 - Requires an embedding model
 - If there are not lots/large docs, then vector-embedded chunks stored local/in-mem.
 - Depending on framework, if you don't specify vector DB, then ChromaDB is used

#### Needed libraries to convert code to ACPS:
 - "collections.abc.AsyncGenerator"
  i.) Forms output type for ACPS
  ii.) It's a generator that cam ITERATIVELY return output from the ACPS
  iii.) IT's an ASYNC generator
 - "acp_sdk.models.{Message,MessagePart}"
  i.) Forms the format used to output the result of the agent
 - "acp_sdk.server.{RunYield,RunYieldResume}"
  i.) Form part of the async generator
 - "acp_sdk.server.Server"
  i.) Forms foundation of ACPS
 - "acp_sdk.server.Server.agent()"
  i.) A decorator for agent you want to make available in the ACPS
  ii.) Agent's docstring will serve as METADATA
 
#### ACPS details
 - Ideally, you want to ACPS to capture DYNAMIC prompts when a user calls the ACPS
  i.) User prompt can be passed as an agent fcn param
  ii.) YOu could potentially LOOP through and create MULTIPLE tasks
 - Since ACPS output is "AsynchGenerator", then agent fcn will ALSO need to be asynch.
  i.) We'll also need to kickoff agent in an asynch manner
  ii.) Since output is generator, use "yield" instead of "return"
 - Agent fcn's yields will be the diff MESSAGE parts from agent output
 - Since diff. ACPSs can exist, you'll need to run ACPS on SPECIFIC PORTS
  i.) ALSO, diff. ACPSs will run on DIFF terminals
 - You can have MULTIPLE AGENTS in ONE ACPS
  
#### "AsynchGenerator[RunYield,RunYieldResume]" details
 - This is the return typethat enables BOTH streaming responses and the "await" pattern
 - "AsynchGenerator" = asynch generator obkect that can be ITERATED with "await"
  i.) Also supports "await" operations
 - "RunYield" = the TYPE of values this generator YIELDS (sends out)
 - "RunYieldResume" = the TYPE of values this generator RECEIVES when RESUMED (sent back)
 
#### Steps to run ACPS locally
 1.) Define the ACPS's file dependencies
 2.) Specify necessary API keys in a ".env" file
 3.) Open terminal, CD to ACPS folder
 4.) Run "$uv run <ACPSPYthonFile>"

#### "Message" and "MessagePart" details:
 - Since ACPS is ASYNCH, then it will yield results in an asynch manner
 - Specifically, it will yield "MessagePart"s as they appear
 - BUT the final object type itself is the "Message" type
 - Thus, the agent output will need to be WRAPPED inside "MessagePart"
  i.) Then, "MessagePart" will need to be WRAPPED inside "Message"
  ii.) Finally, it is this "Message" that gets YIELDED back to ACPC
  
#### Steps to start a project with "uv"
 1.) Create a project folder
 2.) CD into project folder
 3.) Run "$uv init" - intiializes project folder structure
 4.) Run "$uv venv" - creates virt ennvironment
 5.) Add project dependencies: "$uv add dependcy1 dependecy2 ..."
 6.) Create project file itself: "$touch <fileNAme>"
 7.) Run "$uv run" - runs the project file

#### ACPS pre-reqs to running an ACPC:
 - The ACPS MUST be running in it's terminal
  i.) This allows the ACPC to connect to the ACPS
 - Multiple ACPS will have their OWN terminal
  i.) EAch ACPS you're trying to connect to must also be running

#### Steps to run an ACPC:
 1.) Define an ACPC "Client"
    - Connect to the RUNNING ACPS through it's URL
    - ACPS agent FCN name (ex:"policy_agent") will be the SAME in ACPS and ACPC
     i.) Fcn name is diff. than agent name
       a.) "policy_agent" != "insurance_agent"
       b.) "insurance_agent" is INSIDE "policy_agent"
     ii.) Fcn name is used to identify WHICH agent in ACPS you want to connect with
       a.) Rcall: one ACPS can have multiple agents
 2.) Run "Client.run_sync()" with "await"
    - This is going to be awaited
     i.) Specifically, the NAME of the ACPS fcn that is awaited is NAME of agent's fcn
    - This is a SYNCHRONOUS execution that WAITS until ACPS COMPLETES PROCESSING
     i.) After completion, this fcn returns a SINGLE RESPONSE
    - This fcn actually INVOKES/CALLS the agent inside the agent fcn in ACPS
     i.) Will create a KICKOFF in the ACPS terminal
       a.) I.e., the ACPS terminal will display the EXECUTION details of the agent
    - "Client.run_sync.input" = will hold the ACPC/user's query for ACPS's agent
 3.) Unpack the ACPS's agent package/return
    - Pseudocode: "Client.run_sync().output[0].parts[0].content"
    
#### Relationship b/w ACPC, ACPS, and multiple agents:
 - ACP allows for a 1-1 relationship b/w ACPC an ACPS
 - If there are multiple agents in one ACPS, then ACPC only needs 1 connection to it
 - If there are multiple agents in DIFF ACPS, then there needs to be diff. connections
 
#### Reasons to create diff ACPSs:
 - Diff. organizations should have diff ACPSs
  i.) Thus each/all agents in a given ACPS will be related to the corresponding organizat.
 - Diff. teams can also make use of diff ACPSs
 
#### ACPS common libraries imported:
 - "collections.abc.AsyncGenerator"
  i.) TYPE returned from agent fcn
 - "acp_sdk.models.{Message, MessagePart}"
  i.) Used to structure the OUTPUT send back
  ii.) USed from ACPS to ACPC and VICEVERSA
 - Agent's framework: tools, models, and agents
 
#### GEneral communication b/w ACPC and ACPS of diff frameworks - IMPORTANT:
 - ACPC sends inputs to ACPS by using "MEssage"
  i.) "Message" is a SPECIAL ACP type
  ii.) ACPC can send prompts to ACPS through this way too
 - ACPS unpacks ACPC's inputs
  i.) ACPS formats them according to DESIRED agent INPUTS
 - Use the ACPS agent's framework invocation/execution
  i.) USING the ACPC inputs as the execution PARAMS.
 - Essentially, ACPC and ACPS communication through "MEssages"

#### Sequentiall calling overview:
 - ACPS sequential calls can be done by PASSING CONTEXT from one agent to another
 - Agent calls will be CHAINED
  i.) I.e., ACPS will performs ACPS's agent call in a CHAINED manner
 - Recall: you need 1-1 connection b/w ACPC and ACPS
 - Passing context from one agent to another
  i.) The OUTPUT of one agent becomes the INPUT of the other agent
  ii.) Context will be passed through "MEssage" types
 - REcall: ALL ACP agents have inputs and outputs with "MEssage" types
  i.) "Message" is how ACP agents COMMUNICATE
  ii.) "Message" type needs to be unpacked to get desired content
    a.) Since "MEssage" has many attrs

#### Sequential calling and ACPC:
 - Sequetiall calling is done through ACPC's MAIN program
 - Main program will create N ACPCs for N ACPSs
 - Diff. agents in the SAME ACPS can be invoked using the same "Client.base_url"
  i.) Only the "Client.run_sync().agent" name will change, corresponding to target agent
  
#### Steps to perform sequential agent chaining from ACPC's main code:
 1.) Create diff ACPC connections, 1 for each ACPS
    - Ex: client1 = Client(base_url=...); ... ; clientN = Client(base_url=...)
 2.) Run the first client and store the output's results
    - Ex: run1 = await client1.run_sync(agent=..., input=...)
 3.) UNPACK the first client's output
    - Set it up as CONTEXT for the next agent
    - Ex: content = run1.output[0].parts[0].content
 4.) Pass the context to next agent by appending conetxt info into "run_sync(input=..)"
    - Ex: run2 = await client2.run_sync(agent=..., input=f"Context:{content}, <userQ>")
 5.) Repeat steps 2-4 until last agent's output, which is final answer
 
#### Passing in MULTIPLE prompts to SAME agent in one ACPC call:
 - Note: ACPS agents are capable of taking in many user queries/prompts in one ACPC call
 - The answer to each prompt will be reflected in "Client.run_sync.output" LIST
 - Ex:
  i.) run1 = client1.run_sync(agent=..., input=[<userQ1>, <userQ2>])
      prompt1Response = run1.output[0]
      prompt2Response = run1.output[1]
 - This can help complexify a MAS

#### Hierarchical chaining overview:
 - Router agent in ACPC can AUTOMATICALLY navigate the diff. ACPS by itself
  i.) Auto navigation based on how it THINKS to the best answer
  ii.) Router must be AWARE of available agents in ACPSs
 - Similar to sequential chaining
  i.) BUT, instead of doing it MANUALLY using prompts and client, router will automate
   a.) Automatization happens with "ACPCallingAgent", explained below
  
#### Libraries needed for hierarchical chaining - "AgentCollection" and "ACPCallingAgent":
 - "fastacp.AgentCollection"
  i.) structures ACP agents into USABLE format
 - "fastacp.ACPCallingAgent"
  i.) Chooses which ACPS agent is needed to answer user Q
  ii.) Breaks up user Q into DIFF PROMPTS
    a.) Diff prompts CAN be fed to diff ACPS agents
  iii.) Uses JSON-like ACP agent calls
     a.) Similar to how "ToolCallingAgent" uses tool calls
        * But directed at REMOTE ACP agents instead of LOCAL tools
  iv.) Has the following args:
     a.) "acp_agents"= dict[str,Agent]
        * ACP agents that can be CALLED
     b.) "model" = Callable[[list[dict[str,str]]], Message]
        * model that will GENERATE a given agent's ACTIONS
     c.) "prompt_templates" = [Dict[str, str]]
        * Optional prompt templates
     d.) "planning_interval" = int
        * optional interval at which given agent will run a planning step
        
#### Steps to perform hierarchical workflows:
 1.) Agents hosted on EACH ACPS are discovered by their corresponding client object
    - Done through "client.agents()"
 2.) Discovered agents are converted to TOOLS for the router agent
 3.) Router agent breaks down user Q into SMALLER steps
    - Each step can be executed by the specialized agent
 4.) For a given step, router uses client of the specialized agent to send request to it
    - Agent is executed using "client.run_sync()"

#### Integrating MCP overview:
 - ACP and MCP can work together
 - BUT ACPS and MCPS MUST be in separate files
  i.) They'll communicate through stdio

#### Libraries needed to integrate MCP into ACP:
 - "smolagents.ToolCollection"
  i.) Allows us to discover tools on MCPS
 - "mcp.StdioServerParameters"
  i.) Sets up params. to connect with MCPS
  ii.) See C12VI (Connecting to MCPS)
  
#### "smolagents.ToolCollection" details:
 - "ToolCollection.from_mcp.trust_remote_code"
  i.) boolean that allows us to access tools in MCP
 - "ToolCollection.from_mcp.tools"
  i.) list of ALL found tools








## Course 17: Pydantic for LLM Workflows

#### Pydantic model basics:
 - Can be used to validate:
  i.) User inputs, LLM responses, tool-calling params, etc
 - Help provide a structured output for LLMs
  i.) Help you pass data from LLMs to other components in the pipeline in PREDICATBLE way
 - Can be used DIRECTLY in API calls 
 - CAn turn user's natural language inputs into STRUCTURED queries
 - Genreally: can be used for DATA VALIDATION in ANY SW syst.
  i.) Data validation is at the CORE of ANY SW app
 - They provide field names and their associated data types
  i.) Ex: "typing.Literal[a1,...,aN]"
   a.) => associated field name can only be ONE of {a1,...,aN}

   
#### Steps to integrate Pydantic models w/ tool-calling:
 1.) Define a pydantic model that specifies the PARAMS. for the tool-fcn
 2.) Have LLM produce response that follows the pydantic model's struct.
 3.) Use LLM's pydantic-output as INPUT ARGS. for the tool-fcn
    - Can pass in pydantic mode's NAME as the arg. for the tool
     i.) Similar to using "**kwargs"
       a.) Since pydantic model outputs are python DICTIONARIES
 4.) If needed: defined tool you can use in your LLM API call
 
#### PYdantic basics -FULL OF GOOD INFO:
 - When there are data validation errors, pydantic states WHERE error happens
 - Pydantic has many built-in data types (EX: "pydantic.EmailStr")
  i.) BUT you can define CUSTOM data types too
 - Pydantic models are DICTIONARIES
  i.) REcall: dicts. can be converted into JSON-format
 - Python's "None" == JSON's "null"
 - Sps an INPUT has EXTRA FIELDS not found in the pydantic model
  i.) Pydantic IGNORES these extra fields
  ii.) Pydantic only validates DEFINED FIELDS
 - Pydantic performs data type coercion
  i.) I.e., pydantic CONVERTS the JSON representation into PYTHON representation
  ii.) Pydantic has many representations where coercion happens AUTOMATICALLY
    a.) Ex: pydantic can handle a STRING representation of an int and do autom. conversion
 - Pydantic data coercion can be CUSTOMIZABLE
  i.) If you want to have a SPECIFIC SET of FORMATS be acceptable for fields in your model
 - Data coercion can ALSO be TURNED OFF
  i.) If you want to be very strict about the format you're accepting for a FIELD
 - Pydantic models can inherit from OTHER pydantic models
  i.) Just like in PYthon inheritance
  ii.) When you inherit from other pydantic models, you get all attrs from that model
    a.) You can also add more attrs to the child class you're creating 
 
#### Relationship b/w JSON and Python:
 - Conversion b/w JSON format and Python dicts:
  i.) "json.loads": JSON string -> Pythong dictionary
  ii.) "json.dumps": Python obj./dict -> JSON string
    a.) PYthon obj MUST be in JSON-schema in order to do conversion
 - You can use pydantic on JSON-strs by first turning them to python DICTS.
  i.) Note: LLM can be prompted to return outputs in JSON-str
 - Thus, LLM outputs can be PROMPTED to return outputs in JSON-str by:
  i.) First turning LLM output to Python dicts.
  ii.) Then using pydantic to check the pydantic model
 
#### Pydantic code info - IMPORTANT:
 - "pydantic.BaseModel" = starting pt. for ANY pydantic model
  i.) Has built-in fcnality for data validation
 - "pydantic.ValidationError" = used to CATCH errors
 - "pydantic.BaseModel.model_dump_json()"
  i.) AUTOMATICALLY turns pydantic obj/dict. into JSON-str
  ii.) Returns error if JSON is BADLY formatted to begin with
 - "pydantic.model_validate_json"
  i.) Automatically validates JSON with pydantic WITHOUT needing to convert to Python
 - "pydantic.BaseModel.model_json_schema()"
  i.) Returns pydantic model in FULL-FLEDGE JSON-schema 
  ii.) Although it's in JSON-schema, output type is STILL python object
  iii.) You can provide this JSON-schema INSIDE LLM prompt
     a.) Gets better output to begin with
     b.) This is what MOST frameworks end up doing
 - "pydantic_ai"
  i.) Pydantic's AI framework
  ii.) Mainly used for working with pydantic model structing
  iii.) I.e., it's great for using LLMs to know how to structure a given output
     a.) Sps there is a cusomter complain
        * Pydantic agentic framework is good at structing that complaint into other models
 - "pydantic.field_validator"
  i.) a DECORATOR used to validate COMPLEX, CUSTOM fields in pydantic model

#### Steps to implement TRADITIONAL LLM-output validation:
 1.) Create pydantic model for target output structure
 2.) Create prompt that will give the INITIAL output structure
 3.) DEfine an error handling fcn
    - It'll check for LLM output validity
    - Use this error handling fcn on output from step 2
     i.) It'll return {valid, error}
 4.) Create fcn to RETRY prompt w/ error feedback
    - This fcn only activates if there is an error in step 3
 5.) If step 4 results in error, pass that error to step 3&4 again

#### Fcns to analyze COMPLEX data types - IMPORTANT:
 - "type(some_var).mro"
  i.) MRO = Method Resolution Order
  ii.) Prints out the INHERITANCE STRUCT. of "some_var"
 - "type(some_var).mro.__module__"
  i.) Prints out CLASS of the data type corresponding to "some_var"
 - "type(some_var).mro.__name__"
  i.) Prints out ACTUAL NAME of the DATA TYPE of "some_var"

#### Pydantic models and tools
 - Pydantic models can be used in the defn of tools
  i.) These tools can then be passed in your LLM API call
 - You'll need to use pydantic model's JSON-schema version in tool's params
  i.) Use "pydantic.BaseModel.model_json_schema()"
    a.) Turns pydantic model into JSON-string
 - Recall: tool's defns get passed into LLM's API call
 
#### Validating complex fields/attrs:
 - "pydantic.field_validaor" = decorator to validate complex fields
 - EX: sps "order_id" field MUST have format "ABC-12345"
  i.) YOu'll need to define a FCN that uses REGEX to validate "order_id"
    a.) This fcn MUST be decorated w/ "@field_validator('order_id')"
    b.) The fcn MUST BE INSIDE the pydantic model
 - Can be used to enforce SECURITY
  i.) Ex: ensuring there does not exist SQL injections

















































