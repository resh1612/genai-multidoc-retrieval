Reg no : 212224220080

## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

Extracting specific, nuanced information from a collection of dense academic papers is a slow and inefficient manual process. Standard search tools rely on exact keywords and fail to understand the conceptual context of a user's question. This program aims to build an AI agent that can intelligently query multiple documents to synthesize precise answers to complex questions.

### DESIGN STEPS:

#### STEP 1:

Load PDF documents and create specialized search and summary tools for each paper.

#### STEP 2:

Initialize an AI agent with an OpenAI model, giving it access to all the created tools.

#### STEP 3:

Query the agent with a specific question about one paper to get a detailed answer from its content.

### PROGRAM:

```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
```
```
import nest_asyncio
nest_asyncio.apply()
```
```
urls = [
    "https://openreview.net/pdf?id=kMuQBgPIdg",
    "https://openreview.net/pdf?id=CfZLxT3zIZ",
    "https://openreview.net/pdf?id=m5byThUSNE",
]

papers = [
    "Enhancing.pdf",
    "Fire.pdf",
    "Optimistic.pdf",
]
```
```
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
```
```
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
```
```
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
```
```
len(initial_tools)
```
```
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
```
```
response = agent.query(
    "Give me a summary of AIGB-Pearl and explain how KL-Lipschitz constrained score maximization works ,"
    "Explain successor features and zero-shot reinforcement learning in OpTI-BFM"
)
```
```
response = agent.query("What are the main contributions of the FIRE paper and why is DfI important?,"
                       "Compare the three papers Enhancing Generative Auto-Bidding with Offline Reward Evaluation and Policy Search, Optimistic Task Inference for Behavior Foundation Models, and FIRE in terms of objectives, methods, machine learning techniques, and real-world applications"
                      )
print(str(response))
```

### OUTPUT:

<img width="378" height="63" alt="image" src="https://github.com/user-attachments/assets/aa12f5b1-c161-488c-8b93-b65570963cf2" />

<img width="692" height="508" alt="image" src="https://github.com/user-attachments/assets/c5fb7b0e-32c0-494e-ba8c-c6c9c1eda715" />

<img width="676" height="221" alt="image" src="https://github.com/user-attachments/assets/1c56d6fe-6688-4467-b9a8-3f6c2ebf9316" />



### RESULT:

The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.
