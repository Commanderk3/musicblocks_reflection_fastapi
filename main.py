from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

import config
import json
from utils.prompts import mentor_config, general_instructions, generateAlgorithmPrompt, updateAlgorithmPrompt, generateAnalysis
from utils.parser import convert_music_blocks
from utils.blocks import findBlockInfo
from retriever import getContext

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

# Chat endpoint: thinking disabled for faster, conversational responses
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=config.GOOGLE_API_KEY,
    temperature=0.7,
    thinking_budget=0  # Disable thinking for chat
)

# Algorithm & analysis endpoints: thinking enabled for deeper reasoning
reasoning_llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=config.GOOGLE_API_KEY,
    temperature=0.7,
    thinking_budget=-1  # Dynamic thinking (model decides)
)

# request schemas
class QueryRequest(BaseModel):
    query: str
    messages: List[Dict[str, str]]
    mentor: str
    algorithm: str

class CodeRequest(BaseModel):
    code: str

class AnalysisRequest(BaseModel):
    messages: List[Dict[str, str]]
    summary: str

class CodeUpdateRequest(BaseModel):
    oldcode: str
    newcode: str

# response schemas
class AnalysisSchema(BaseModel):
    response: str
    
class AlgorithmSchema(BaseModel):
    algorithm: str
    response: str
    

@app.get("/")
async def root():
    return {"message": "Hello, Music Blocks!"}    

@app.post("/projectcode/")
async def projectcode(request: CodeRequest):
    code = request.code
    data = json.loads(code)
    flowchart = convert_music_blocks(data)
    blockInfo = findBlockInfo(flowchart)
    structured_llm = reasoning_llm.with_structured_output(AlgorithmSchema)
    answer = structured_llm.invoke(generateAlgorithmPrompt(flowchart, blockInfo))
    
    try:
        return {
            "algorithm": answer.algorithm,
            "response" : answer.response
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/updatecode/")
async def update_projectcode(request: CodeUpdateRequest):
    oldCode = request.oldcode
    newCode = request.newcode

    newFlowchart = convert_music_blocks(json.loads(newCode))
    oldFlowchart = convert_music_blocks(json.loads(oldCode))

    if (newFlowchart == oldFlowchart):
        print("No change detected")
        return {
            "algorithm": "unchanged",
            "response" : "No change detected"
        }

    blockInfo = findBlockInfo(newFlowchart)
    structured_llm = reasoning_llm.with_structured_output(AlgorithmSchema)
    answer = structured_llm.invoke(updateAlgorithmPrompt(oldFlowchart, newFlowchart, blockInfo))
    
    try:
        return {
            "algorithm": answer.algorithm,
            "response" : answer.response
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat/")
async def chat(request: QueryRequest):
    query = request.query.strip()
    raw_messages = request.messages
    mentor = request.mentor.lower()
    algorithm = request.algorithm

    if not query:
        return {"error": "Empty query"}

    messages: List[BaseMessage] = convert_messages(raw_messages)
     
    # Replace or insert system prompt
    system_prompt = mentor_config(general_instructions, algorithm, mentor)
    
    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(content=system_prompt)
    else:
        messages.insert(0, SystemMessage(content=system_prompt))

    # Add relevant context from RAG
    rag_context = getContext(query)
    if rag_context:
        messages.insert(1, HumanMessage(content=f"Relevant context:\n{rag_context}"))

    messages.append(HumanMessage(content=query))

    try:
        result = llm.invoke(messages) #invoking llm with messages, not a single query
        return {
            "response": result.content
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/analysis/")
async def analysis(request: AnalysisRequest):
    raw_messages = request.messages
    old_summary = request.summary
    structured_llm = llm.with_structured_output(AnalysisSchema)
    
    if not raw_messages:
        return {"error": "Empty query"}
    
    try:
        result = structured_llm.invoke(generateAnalysis(old_summary, raw_messages))
        return {
            "response": result.response
        }
    except Exception as e:
        return {"error": str(e)}

def convert_messages(raw_messages: List[Dict[str, str]]) -> List[BaseMessage]:
    converted = []
    for msg in raw_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "user":
            converted.append(HumanMessage(content=content))
        elif role in ["meta", "code", "music", "assistant"]:
            converted.append(AIMessage(content=content))
    return converted