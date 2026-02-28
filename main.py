from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

import config
import json
from utils.prompts import mentor_config, general_instructions, generateAlgorithmPrompt, updateAlgorithmPrompt, generateAnalysis, generateConversationSummary
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

SUMMARIZE_THRESHOLD = 16  # 8 pairs of user+AI messages
KEEP_RECENT = 6  # 3 pairs sent in full next call

# request schemas
class QueryRequest(BaseModel):
    query: str
    messages: List[Dict[str, str]]
    mentor: str
    algorithm: str
    conversation_summary: Optional[str] = None  
    summarized_up_to: int = 0  

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
    conversation_summary = request.conversation_summary
    summarized_up_to = request.summarized_up_to

    if not query:
        return {"error": "Empty query"}

    messages: List[BaseMessage] = convert_messages(raw_messages)
     
    # Replace or insert system prompt
    system_prompt = mentor_config(general_instructions, algorithm, mentor)
    
    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(content=system_prompt)
    else:
        messages.insert(0, SystemMessage(content=system_prompt))

    if conversation_summary:
        messages.insert(1, AIMessage(
            content=f"[Summary of earlier conversation]: {conversation_summary}"
        ))

    rag_context = getContext(query)
    if rag_context:
        enhanced_query = f"[Relevant context: {rag_context}]\n\n{query}"
    else:
        enhanced_query = query

    messages.append(HumanMessage(content=enhanced_query))

    try:
        result = llm.invoke(messages)
    except Exception as e:
        return {"error": str(e)}

    response_data = {
        "response": result.content,
    }

    # Check threshold and summarize if needed
    summarize_result = check_and_summarize(raw_messages, conversation_summary, summarized_up_to)
    if summarize_result:
        response_data["conversation_summary"] = summarize_result["conversation_summary"]
        response_data["summarized_up_to"] = summarize_result["summarized_up_to"]

    return response_data
    
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

def summarize_messages(raw_messages,conversation_summary,summarized_up_to) :
    
    if len(raw_messages) <= SUMMARIZE_THRESHOLD:
        return None

    try:
        messages_to_summarize = raw_messages[:-KEEP_RECENT]

        formatted = "\n".join(
            f"{m.get('role', 'unknown').upper()}: {m.get('content', '')}"
            for m in messages_to_summarize
        )

        summary_result = llm.invoke(
            generateConversationSummary(conversation_summary, formatted)
        )

        new_summarized_up_to = summarized_up_to + len(messages_to_summarize)

        return {
            "conversation_summary": summary_result.content,
            "summarized_up_to": new_summarized_up_to
        }
    except Exception as e:
        return None

def convert_messages(raw_messages: List[Dict[str, str]]) -> List[BaseMessage]:
    converted = []
    for msg in raw_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "user":
            converted.append(HumanMessage(content=content))
        elif role in ("meta", "code", "music"):
            converted.append(AIMessage(content=content))
    return converted