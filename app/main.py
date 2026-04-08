# main.py

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from .schemas import ProposalRequest, ChatRequest
from mangum import Mangum # <-- NEW: AWS Lambda Adapter
from .graph import graph

app = FastAPI(title="AI Proposal Generator API")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate_proposal")
async def generate_proposal_endpoint(req: ProposalRequest):
    """One-shot proposal generation."""
    try:
        # Create a random thread ID for a one-off run
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        print("checking -----------------------------")
        prompt = f"Write a proposal for this job:\n\n{req.job_description}"
        if req.client_name:
            prompt += f"\n\nThe client's name is {req.client_name}."

        state = {
            "messages": [HumanMessage(content=prompt)],
            "job_description": req.job_description
        }
        print("checking -----------------------------xxxxxxxxxxxxxxxxxxxx")
        result = graph.invoke(state, config=config)
        print("checking -----------------------------")
        # Return the last generated AI message
        return {"proposal": result["messages"][-1].content, "thread_id": thread_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import HTTPException # Ensure this is imported

@app.post("/chat_proposal")
async def chat_proposal_endpoint(req: ChatRequest):
    try:
        is_first_call = req.thread_id is None or req.thread_id == ""
        active_thread_id = req.thread_id if not is_first_call else str(uuid.uuid4())
        config = {"configurable": {"thread_id": active_thread_id}}

        if is_first_call:
            if not req.job_description:
                raise HTTPException(status_code=400, detail="job_description is required for the first call.")
            
            prompt = f"Write a proposal for this job:\n\n{req.job_description}\n\n {req.message or ''}"
            state_update = {
                "messages": [HumanMessage(content=prompt)],
                "job_description": req.job_description
            }
        else:
            if not req.message:
                # This is the 400 error that was getting "swallowed" by the 500 block
                raise HTTPException(status_code=400, detail="message is required for follow-up edits.")
                
            state_update = {"messages": [HumanMessage(content=req.message)]}

        result = graph.invoke(state_update, config=config)
        
        return {"proposal": result["messages"][-1].content, "thread_id": active_thread_id}

    # 1. ADD THIS: Catch HTTPExceptions first so they don't hit the 500 block
    except HTTPException as http_err:
        raise http_err

    # 2. This now only catches actual code crashes (database down, syntax error, etc.)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")


# --- AWS LAMBDA HANDLER ---
# Mangum wraps the FastAPI app so AWS API Gateway knows how to talk to it.
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)