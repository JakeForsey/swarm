import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from swarm.vibevolve import (
    PROMPT,
    OPPONENTS,
    run_tournament,
    persist_in_history,
)

templates = Jinja2Templates(directory="templates")
app = FastAPI()


class CompletionRequest(BaseModel):
    completion: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "prompt": PROMPT,
            "completion": "",
            "results": "",
        },
    )


@app.post("/submit")
async def submit(completion_request: CompletionRequest, request: Request):
    completion = completion_request.completion
    results = run_tournament(
        completion,
        OPPONENTS,
        num_rounds_per_matchup=16,
        episode_length=128,
    )
    persist_in_history("dev", str(time.time()), None, time.time(), results, completion)
    for result in results:
        if result["name"] == "tmp_agent":
            result["name"] = "aggregate"
        else:
            result["reward"] = -result["reward"]
    results.sort(key=lambda x: -x["reward"] if x["name"] != "aggregate" else -2.0)
    formatted_results = "\n".join(
        [f"{result['name']:>15} {result['reward']:.2f}" for result in results]
    )
    return {"results": formatted_results}
