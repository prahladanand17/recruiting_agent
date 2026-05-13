import json
import os
from zipfile import MAX_EXTRACT_VERSION
from agent.tool_impl import get_recruiting_emails, search_emails, research_company, evaluate_fit, draft_response
import chromadb
from anthropic import Anthropic
import voyageai

class Agent:
    def __init__(self, name, prompt, tools, max_iter = 20):
        self.name = name
        self.prompt = prompt
        self.tools = tools

        self.anthropic_client = Anthropic()
        self.anthropic_model = "claude-sonnet-4-20250514"

        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="emails")

        self.voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))   
        self.voyage_model = "voyage-4-large"
        self.max_iter = max_iter

    def run(self, query, history=[], stream_callback=None):

        messages = history + [{"role": "user", "content": query}]

        print(json.dumps(messages, indent=2))

        for i in range(self.max_iter):
            full_text = ""

            with self.anthropic_client.messages.stream(
                model=self.anthropic_model,
                max_tokens=2000,
                system=self.prompt,
                tools=self.tools,
                messages=messages
            ) as stream:
                # collect tokens as they arrive
                for text in stream.text_stream:
                    full_text += text
                    yield text
                
                # stream finished — get final message for stop reason + tool calls
                final = stream.get_final_message()
            
                if final.stop_reason == "end_turn":
                    return
                
                # handle tool calls
                tool_calls = [b for b in final.content if b.type == "tool_use"]
                tool_results = []
                for tc in tool_calls:
                    result = self.dispatch_tool(tc.name, tc.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": json.dumps(result)
                    })
                
                messages.append({"role": "assistant", "content": final.content})
                messages.append({"role": "user", "content": tool_results})

        
        return "Max iterations reached — the query may be too complex. Try asking about fewer emails at once"

    def dispatch_tool(self, name, inputs):
        #Dispatch the tool to the appropriate function

        if name == "get_recruiting_emails":
            result = get_recruiting_emails(**inputs, collection=self.collection)
        elif name == "search_emails":
            result = search_emails(**inputs, collection=self.collection, client=self.voyage_client)
        elif name == "research_company":
            # no hard enforcement needed here
            result = research_company(**inputs, client=self.anthropic_client)
        elif name == "evaluate_fit":
            # hard enforcement — company_research must be in inputs
            if "company_research" not in inputs or not inputs["company_research"]:
                result = {"error": "company_research is required. Call research_company first."}
            else:
                result = evaluate_fit(**inputs, client=self.anthropic_client)
        elif name == "draft_response":
            result = draft_response(**inputs, client=self.anthropic_client)
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        print(f"   Result preview: {str(result)[:200]}")
        return result

    def stream_response(self, query, history=[]):
        response = self.run(query, history)
        for word in response.split(" "):
            yield word + " "



        