import json
from datetime import datetime, timedelta


def embed(text, client=None):
    result = client.embed([text], model="voyage-4-large")
    return result.embeddings[0]

# ── Tool 1: Metadata filter ──────────────────────────────────
def get_recruiting_emails(days=None, sender=None, read=None, max_results=50, collection=None):
    """Fetch emails by metadata filters — no embeddings involved"""
    
    where_filters = []
    
    if days is not None:
        cutoff = (datetime.now() - timedelta(days=days)).timestamp()  # float not string
        where_filters.append({"date": {"$gte": cutoff}})
    
    if sender is not None:
        where_filters.append({
            "$or": [
                {"sender": {"$contains": sender.lower()}},
                {"subject": {"$contains": sender}}
            ]
        })
    
    if read is not None:
        where_filters.append({"read": read})
    
    # build final where clause
    if len(where_filters) == 0:
        where = None
    elif len(where_filters) == 1:
        where = where_filters[0]
    else:
        where = {"$and": where_filters}
    
    results = collection.get(
        where=where,
        limit=max_results,
        include=["documents", "metadatas"]
    )
    
    # deduplicate by email_id — return one entry per email not per chunk
    seen = {}
    for i, meta in enumerate(results["metadatas"]):
        email_id = meta["id"]
        if email_id not in seen:
            seen[email_id] = {
                "email_id": email_id,
                "sender_name": meta["sender_name"],
                "sender": meta["sender"],
                "subject": meta["subject"],
                "date": meta["date"],
                "read": meta["read"],
                "body_preview": results["documents"][i][:300] + "..."
            }
    
    emails = list(seen.values())
    emails.sort(key=lambda x: x["date"], reverse=True)
    
    return {
        "count": len(emails),
        "emails": emails
    }

# ── Tool 2: Semantic search ──────────────────────────────────
def search_emails(query, max_results=10, collection=None, client=None):
    """Semantic search over email content using embeddings"""
    
    query_embedding = embed(query, client=client)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max_results * 2,  # over-fetch for deduplication
        include=["documents", "metadatas", "distances"]
    )
    
    # deduplicate by email_id, keep best scoring chunk per email
    seen = {}
    for i, meta in enumerate(results["metadatas"][0]):
        email_id = meta["id"]
        distance = results["distances"][0][i]
        
        if email_id not in seen or distance < seen[email_id]["distance"]:
            seen[email_id] = {
                "email_id": email_id,
                "sender_name": meta["sender_name"],
                "subject": meta["subject"],
                "date": meta["date"],
                "read": meta["read"],
                "relevant_excerpt": results["documents"][0][i],
                "distance": distance
            }
    
    emails = sorted(seen.values(), key=lambda x: x["distance"])[:max_results]

    
    return {
        "count": len(emails),
        "emails": emails
    }

# ── Tool 3: Research company ─────────────────────────────────
def research_company(company_name, client=None):
    """Web search + LLM extraction for company info"""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{
            "role": "user",
            "content": f"""Search for information about {company_name} and extract the following.
            
        Return ONLY a valid JSON object with these exact fields, no other text:
        {{
            "company_name": "{company_name}",
            "funding_stage": "Series A/B/C/etc or unknown",
            "total_funding": "dollar amount or unknown",
            "employee_count": "number or range or unknown",
            "nyc_office": true or false or "unknown",
            "what_they_build": "one sentence description",
            "ai_focus": "specific AI focus area e.g. RAG, agent engineering, LLM infra, etc",
            "verdict": "promising or not_a_fit or unknown"
        }}"""
                }]
            )
    
    # extract text from response — may have tool_use blocks + text
    full_text = " ".join([
        block.text for block in response.content
        if hasattr(block, "text")
    ])
    
    try:
        # find JSON in response
        start = full_text.find("{")
        end = full_text.rfind("}") + 1
        return json.loads(full_text[start:end])
    except:
        return {
            "company_name": company_name,
            "verdict": "unknown",
            "raw": full_text[:500]
        }

# ── Tool 4: Evaluate fit ─────────────────────────────────────
def evaluate_fit(company_name, role, company_research, client=None, email_body=None):
    email_context = ""
    if email_body:
        email_context = f"\nEmail details (use for compensation, specific role info, location):\n{email_body[:500]}"
    
    criteria = """
    GOOD FIT criteria (all should be true):
    - Role type: agent engineering, AI backend, applied AI, RAG/retrieval systems, ML/AI engineering, LLM infrastructure, etc.
    - NOT data engineering, DevOps, frontend-only, or travel-heavy
    - Company stage: Series A, B, or C preferred (D/E ok if other criteria are met)
    - Location: NYC office or fully remote
    - Company is building serious AI products (not just adding AI as a feature)
    
    BAD FIT if any of:
    - Data engineering or analytics role
    - DevOps or infrastructure role
    - Frontend-only role
    - 25%+ travel required
    - Pre-seed or seed stage (too early)
    - No NYC presence and not remote-friendly
    - Below $200K base
    """
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""Evaluate this role against the criteria below. Use the email context to help you evaluate the fit. When email body and web search conflict on funding stage:
- Do NOT reject based on stage discrepancy alone
- Flag it as "stage_unverified" 
- Evaluate on role type, location, and compensation only 

Company: {company_name}
Role: {role}
Research: {json.dumps(company_research)}

Criteria:
{criteria}

Email context: {email_context}

Return ONLY a valid JSON object:
{{
    "fit": true or false,
    "score": 1-10,
    "reason": "one sentence explanation",
    "flags": ["list of specific concerns if any"]
}}"""
        }]
    )
    
    try:
        text = response.content[0].text
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        return {"fit": False, "score": 0, "reason": "Could not evaluate", "flags": []}

# ── Tool 5: Draft response ───────────────────────────────────
def draft_response(to, sender_name, fit, context, tone, client=None):
    """Draft a recruiting email response"""
    
    if tone == "interested":
        instruction = """Write an enthusiastic but professional response. 
        Express genuine interest, mention 1-2 specific things about the role/company that appeal to you.
        Include availability for a call (suggest Tuesday-Thursday afternoons).
        Keep it concise — 3-4 short paragraphs max."""
    else:
        instruction = """Write a polite, warm decline.
        Thank them for reaching out.
        Briefly mention you're focused on a specific type of role (agent engineering at earlier stage companies).
        Leave the door open for future opportunities.
        Keep it to 2-3 sentences."""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Draft a response to this recruiting email. 
To: {sender_name} ({to})

Context about the role/company: {context}

Fit evaluation: {"Good fit" if fit else "Not a fit"}

Instructions: {instruction}

Write the email body only — no subject line, no "Here is a draft:" preamble."""
        }]
    )
    
    return {
        "to": to,
        "sender_name": sender_name,
        "draft": response.content[0].text
    }

# ── Tool dispatcher ──────────────────────────────────────────
def execute_tool(name, inputs):
    print(f"\n🔧 Tool: {name}")
    print(f"   Inputs: {json.dumps(inputs, indent=2)[:200]}")
    
    if name == "get_recruiting_emails":
        result = get_recruiting_emails(**inputs)
    elif name == "search_emails":
        result = search_emails(**inputs)
    elif name == "research_company":
        # no hard enforcement needed here
        result = research_company(**inputs)
    elif name == "evaluate_fit":
        # hard enforcement — company_research must be in inputs
        if "company_research" not in inputs or not inputs["company_research"]:
            result = {"error": "company_research is required. Call research_company first."}
        else:
            result = evaluate_fit(**inputs)
    elif name == "draft_response":
        result = draft_response(**inputs)
    else:
        result = {"error": f"Unknown tool: {name}"}
    
    print(f"   Result preview: {str(result)[:200]}")
    return result