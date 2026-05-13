import re

from anthropic import Anthropic


def extract_company(subject, body=None, client=None):
    # common patterns in recruiting subjects
    patterns = [
        r' at ([A-Z][a-zA-Z\s\.]+?)(?:\s*[-\(]|$)',  # "Engineer at Hebbia"
        r'- ([A-Z][a-zA-Z\s\.]+?)(?:\s*[-\(]|$)',     # "Engineer - Hebbia"
        r'\| ([A-Z][a-zA-Z\s\.]+?)(?:\s*[-\(]|$)',    # "Engineer | Hebbia"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, subject)
        if match:
            return match.group(1).strip()
    
    # fall back to body if subject doesn't have company name
    if body:
        return extract_company_from_body(body, client=client)

    return "unknown"


def extract_company_from_body(body, client=None):
    """Uses Anthropic when regex fails; pass ``client`` when calling from thread pools."""
    c = client if client is not None else Anthropic()
    # look in first 2 sentences — company name almost always mentioned early
    first_sentences = ". ".join(body.split(".")[:2])
    
    response = c.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": f"""Extract the company name being recruited for from this text.
If it's a 3rd party recruiter, extract the company they're recruiting for, not the recruiting firm.
Return only the company name. If unclear, return "unknown".

Text: {first_sentences}"""
        }]
    )
    return response.content[0].text.strip()