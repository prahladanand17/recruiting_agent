tool_schema = [
    {
    "name": "get_recruiting_emails",
    "description": "Fetch recruiting emails by metadata filters. Use for queries about specific senders, date ranges, or read status. All parameters are optional.",
    "input_schema": {
        "type": "object",
        "properties": {
            "days": {"type": "integer", "description": "Number of days to look back. Omit for all emails."},
            "sender": {"type": "string", "description": "Filter by sender email or company name"},
            "read": {"type": "boolean", "description": "Filter by read/unread status. Omit for both."},
            "max_results": {"type": "integer", "description": "Max emails to return. Default 50."}
        }
    }
    },
    {
        "name": "search_emails",
        "description": "Search indexed emails by text content. Use for keyword queries or full-text searches.",
        "input_schema": {
            "type": "object",
            "query": {"type": "string", "description": "Text query to search within email bodies"}
        }
    },
    {
        "name": "research_company",
        "description": "Research a company given its name. Returns funding stage, size, what they build, NYC presence, and market insight for the company.",
        "input_schema": {
            "type": "object",
            "company_name": {"type": "string"}
        }
    },
    {
    "name": "evaluate_fit",
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Name of the company being evaluated"},
                "role": {"type": "string", "description": "Role being evaluated"},
                "company_research": {"type": "object", "description": "Research about the company"},
                "email_body": {"type": "string", "description": "Full body of the recruiting email"}
            },
            "required": ["company_name", "role", "company_research", "email_body"]
        }
    },
    {
        "name": "draft_response",
        "description": "Draft a response to a recruiting email. Either a polite decline or an enthusiastic reply with availability. Use the sender_name (e.g. 'Dylan Baker'), sender email from the email returned by get_recruiting_emails or search_emails.",
        "input_schema": {
            "type": "object",
            "to": {"type": "string"},
            "sender_name": {"type": "string"},
            "fit": {"type": "boolean"},
            "context": {"type": "string"},
            "tone": {"type": "string", "enum": ["interested", "decline"]}
        }
    }
]