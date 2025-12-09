"""Component implementations for the unified API interface.

All components follow the same pattern:
- Input: ComponentInput (task, input list, previous_outputs)
- Output: ComponentOutput (task, output, component)
"""

import json
import logging
from typing import List

from src.models.models import (
    ComponentInput, 
    ComponentOutput, 
    ComponentOutputData,
    InputItem, 
    PreviousOutput
)
from src.services.llm_client import generate_response, get_llm_client
from src.core.conversation import ConversationContext
from src.services.playbook_service import PlaybookService

logger = logging.getLogger(__name__)

# Initialize playbook service (will be set up when first used)
_playbook_service = None


def get_playbook_service() -> PlaybookService:
    """Get or create playbook service instance."""
    global _playbook_service
    if _playbook_service is None:
        llm_client = get_llm_client()
        _playbook_service = PlaybookService(llm_client)
    return _playbook_service


async def get_context_additions(
    component_input: ComponentInput,
    context: ConversationContext,
    component_name: str
) -> tuple[list, str]:
    """
    Get conversation history and playbook context based on component input settings.
    
    Args:
        component_input: Component input with settings
        context: Conversation context
        component_name: Name of the component (for logging)
        
    Returns:
        Tuple of (conversation_history, playbook_context_string)
    """
    # Get conversation history if enabled
    conversation_history = []
    if component_input.use_conversation_history:
        conversation_history = context.get_recent_messages(count=5)
        logger.info(f"[{component_name}] Using conversation history: {len(conversation_history)} messages")
    else:
        logger.info(f"[{component_name}] Conversation history disabled")
    
    # Get playbook context if enabled
    playbook_context = ""
    if component_input.use_playbook:
        try:
            playbook_service = get_playbook_service()
            playbook_entries = await playbook_service.get_playbook(component_input.cid)
            if playbook_entries:
                playbook_context = "\n\n" + playbook_service.format_playbook_context(playbook_entries)
                logger.info(f"[{component_name}] Using playbook: {len(playbook_entries)} entries")
        except Exception as e:
            logger.warning(f"[{component_name}] Failed to load playbook: {e}")
            playbook_context = ""  # Ensure empty string on failure
    else:
        logger.info(f"[{component_name}] Playbook disabled")
    
    return conversation_history, playbook_context





async def component_complete(
    component_input: ComponentInput,
    context: ConversationContext
) -> ComponentOutput:
    """
    Complete component: Process tasks with optional conversation history and playbook.
    
    Args:
        component_input: Unified component input
        context: Conversation context with history
        
    Returns:
        ComponentOutput with the completed task
    """
    logger.info(f"[complete] Processing task: {component_input.task}")
    
    # Build input text from all input items
    input_text_parts = []
    for idx, item in enumerate(component_input.input, 1):
        input_text_parts.append(f"Query {idx}: {item.user_query}")
    
    input_text = "\n\n".join(input_text_parts)
    
    # Build previous outputs context - LLM will read everything and decide intelligently
    previous_context = ""
    if component_input.previous_outputs:
        previous_context = "\n\nPrevious component outputs:\n"
        for prev in component_input.previous_outputs:
            # Show the complete output with immediate_response and notebook
            previous_context += f"\n[{prev.component}] {prev.task}:\n"
            previous_context += f"  Response: {prev.output.immediate_response}\n"
            if prev.output.notebook and prev.output.notebook != "no update":
                previous_context += f"  Notebook: {prev.output.notebook}\n"
    
    # Get conversation history and playbook context
    conversation_history, playbook_context = await get_context_additions(
        component_input, context, "complete"
    )
    
    # Build system prompt optimized for accuracy and all evaluation criteria
    system_prompt = """You are a highly precise and accurate AI assistant. Your primary goal is to provide CORRECT, ACCURATE answers above all else.

CRITICAL QUALITY STANDARDS (in order of importance):

1. ACCURACY (MOST IMPORTANT - 70% of evaluation):
   - Verify all facts, calculations, and logic before responding
   - For mathematical problems: Show step-by-step work, double-check calculations, verify the answer makes sense
   - For factual questions: Ensure information is correct and up-to-date
   - For code/logic problems: Test your reasoning and verify correctness
   - If uncertain, clearly state limitations but still provide your best answer
   - NEVER guess - use logical deduction and verification

2. RELEVANCE (7.5% of evaluation):
   - Directly address what is being asked
   - Stay focused on the specific question or task
   - Don't provide tangential information unless it's directly relevant

3. COMPLETENESS (7.5% of evaluation):
   - Answer ALL parts of the question
   - Provide sufficient detail to fully address the request
   - If multiple questions are asked, answer each one thoroughly

4. CLARITY (5% of evaluation):
   - Write in clear, understandable language
   - Use proper grammar and sentence structure
   - Organize your response logically
   - Explain complex concepts in accessible terms

5. FOLLOWING INSTRUCTIONS (5% of evaluation):
   - Read the task description carefully
   - Follow all specified format requirements
   - Adhere to any constraints or guidelines provided
   - Match the requested output style

6. FORMAT/STRUCTURE (2.5% of evaluation):
   - Structure your response well
   - Use appropriate formatting (lists, paragraphs, headers as needed)
   - Make the response easy to scan and understand

7. SAFETY (2.5% of evaluation):
   - Provide appropriate, ethical responses
   - Avoid harmful, dangerous, or illegal content
   - Be respectful and professional

RESPONSE FORMAT REQUIREMENTS:

You MUST respond in valid JSON format with exactly two fields:
{
  "immediate_response": "Your natural language explanation showing work, reasoning, and the final answer",
  "notebook": "Updated notebook content OR 'no update'"
}

Guidelines for notebook field:
- If task is conversational only (no code/document editing): Return "no update"
- If there's ONE notebook and no changes needed: Return "no update"
- If there's ONE notebook and changes needed: Return the updated version
- If there are MULTIPLE notebooks: You MUST create new content (combine/choose/merge) - NEVER "no update"
- If creating new notebook: Return the full content
- Always provide valid, properly formatted JSON

PROCESSING APPROACH:
1. Read the task and input carefully
2. Identify what is being asked (especially for math/logic problems)
3. Show your reasoning process (especially for calculations)
4. Verify your answer before responding
5. Format as valid JSON

Remember: ACCURACY IS PARAMOUNT. A correct, well-reasoned answer is far more valuable than a fast but incorrect one."""
    
    if playbook_context:
        system_prompt += f"\n\nUser preferences and context:\n{playbook_context}"
    
    # Build task prompt with enhanced instructions
    task_prompt = f"""Task: {component_input.task}

Input:
{input_text}
{previous_context}

Instructions:
- Read the task and input carefully
- Show your reasoning process, especially for mathematical or logical problems
- Verify your answer before responding
- Ensure your answer directly addresses what is being asked
- Respond in valid JSON format with "immediate_response" and "notebook" fields"""
    
    # Generate response with optional conversation history (lower temperature for better accuracy)
    response = await generate_response(
        prompt=task_prompt,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        temperature=0.2  # Lower temperature for more deterministic, accurate answers
    )
    
    # Parse JSON response
    try:
        # Try to extract JSON from response (handle markdown code blocks)
        response_text = response.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        immediate_response = result.get("immediate_response", response)
        notebook_output = result.get("notebook", "no update")
        
        # Ensure notebook is a string (convert dict/object to JSON string if needed)
        if isinstance(notebook_output, dict):
            logger.warning(f"[complete] Notebook returned as dict, converting to JSON string")
            notebook_output = json.dumps(notebook_output, indent=2)
        elif not isinstance(notebook_output, str):
            logger.warning(f"[complete] Notebook is not a string (type: {type(notebook_output)}), converting")
            notebook_output = str(notebook_output)
            
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning(f"[complete] Failed to parse JSON response: {e}. Using raw response.")
        immediate_response = response
        notebook_output = "no update"
    
    # Resolve "no update" for notebook - return previous notebook if exists
    if notebook_output == "no update" and component_input.previous_outputs:
        resolved = False
        for prev in component_input.previous_outputs:
            if prev.output.notebook and prev.output.notebook != "no update":
                notebook_output = prev.output.notebook
                logger.info(f"[complete] Resolved 'no update' to previous notebook from [{prev.component}]")
                resolved = True
                break
        
        if not resolved:
            logger.info(f"[complete] No previous notebook found to resolve - keeping 'no update'")
    
    # Store in conversation history
    context.add_user_message(f"Task: {component_input.task}\n{input_text}")
    context.add_assistant_message(immediate_response)
    
    return ComponentOutput(
        cid=component_input.cid,
        task=component_input.task,
        input=component_input.input,
        output=ComponentOutputData(
            immediate_response=immediate_response,
            notebook=notebook_output  # Resolved: new content, previous notebook, or "no update"
        ),
        component="complete"
    )


async def component_refine(
    component_input: ComponentInput,
    context: ConversationContext
) -> ComponentOutput:
    """
    Refine component: Improve outputs with optional conversation history and playbook.
    
    Args:
        component_input: Unified component input with previous outputs to refine
        context: Conversation context
        
    Returns:
        ComponentOutput with refined output
    """
    logger.info(f"[refine] Processing task: {component_input.task}")
    
    # Build input text
    input_text_parts = []
    for idx, item in enumerate(component_input.input, 1):
        input_text_parts.append(f"Query {idx}: {item.user_query}")
    
    input_text = "\n\n".join(input_text_parts)
    
    # Build previous outputs context - LLM will read everything and decide intelligently
    previous_outputs_text = ""
    if component_input.previous_outputs:
        previous_outputs_text = "\n\nPrevious outputs to refine:\n"
        for prev in component_input.previous_outputs:
            previous_outputs_text += f"\n[{prev.component}] {prev.task}:\n"
            previous_outputs_text += f"  Response: {prev.output.immediate_response}\n"
            if prev.output.notebook and prev.output.notebook != "no update":
                previous_outputs_text += f"  Notebook: {prev.output.notebook}\n"
    
    # Get conversation history and playbook context
    conversation_history, playbook_context = await get_context_additions(
        component_input, context, "refine"
    )
    
    # Build system prompt optimized for accuracy and quality
    system_prompt = """You are an AI assistant that refines and improves outputs with precision and accuracy.

CRITICAL QUALITY STANDARDS:

1. ACCURACY (MOST IMPORTANT):
   - Verify all improvements are correct
   - Double-check any corrections to calculations, facts, or logic
   - Ensure refined content is accurate and valid

2. RELEVANCE:
   - Focus improvements on what was actually requested
   - Don't change aspects that are already correct unless asked

3. COMPLETENESS:
   - Address all aspects that need refinement
   - Ensure the refined output is complete and thorough

4. CLARITY:
   - Improve clarity and readability
   - Make explanations clearer and more understandable

5. FOLLOWING INSTRUCTIONS:
   - Follow the refinement task exactly
   - Adhere to format requirements

6. FORMAT:
   - Maintain or improve structure and organization

7. SAFETY:
   - Ensure refined content is appropriate and ethical

RESPONSE FORMAT REQUIREMENTS:

You MUST respond in valid JSON format:
{
  "immediate_response": "Clear explanation of what you refined, why, and how the improvements enhance accuracy and quality",
  "notebook": "The refined/improved content OR 'no update'"
}

Guidelines for notebook field:
- If providing feedback only: Set notebook to "no update"
- If there's ONE notebook and no improvements needed: Set to "no update"
- If there's ONE notebook and improvements needed: Write the improved, more accurate version
- If there are MULTIPLE notebooks: You MUST create new content (refine one, combine, or merge) - NEVER "no update"
- Always provide valid JSON"""
    
    if playbook_context:
        system_prompt += f"\n\nUser preferences:\n{playbook_context}"
    
    # Build refine prompt with enhanced instructions
    refine_prompt = f"""Task: {component_input.task}

Original Input:
{input_text}
{previous_outputs_text}

Instructions:
- Carefully analyze the outputs to identify areas for improvement
- Focus especially on improving accuracy, correctness, and clarity
- Verify that all improvements are correct
- Respond in valid JSON format with "immediate_response" and "notebook" fields"""
    
    # Generate response (lower temperature for more precise refinements)
    response = await generate_response(
        prompt=refine_prompt,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        temperature=0.3  # Lower temperature for more accurate refinements
    )
    
    # Parse JSON response
    try:
        response_text = response.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        immediate_response = result.get("immediate_response", response)
        notebook_output = result.get("notebook", "no update")
        
        # Ensure notebook is a string (convert dict/object to JSON string if needed)
        if isinstance(notebook_output, dict):
            logger.warning(f"[refine] Notebook returned as dict, converting to JSON string")
            notebook_output = json.dumps(notebook_output, indent=2)
        elif not isinstance(notebook_output, str):
            logger.warning(f"[refine] Notebook is not a string (type: {type(notebook_output)}), converting")
            notebook_output = str(notebook_output)
            
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning(f"[refine] Failed to parse JSON response: {e}. Using raw response.")
        immediate_response = response
        notebook_output = "no update"
    
    # Resolve "no update" for notebook - return previous notebook if exists
    if notebook_output == "no update" and component_input.previous_outputs:
        resolved = False
        for prev in component_input.previous_outputs:
            if prev.output.notebook and prev.output.notebook != "no update":
                notebook_output = prev.output.notebook
                logger.info(f"[refine] Resolved 'no update' to previous notebook from [{prev.component}]")
                resolved = True
                break
        
        if not resolved:
            logger.info(f"[refine] No previous notebook found to resolve - keeping 'no update'")
    
    # Store in conversation history
    context.add_user_message(f"Refine task: {component_input.task}")
    context.add_assistant_message(immediate_response)
    
    return ComponentOutput(
        cid=component_input.cid,
        task=component_input.task,
        input=component_input.input,
        output=ComponentOutputData(
            immediate_response=immediate_response,
            notebook=notebook_output  # Resolved: refined content, previous notebook, or "no update"
        ),
        component="refine"
    )


async def component_feedback(
    component_input: ComponentInput,
    context: ConversationContext
) -> ComponentOutput:
    """
    Feedback component: Analyze outputs and provide structured feedback.
    
    Args:
        component_input: Unified component input with outputs to analyze
        context: Conversation context
        
    Returns:
        ComponentOutput with structured feedback
    """
    logger.info(f"[feedback] Processing task: {component_input.task}")
    
    # Build previous outputs to analyze
    outputs_to_analyze = ""
    if component_input.previous_outputs:
        outputs_to_analyze = "\n\nOutputs to analyze:\n"
        for prev in component_input.previous_outputs:
            # Access Pydantic object attributes
            outputs_to_analyze += f"\n[{prev.component}] {prev.task}:\n"
            outputs_to_analyze += f"  Response: {prev.output.immediate_response}\n"
            if prev.output.notebook and prev.output.notebook != "no update":
                outputs_to_analyze += f"  Notebook: {prev.output.notebook}\n"
    
    # Get conversation history and playbook context
    conversation_history, playbook_context = await get_context_additions(
        component_input, context, "feedback"
    )
    
    # Build system prompt optimized for quality feedback
    system_prompt = """You are an AI assistant that provides accurate, constructive, and insightful feedback.

CRITICAL QUALITY STANDARDS:

1. ACCURACY (MOST IMPORTANT):
   - Provide accurate, factual feedback
   - Correctly identify strengths and weaknesses
   - Base suggestions on accurate analysis

2. RELEVANCE:
   - Focus feedback on aspects that directly relate to the task
   - Address the most important improvement areas

3. COMPLETENESS:
   - Cover all major aspects: strengths, weaknesses, and suggestions
   - Provide comprehensive analysis

4. CLARITY:
   - Write clear, understandable feedback
   - Structure feedback logically with clear sections
   - Use specific examples where helpful

5. FOLLOWING INSTRUCTIONS:
   - Follow the feedback task requirements
   - Address all requested aspects

6. FORMAT:
   - Organize feedback with clear sections
   - Use appropriate formatting for readability

7. SAFETY:
   - Provide respectful, constructive feedback
   - Maintain appropriate tone and content"""
    
    if playbook_context:
        system_prompt += f"\n\nUser preferences:\n{playbook_context}"
    
    # Build feedback prompt with enhanced instructions
    feedback_prompt = f"""Task: {component_input.task}
{outputs_to_analyze}

Instructions:
- Analyze the outputs accurately and provide structured feedback
- For each output, identify:
  1. Strengths (what works well and is correct)
  2. Weaknesses (what could be improved, including accuracy issues)
  3. Specific, actionable suggestions for improvement (focus on improving accuracy and quality)
- Ensure your feedback is accurate, relevant, complete, and clearly formatted
- Format your feedback with clear sections for easy reading"""
    
    # Generate feedback (moderate temperature for balanced feedback)
    response = await generate_response(
        prompt=feedback_prompt,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        temperature=0.4  # Lower temperature for more accurate, focused feedback
    )
    
    # Store in conversation history
    context.add_user_message(f"Feedback request: {component_input.task}")
    context.add_assistant_message(response)
    
    # Feedback is conversational - no notebook editing
    return ComponentOutput(
        cid=component_input.cid,
        task=component_input.task,
        input=component_input.input,
        output=ComponentOutputData(
            immediate_response=response,
            notebook="no update"
        ),
        component="feedback"
    )


async def component_human_feedback(
    component_input: ComponentInput,
    context: ConversationContext
) -> ComponentOutput:
    """
    Human feedback component: Process and extract structured insights to playbook.
    
    Uses LLM to extract actionable insights from human feedback and stores them
    in a structured playbook (knowledge base) with operations (insert/update/delete).
    
    Inspired by: https://github.com/kayba-ai/agentic-context-engine
    
    Args:
        component_input: Unified component input with human feedback
        context: Conversation context
        
    Returns:
        ComponentOutput with summary of extracted insights
    """
    logger.info(f"[human_feedback] Processing task: {component_input.task}")
    
    # Extract human feedback from input
    feedback_text_parts = []
    for item in component_input.input:
        if item.user_query:
            feedback_text_parts.append(item.user_query)
    
    feedback_text = "\n".join(feedback_text_parts)
    

    
    if not feedback_text.strip():
        return ComponentOutput(
            cid=component_input.cid,
            task=component_input.task,
            input=component_input.input,
            output=ComponentOutputData(
                immediate_response="No feedback text provided.",
                notebook="no update"
            ),
            component="human_feedback"
        )
    
    logger.info(f"[human_feedback] Received feedback: {feedback_text[:100]}...")
    
    try:
        # Get playbook service
        playbook_service = get_playbook_service()
        
        # Get conversation context for better extraction
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content'][:100]}..."
            for msg in context.get_messages()[-5:]  # Last 5 messages
        ])
        
        # Extract insights using LLM
        insights = await playbook_service.extract_insights(
            feedback=feedback_text,
            cid=component_input.cid,
            context=conversation_context
        )
        
        # Apply operations to playbook
        entries = await playbook_service.apply_operations(
            insights=insights,
            cid=component_input.cid,
            source_feedback=feedback_text
        )
        
        # Format response
        if insights:
            response_parts = [
                "✅ Thank you for your feedback! I've analyzed it and extracted the following insights:\n"
            ]
            
            for idx, insight in enumerate(insights, 1):
                operation_emoji = {
                    "insert": "➕",
                    "update": "🔄",
                    "delete": "❌"
                }.get(insight["operation"], "•")
                
                response_parts.append(
                    f"{operation_emoji} **{insight['insight_type'].title()}** ({insight['operation']})\n"
                    f"   Key: `{insight['key']}`\n"
                    f"   Value: {insight['value']}\n"
                    f"   Confidence: {insight.get('confidence_score', 0.8):.0%}"
                )
                if insight.get('tags'):
                    response_parts.append(f"   Tags: {', '.join(insight['tags'])}")
                response_parts.append("")
            
            response_parts.append(
                f"\n📚 Your playbook now has {len(entries)} active entries. "
                "I'll use this knowledge in our future conversations!"
            )
            
            message = "\n".join(response_parts)
        else:
            message = (
                "Thank you for your feedback. However, I couldn't extract any "
                "actionable insights to add to your playbook. Your feedback has "
                "been stored in the conversation history for context."
            )
        
        logger.info(f"[human_feedback] Extracted {len(insights)} insights, created/updated {len(entries)} entries")
        
        # Store in conversation history
        context.add_user_message(f"User feedback: {feedback_text}")
        context.add_assistant_message(message)
        
        # Create JSON summary of insights for notebook
        notebook_data = {
            "feedback": feedback_text,
            "insights_extracted": len(insights),
            "entries_modified": len(entries),
            "insights": insights
        }
        
        notebook_json = json.dumps(notebook_data, indent=2)
        
        return ComponentOutput(
            cid=component_input.cid,
            task=component_input.task,
            input=component_input.input,
            output=ComponentOutputData(
                immediate_response=message,
                notebook=notebook_json  # Structured insights data
            ),
            component="human_feedback"
        )
        
    except Exception as e:
        logger.error(f"[human_feedback] Error processing feedback: {e}", exc_info=True)
        
        # Fallback to simple storage
        message = (
            f"Thank you for your feedback. I've noted it for future reference:\n\n"
            f"{feedback_text}\n\n"
            f"(Note: Advanced insight extraction encountered an error, but your "
            f"feedback is stored in conversation history)"
        )
        
        context.add_user_message(f"User feedback: {feedback_text}")
        context.add_assistant_message(message)
        
        return ComponentOutput(
            cid=component_input.cid,
            task=component_input.task,
            input=component_input.input,
            output=ComponentOutputData(
                immediate_response=message,
                notebook="no update"  # Error case
            ),
            component="human_feedback"
        )


async def component_internet_search(
    component_input: ComponentInput,
    context: ConversationContext
) -> ComponentOutput:
    """
    Internet search component: Search the internet for information.
    
    NOTE: This is a template implementation. Miners should implement actual
    internet search functionality using services like:
    - Google Custom Search API
    - Bing Search API
    - DuckDuckGo API
    - SerpAPI
    - Or any other search service
    
    Args:
        component_input: Unified component input with search queries
        context: Conversation context
        
    Returns:
        ComponentOutput with search results (currently returns "unavailable service")
    """
    logger.info(f"[internet_search] Processing task: {component_input.task}")
    
    # Extract search queries
    search_queries = []
    for item in component_input.input:
        search_queries.append(item.user_query)
    
    # Template response - miners should replace this with actual implementation
    response = f"""Internet Search Service: UNAVAILABLE (Template)

This is a template implementation. Miners should implement actual internet search.

Queries received:
{chr(10).join(f"- {q}" for q in search_queries)}

IMPLEMENTATION NOTES FOR MINERS:
==================================
To implement internet search, you can use:

1. Google Custom Search API:
   - Create API key at: https://console.cloud.google.com/
   - Use googleapis Python library
   - Example: google.search(query, num_results=10)

2. Bing Search API:
   - Get API key from Azure Cognitive Services
   - Use requests library to call Bing API

3. DuckDuckGo API:
   - Free and no API key required
   - Use duckduckgo-search Python library
   - Example: from duckduckgo_search import DDGS

4. SerpAPI:
   - Multi-engine search API
   - Supports Google, Bing, Yahoo, etc.
   - Example: serpapi.search(query)

Implementation should:
- Parse search queries from component_input.input
- Execute searches using your chosen service
- Format results as structured text
- Return ComponentOutput with results
- Handle rate limiting and errors gracefully

Replace this function body with your actual search implementation."""
    
    # Store in conversation history
    context.add_user_message(f"Search: {', '.join(search_queries)}")
    context.add_assistant_message(response)
    

    
    # Internet search is conversational - no notebook editing
    return ComponentOutput(
        cid=component_input.cid,
        task=component_input.task,
        input=component_input.input,
        output=ComponentOutputData(
            immediate_response=response,
            notebook="no update"
        ),
        component="internet_search"
    )


async def component_summary(
    component_input: ComponentInput,
    context: ConversationContext
) -> ComponentOutput:
    """
    Summary component: Use LLM to summarize previous outputs.
    
    Args:
        component_input: Unified component input with outputs to summarize
        context: Conversation context
        
    Returns:
        ComponentOutput with summarized content
    """
    logger.info(f"[summary] Processing task: {component_input.task}")
    
    # Build content to summarize from previous outputs
    content_to_summarize = []
    if component_input.previous_outputs:
        for prev in component_input.previous_outputs:
            # Access Pydantic object attributes
            output_text = f"[{prev.component}] {prev.task}:\n"
            output_text += f"Response: {prev.output.immediate_response}\n"
            if prev.output.notebook and prev.output.notebook != "no update":
                output_text += f"Notebook: {prev.output.notebook}\n"
            content_to_summarize.append(output_text)
    

    
    if not content_to_summarize:
        return ComponentOutput(
            cid=component_input.cid,
            task=component_input.task,
            input=component_input.input,
            output=ComponentOutputData(
                immediate_response="No previous outputs to summarize.",
                notebook="no update"
            ),
            component="summary"
        )
    
    combined_content = "\n\n---\n\n".join(content_to_summarize)
    
    # Get conversation history and playbook context
    conversation_history, playbook_context = await get_context_additions(
        component_input, context, "summary"
    )
    
    # Build system prompt optimized for accuracy and completeness
    system_prompt = """You are an AI assistant that creates accurate, comprehensive summaries.

CRITICAL QUALITY STANDARDS:

1. ACCURACY (MOST IMPORTANT):
   - Ensure all facts, numbers, and key information in the summary are correct
   - Preserve the meaning and intent of the original content accurately
   - Don't introduce errors or misinterpretations

2. RELEVANCE:
   - Include only relevant and important information
   - Focus on the main points and key insights

3. COMPLETENESS:
   - Capture all main points and key insights
   - Don't omit critical information
   - Ensure the summary is comprehensive yet concise

4. CLARITY:
   - Write clearly and understandably
   - Organize information logically
   - Use proper structure and formatting

5. FOLLOWING INSTRUCTIONS:
   - Follow the summary task requirements
   - Adhere to format guidelines

6. FORMAT:
   - Structure the summary well
   - Use appropriate organization

7. SAFETY:
   - Maintain appropriate content standards

RESPONSE FORMAT REQUIREMENTS:

You MUST respond in valid JSON format with two fields:
{
  "immediate_response": "Your summary explanation that accurately captures key points",
  "notebook": "Summarized notebook content OR 'no update'"
}

Guidelines for notebook field:
- If there's NO notebook content in inputs: Return "no update"
- If there's ONE notebook to summarize: Return the accurately summarized version
- If there are MULTIPLE notebooks: Create a combined summary that accurately represents all content
- Always provide valid JSON"""
    
    if playbook_context:
        system_prompt += f"\n\nUser preferences:\n{playbook_context}"
    
    # Build summary prompt with enhanced instructions
    summary_prompt = f"""Task: {component_input.task}

Content to summarize:
{combined_content}

Instructions:
- Create a comprehensive, accurate summary that:
  1. Captures ALL main points and key insights correctly
  2. Maintains accuracy of all facts and numbers
  3. Removes redundancy while preserving important details
  4. Organizes information logically and clearly
- Verify that your summary accurately represents the source content
- Respond in valid JSON format with "immediate_response" and "notebook" fields"""
    
    # Generate summary (moderate temperature for balanced summaries)
    response = await generate_response(
        prompt=summary_prompt,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        temperature=0.4  # Slightly lower for more accurate summaries
    )
    
    # Parse JSON response
    try:
        response_text = response.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        immediate_response = result.get("immediate_response", response)
        notebook_output = result.get("notebook", "no update")
        
        # Ensure notebook is a string
        if isinstance(notebook_output, dict):
            logger.warning(f"[summary] Notebook returned as dict, converting to JSON string")
            notebook_output = json.dumps(notebook_output, indent=2)
        elif not isinstance(notebook_output, str):
            logger.warning(f"[summary] Notebook is not a string (type: {type(notebook_output)}), converting")
            notebook_output = str(notebook_output)
            
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning(f"[summary] Failed to parse JSON response: {e}. Using raw response.")
        immediate_response = response
        notebook_output = "no update"
    
    # Resolve "no update" for notebook - return previous notebook if exists
    if notebook_output == "no update" and component_input.previous_outputs:
        resolved = False
        for prev in component_input.previous_outputs:
            if prev.output.notebook and prev.output.notebook != "no update":
                notebook_output = prev.output.notebook
                logger.info(f"[summary] Resolved 'no update' to previous notebook from [{prev.component}]")
                resolved = True
                break
        
        if not resolved:
            logger.info(f"[summary] No previous notebook found to resolve - keeping 'no update'")
    
    # Store in conversation history
    context.add_user_message(f"Summarize: {component_input.task}")
    context.add_assistant_message(immediate_response)
    
    return ComponentOutput(
        cid=component_input.cid,
        task=component_input.task,
        input=component_input.input,
        output=ComponentOutputData(
            immediate_response=immediate_response,
            notebook=notebook_output  # Resolved: summarized content, previous notebook, or "no update"
        ),
        component="summary"
    )


async def component_aggregate(
    component_input: ComponentInput,
    context: ConversationContext
) -> ComponentOutput:
    """
    Aggregate component: Perform majority voting on previous outputs.
    
    This component analyzes multiple previous outputs and identifies the most
    common or agreed-upon answer through majority voting logic.
    
    Args:
        component_input: Unified component input with outputs to aggregate
        context: Conversation context
        
    Returns:
        ComponentOutput with aggregated result
    """
    logger.info(f"[aggregate] Processing task: {component_input.task}")
    

    
    if not component_input.previous_outputs:
        return ComponentOutput(
            cid=component_input.cid,
            task=component_input.task,
            input=component_input.input,
            output=ComponentOutputData(
                immediate_response="No previous outputs to aggregate.",
                notebook="no update"
            ),
            component="aggregate"
        )
    
    # Build outputs for analysis
    outputs_text = []
    for idx, prev in enumerate(component_input.previous_outputs, 1):
        # Access Pydantic object attributes
        output_text = f"Output {idx} [{prev.component}]:\n"
        output_text += f"Response: {prev.output.immediate_response}\n"
        if prev.output.notebook and prev.output.notebook != "no update":
            output_text += f"Notebook: {prev.output.notebook}\n"
        outputs_text.append(output_text)
    
    combined_outputs = "\n\n---\n\n".join(outputs_text)
    
    # Get conversation history and playbook context
    conversation_history, playbook_context = await get_context_additions(
        component_input, context, "aggregate"
    )
    
    # Build system prompt optimized for accurate aggregation
    system_prompt = """You are an AI assistant that aggregates multiple outputs using accurate majority voting and consensus analysis.

CRITICAL QUALITY STANDARDS:

1. ACCURACY (MOST IMPORTANT):
   - Verify which answer is most commonly correct across outputs
   - Use logical analysis to determine the best consensus answer
   - Prefer the most accurate answer, not just the most common one
   - If one answer is clearly correct, choose it even if less common

2. RELEVANCE:
   - Focus on answers that directly address the question
   - Filter out irrelevant or off-topic responses

3. COMPLETENESS:
   - Consider all aspects of the outputs
   - Ensure the consensus answer is complete

4. CLARITY:
   - Clearly explain the consensus and voting logic
   - Show which outputs agree and which differ

5. FOLLOWING INSTRUCTIONS:
   - Follow majority voting logic accurately
   - Adhere to format requirements

6. FORMAT:
   - Structure the aggregation clearly
   - Organize consensus explanation well

7. SAFETY:
   - Ensure aggregated content is appropriate

RESPONSE FORMAT REQUIREMENTS:

You MUST respond in valid JSON format with two fields:
{
  "immediate_response": "Clear explanation of the consensus, voting results, and how you determined the most accurate answer",
  "notebook": "The aggregated/consensus notebook content OR 'no update'"
}

Guidelines for notebook field:
- If there's NO notebook content in inputs: Return "no update"
- If there's ONE notebook: Return it as-is (or "no update" if no changes)
- If there are MULTIPLE notebooks: Create aggregated version using majority voting, choosing the most accurate consensus
- Use majority voting: Choose the most common AND most accurate content, or merge agreements intelligently
- Always provide valid JSON"""
    
    if playbook_context:
        system_prompt += f"\n\nUser preferences:\n{playbook_context}"
    
    # Build aggregate prompt with enhanced instructions
    aggregate_prompt = f"""Task: {component_input.task}

Multiple outputs to aggregate:
{combined_outputs}

Instructions:
- Carefully analyze these outputs and determine the consensus answer by:
  1. Identifying common themes and agreements (especially correct answers)
  2. Noting where outputs differ and why
  3. Using majority voting logic: Choose the most commonly correct answer
  4. Prioritizing accuracy: If one answer is clearly more accurate, prefer it
  5. Highlighting any important minority opinions that might be valid
- Verify the consensus answer is correct and well-reasoned
- Respond in valid JSON format with "immediate_response" and "notebook" fields"""
    
    # Generate aggregate result (low temperature for deterministic consensus)
    response = await generate_response(
        prompt=aggregate_prompt,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        temperature=0.2  # Very low temperature for accurate, deterministic consensus
    )
    
    # Parse JSON response
    try:
        response_text = response.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        immediate_response = result.get("immediate_response", response)
        notebook_output = result.get("notebook", "no update")
        
        # Ensure notebook is a string
        if isinstance(notebook_output, dict):
            logger.warning(f"[aggregate] Notebook returned as dict, converting to JSON string")
            notebook_output = json.dumps(notebook_output, indent=2)
        elif not isinstance(notebook_output, str):
            logger.warning(f"[aggregate] Notebook is not a string (type: {type(notebook_output)}), converting")
            notebook_output = str(notebook_output)
            
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning(f"[aggregate] Failed to parse JSON response: {e}. Using raw response.")
        immediate_response = response
        notebook_output = "no update"
    
    # Resolve "no update" for notebook - return previous notebook if exists
    if notebook_output == "no update" and component_input.previous_outputs:
        resolved = False
        for prev in component_input.previous_outputs:
            if prev.output.notebook and prev.output.notebook != "no update":
                notebook_output = prev.output.notebook
                logger.info(f"[aggregate] Resolved 'no update' to previous notebook from [{prev.component}]")
                resolved = True
                break
        
        if not resolved:
            logger.info(f"[aggregate] No previous notebook found to resolve - keeping 'no update'")
    
    # Store in conversation history
    context.add_user_message(f"Aggregate: {component_input.task}")
    context.add_assistant_message(immediate_response)
    
    return ComponentOutput(
        cid=component_input.cid,
        task=component_input.task,
        input=component_input.input,
        output=ComponentOutputData(
            immediate_response=immediate_response,
            notebook=notebook_output  # Resolved: aggregated content, previous notebook, or "no update"
        ),
        component="aggregate"
    )
