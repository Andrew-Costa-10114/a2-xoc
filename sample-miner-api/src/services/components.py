"""Component implementations for the unified API interface.

All components follow the same pattern:
- Input: ComponentInput (task, input list, previous_outputs)
- Output: ComponentOutput (task, output, component)
"""

import json
import logging
import re
from typing import List, Tuple

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

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


def parse_json_response(response: str, component_name: str = "component", fallback_response: str = None) -> Tuple[str, str]:
    """
    Robustly parse JSON response from LLM output.
    
    Handles various formats:
    - Direct JSON objects
    - JSON wrapped in markdown code blocks (```json or ```)
    - Malformed JSON with common issues
    - Missing or incorrect field types
    
    Args:
        response: Raw response string from LLM
        component_name: Name of component (for logging)
        fallback_response: Fallback text to use if parsing fails completely
        
    Returns:
        Tuple of (immediate_response, notebook_output)
    """
    if not response or not response.strip():
        logger.warning(f"[{component_name}] Empty response received")
        return (fallback_response or ""), "no update"
    
    response_text = response.strip()
    original_response = response_text
    
    # Method 1: Try extracting from markdown code blocks
    json_candidates = []
    
    # Extract from ```json blocks
    if "```json" in response_text:
        try:
            extracted = response_text.split("```json")[1].split("```")[0].strip()
            if extracted:
                json_candidates.append(extracted)
        except (IndexError, AttributeError):
            pass
    
    # Extract from generic ``` blocks (if no json block found)
    if not json_candidates and "```" in response_text:
        try:
            # Try to find code blocks and extract JSON-like content
            code_blocks = re.findall(r'```[a-z]*\s*\n(.*?)\n```', response_text, re.DOTALL)
            for block in code_blocks:
                stripped = block.strip()
                if stripped.startswith('{') and stripped.endswith('}'):
                    json_candidates.append(stripped)
        except (AttributeError, IndexError):
            pass
    
    # Method 2: Try the whole response if it looks like JSON
    if response_text.startswith('{') and response_text.endswith('}'):
        json_candidates.append(response_text)
    
    # Method 3: Try to find JSON object in the response (for cases where there's extra text)
    if not json_candidates:
        # Find the first { ... } pair that looks like JSON
        brace_start = response_text.find('{')
        if brace_start >= 0:
            brace_count = 0
            for i in range(brace_start, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = response_text[brace_start:i+1]
                        if ':' in candidate and ('"immediate_response' in candidate or '"notebook' in candidate):
                            json_candidates.append(candidate.strip())
                        break
    
    # Try parsing each candidate
    for candidate in json_candidates:
        try:
            result = json.loads(candidate)
            
            # Extract fields with proper defaults
            immediate_response = result.get("immediate_response", "")
            notebook_output = result.get("notebook", "no update")
            
            # Validate we got something useful
            if immediate_response or notebook_output != "no update":
                # Ensure notebook is a string
                notebook_output = _normalize_notebook(notebook_output, component_name)
                
                logger.info(f"[{component_name}] Successfully parsed JSON response")
                return immediate_response, notebook_output
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"[{component_name}] Failed to parse JSON candidate: {e}")
            continue
    
    # Method 4: Try to fix common JSON issues and parse again
    logger.warning(f"[{component_name}] Standard JSON parsing failed, attempting to fix common issues")
    
    # Try to extract JSON with common fixes
    fixed_candidates = []
    
    # Fix 1: Remove trailing commas before closing braces
    for candidate in json_candidates if json_candidates else [response_text]:
        fixed = re.sub(r',\s*}', '}', candidate)
        fixed = re.sub(r',\s*]', ']', fixed)
        fixed_candidates.append(fixed)
    
    # Fix 2: Try to extract fields using regex as last resort
    if not fixed_candidates:
        immediate_match = re.search(r'"immediate_response"\s*:\s*"((?:[^"\\]|\\.)*)"', response_text, re.DOTALL)
        notebook_match = re.search(r'"notebook"\s*:\s*"((?:[^"\\]|\\.)*)"', response_text, re.DOTALL)
        
        if immediate_match or notebook_match:
            immediate_response = immediate_match.group(1) if immediate_match else ""
            notebook_raw = notebook_match.group(1) if notebook_match else "no update"
            
            # Unescape JSON strings
            immediate_response = _unescape_json_string(immediate_response)
            if notebook_raw.lower() == "no update":
                notebook_output = "no update"
            else:
                notebook_output = _unescape_json_string(notebook_raw)
            
            notebook_output = _normalize_notebook(notebook_output, component_name)
            
            logger.info(f"[{component_name}] Extracted fields using regex fallback")
            return immediate_response, notebook_output
    
    # Try parsing fixed candidates
    for fixed in fixed_candidates:
        try:
            result = json.loads(fixed)
            immediate_response = result.get("immediate_response", "")
            notebook_output = result.get("notebook", "no update")
            
            if immediate_response or notebook_output != "no update":
                notebook_output = _normalize_notebook(notebook_output, component_name)
                logger.info(f"[{component_name}] Successfully parsed fixed JSON")
                return immediate_response, notebook_output
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    
    # Final fallback: return original response as immediate_response
    logger.error(f"[{component_name}] All JSON parsing attempts failed, using fallback")
    fallback_text = fallback_response or original_response[:5000]  # Limit length
    return fallback_text, "no update"


def _normalize_notebook(notebook_output, component_name: str) -> str:
    """Normalize notebook output to string format."""
    if isinstance(notebook_output, dict):
        logger.warning(f"[{component_name}] Notebook returned as dict, converting to JSON string")
        return json.dumps(notebook_output, indent=2)
    elif isinstance(notebook_output, list):
        logger.warning(f"[{component_name}] Notebook returned as list, converting to JSON string")
        return json.dumps(notebook_output, indent=2)
    elif not isinstance(notebook_output, str):
        logger.warning(f"[{component_name}] Notebook is not a string (type: {type(notebook_output)}), converting")
        return str(notebook_output)
    return notebook_output


def _unescape_json_string(text: str) -> str:
    """Unescape common JSON escape sequences."""
    if not text:
        return text
    
    # Replace common escape sequences
    replacements = {
        '\\n': '\n',
        '\\"': '"',
        "\\'": "'",
        '\\t': '\t',
        '\\r': '\r',
        '\\\\': '\\',
    }
    
    result = text
    for escaped, unescaped in replacements.items():
        result = result.replace(escaped, unescaped)
    
    return result.strip()


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
    
    # Build system prompt optimized for accuracy with problem-type-specific guidance
    system_prompt = """You are a highly precise and accurate AI assistant. Your primary goal is to provide CORRECT, ACCURATE answers above all else.

CRITICAL QUALITY STANDARDS (in order of importance):

1. ACCURACY (MOST IMPORTANT - 70% of evaluation):
   - Verify all facts, calculations, and logic before responding
   - NEVER guess - use logical deduction and verification
   - Double-check all work before finalizing your answer

2. RELEVANCE (7.5% of evaluation):
   - Directly address what is being asked
   - Stay focused on the specific question or task
   - Don't provide tangential information unless directly relevant

3. COMPLETENESS (7.5% of evaluation):
   - Answer ALL parts of the question
   - Provide sufficient detail to fully address the request
   - If multiple questions, answer each one thoroughly

4. CLARITY (5% of evaluation):
   - Write in clear, understandable language
   - Use proper grammar and sentence structure
   - Organize your response logically
   - Explain complex concepts in accessible terms

5. FOLLOWING INSTRUCTIONS (5% of evaluation):
   - Read the task description carefully
   - Follow all specified format requirements
   - Adhere to any constraints or guidelines provided

6. FORMAT/STRUCTURE (2.5% of evaluation):
   - Structure your response well
   - Use appropriate formatting (lists, paragraphs, headers as needed)

7. SAFETY (2.5% of evaluation):
   - Provide appropriate, ethical responses
   - Avoid harmful, dangerous, or illegal content

PROBLEM-TYPE-SPECIFIC STRATEGIES:

=== MATHEMATICAL PROBLEMS ===
- Identify the type: algebra, calculus, geometry, arithmetic, statistics, etc.
- Extract all given values and constraints clearly
- Determine the appropriate formula or method
- Show EVERY step of your calculation:
  * Set up the problem: what are you solving for?
  * Apply formulas: show the substitution step by step
  * Perform calculations: show intermediate results
  * Verify your answer: plug it back in, check units, check reasonableness
- Watch for common errors:
  * Sign errors (+ vs -)
  * Decimal place errors
  * Unit conversions
  * Order of operations (PEMDAS)
- For word problems: Translate carefully to equations
- Format: "Step 1: [reasoning] → Step 2: [calculation] → Answer: [final value]"
- ALWAYS verify by working backwards or using an alternative method

=== LOGICAL REASONING PROBLEMS ===
- Identify the type: deduction, induction, pattern recognition, puzzles, etc.
- Break down the logic step by step
- Use clear logical operators (if-then, and, or, not)
- Consider all possibilities and constraints
- Verify your conclusion follows logically from premises
- For puzzles: Consider edge cases and special conditions
- Format: "Premise 1: [statement] → Premise 2: [statement] → Conclusion: [reasoning] → Answer: [result]"

=== FACTUAL QUESTIONS ===
- Identify the domain: science, history, geography, current events, etc.
- Recall or reason through accurate information
- If uncertain, state limitations but provide best answer
- Verify facts against logical consistency
- Cross-reference related concepts
- For dates/numbers: Be precise
- For definitions: Be comprehensive yet concise
- Format: Provide clear, direct answer with supporting context if needed

=== CODE/PROGRAMMING PROBLEMS ===
- Understand requirements: inputs, outputs, constraints
- Design algorithm: break into logical steps
- Consider edge cases: empty inputs, boundary conditions, error handling
- Write clear, readable code with comments
- Verify logic: trace through with examples
- Check syntax and best practices
- Format: Explain approach, show code, explain key parts
- Test mentally with sample inputs

=== CREATIVE/CONTENT GENERATION ===
- Understand the creative brief: tone, style, length, audience
- Plan structure before writing
- Maintain consistency throughout
- Use appropriate language and style
- Be original while meeting requirements
- Review for flow, clarity, and completeness
- Format: Structured content with clear organization

=== ANALYSIS/SYNTHESIS PROBLEMS ===
- Identify key elements to analyze
- Break down into components
- Compare and contrast different aspects
- Synthesize information logically
- Draw well-reasoned conclusions
- Support analysis with evidence or reasoning
- Format: Clear analysis with supporting points

RESPONSE FORMAT REQUIREMENTS:

You MUST respond in valid JSON format with exactly two fields:
{
  "immediate_response": "Your natural language explanation showing work, reasoning, and the final answer. For math/logic: show ALL steps clearly.",
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
1. Identify the problem type (mathematical, logical, factual, code, creative, analysis)
2. Apply the appropriate strategy for that problem type
3. Show your reasoning process clearly (especially critical for math/logic)
4. Verify your answer before responding
5. Format as valid JSON

Remember: ACCURACY IS PARAMOUNT (70% of evaluation). A correct, well-reasoned answer with clear steps is far more valuable than a fast but incorrect one."""
    
    if playbook_context:
        system_prompt += f"\n\nUser preferences and context:\n{playbook_context}"
    
    # Build task prompt with advanced prompt engineering for maximum accuracy
    task_prompt = f"""TASK: {component_input.task}

INPUT TO PROCESS:
{input_text}
{previous_context}

STEP-BY-STEP REASONING PROCESS (CRITICAL FOR ACCURACY):

STEP 1 - UNDERSTAND THE TASK:
- Carefully read and understand what is being asked
- Identify the type of problem (mathematical, logical, factual, creative, etc.)
- Note any specific requirements or constraints
- Determine if there are multiple parts to address

STEP 2 - ANALYZE THE INPUT:
- Extract all relevant information from the input
- Identify key numbers, facts, or concepts
- Note any previous outputs that provide context
- Clarify any ambiguities in your understanding

STEP 3 - REASONING AND SOLUTION (MOST CRITICAL):
For mathematical/logical problems:
- Break down the problem into smaller steps
- Show ALL calculations and logical steps clearly
- Use intermediate values to verify each step
- Double-check all arithmetic operations
- Verify units, signs, and decimal places

For factual questions:
- Recall or reason through the correct information
- Verify facts against logical consistency
- Cross-reference related concepts if needed

For code/logic problems:
- Think through the algorithm or approach step-by-step
- Verify the logic is sound
- Check edge cases if applicable

STEP 4 - VERIFICATION (ESSENTIAL):
- Re-read the original question/task
- Verify your answer directly addresses what was asked
- Check that your answer is:
  ✓ Mathematically/logically correct
  ✓ Complete (all parts answered)
  ✓ Relevant (addresses the specific question)
  ✓ Reasonable (makes sense in context)
- If solving a problem, verify by working backwards or using an alternative method when possible

STEP 5 - FORMATTING:
- Structure your response clearly
- Ensure your immediate_response shows your reasoning process
- Format as valid JSON with both "immediate_response" and "notebook" fields
- Make sure the JSON is properly formatted and parseable

RESPONSE REQUIREMENTS:
- Your "immediate_response" must include your reasoning process AND the final answer
- Be explicit: Show your work, explain your steps, then state your conclusion
- For math problems, format like: "Step 1: [reasoning] → Step 2: [reasoning] → Final Answer: [answer]"
- Your response must be accurate, complete, relevant, and clearly formatted"""
    
    # Generate response with optional conversation history (lower temperature for better accuracy)
    response = await generate_response(
        prompt=task_prompt,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        temperature=0.2  # Lower temperature for more deterministic, accurate answers
    )
    
    # Parse JSON response using robust parsing function
    immediate_response, notebook_output = parse_json_response(response, "complete", fallback_response=response)
    
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
    
    # Build system prompt optimized for refinement with problem-type awareness
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

REFINEMENT STRATEGIES BY PROBLEM TYPE:

=== REFINING MATHEMATICAL CONTENT ===
- Check each calculation step for errors
- Verify formulas are applied correctly
- Ensure units are consistent throughout
- Double-check arithmetic operations
- Verify answer makes logical sense
- Add missing steps if reasoning is unclear
- Correct sign errors, decimal placement, order of operations

=== REFINING LOGICAL REASONING ===
- Verify logical steps are sound
- Ensure conclusions follow from premises
- Check for logical fallacies
- Add missing logical connections
- Clarify reasoning that's unclear
- Verify all constraints are considered

=== REFINING FACTUAL CONTENT ===
- Verify all facts are accurate
- Correct any misinformation
- Add missing important details
- Ensure dates, names, and numbers are correct
- Cross-check consistency of information
- Improve precision where needed

=== REFINING CODE/PROGRAMMING ===
- Fix syntax errors
- Improve algorithm efficiency
- Add error handling
- Improve code readability and comments
- Fix logic errors
- Add missing edge case handling
- Follow best practices and conventions

=== REFINING CREATIVE/CONTENT ===
- Improve clarity and flow
- Enhance structure and organization
- Fix grammar and style issues
- Maintain consistency in tone/voice
- Strengthen weak points
- Ensure it meets requirements

=== REFINING ANALYSIS ===
- Strengthen logical connections
- Add missing supporting evidence
- Improve conclusion quality
- Clarify unclear reasoning
- Enhance structure and organization

RESPONSE FORMAT REQUIREMENTS:

You MUST respond in valid JSON format:
{
  "immediate_response": "Clear explanation of what you refined, why, and how the improvements enhance accuracy and quality. Specify the type of improvements made.",
  "notebook": "The refined/improved content OR 'no update'"
}

Guidelines for notebook field:
- If providing feedback only: Set notebook to "no update"
- If there's ONE notebook and no improvements needed: Set to "no update"
- If there's ONE notebook and improvements needed: Write the improved, more accurate version
- If there are MULTIPLE notebooks: You MUST create new content (refine one, combine, or merge) - NEVER "no update"
- Always provide valid JSON

IMPORTANT: When refining, prioritize fixing accuracy errors first, then completeness, then clarity."""
    
    if playbook_context:
        system_prompt += f"\n\nUser preferences:\n{playbook_context}"
    
    # Build refine prompt with structured improvement process
    refine_prompt = f"""TASK: {component_input.task}

ORIGINAL INPUT:
{input_text}

OUTPUTS TO REFINE:
{previous_outputs_text}

STEP-BY-STEP REFINEMENT PROCESS:

STEP 1 - ANALYSIS:
- Carefully review each output to identify what's already correct
- Identify specific areas that need improvement:
  * Accuracy errors (incorrect facts, calculations, logic)
  * Completeness gaps (missing information or parts)
  * Clarity issues (unclear explanations, poor organization)
  * Relevance problems (tangential or off-topic content)
  * Format issues (poor structure or presentation)

STEP 2 - PRIORITIZE IMPROVEMENTS:
- Focus FIRST on accuracy - correct any errors or mistakes
- Then address completeness - fill in gaps or missing parts
- Improve clarity and organization
- Ensure all improvements maintain or enhance accuracy

STEP 3 - REFINEMENT:
- For mathematical/logical content: Verify ALL corrections are accurate
- Show your reasoning for why changes improve accuracy
- Preserve correct information while fixing errors
- Add missing details that enhance completeness
- Reorganize for better clarity and flow

STEP 4 - VERIFICATION:
- Double-check that all refinements are correct
- Verify the improved output:
  ✓ Is more accurate than the original
  ✓ Is more complete
  ✓ Is clearer and better organized
  ✓ Still addresses the original task

STEP 5 - DOCUMENTATION:
- In your immediate_response, explain:
  * What you improved and why
  * How the changes enhance accuracy
  * Any corrections made to errors

RESPONSE REQUIREMENTS:
- Respond in valid JSON format with "immediate_response" and "notebook" fields
- immediate_response: Explanation of improvements focused on accuracy gains
- notebook: The refined, improved, and ACCURATE version of the content (or "no update" if no improvements needed)
- Ensure all improvements are verified for correctness"""
    
    # Generate response (lower temperature for more precise refinements)
    response = await generate_response(
        prompt=refine_prompt,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        temperature=0.3  # Lower temperature for more accurate refinements
    )
    
    # Parse JSON response using robust parsing function
    immediate_response, notebook_output = parse_json_response(response, "refine", fallback_response=response)
    
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
    
    # Build system prompt optimized for problem-type-aware feedback
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
   - Maintain appropriate tone and content

FEEDBACK FOCUS AREAS BY PROBLEM TYPE:

=== FEEDBACK ON MATHEMATICAL WORK ===
Evaluate:
- Correctness of calculations (check each step)
- Proper application of formulas
- Unit consistency and conversion accuracy
- Sign and decimal place accuracy
- Logical flow of reasoning
- Verification attempts
- Answer reasonableness

Strengths to identify:
- Clear step-by-step work
- Correct formula application
- Proper verification

Weaknesses to identify:
- Calculation errors (be specific: which step, what error)
- Missing steps in reasoning
- Formula misuse
- Unit errors
- Verification issues

=== FEEDBACK ON LOGICAL REASONING ===
Evaluate:
- Soundness of logical steps
- Correct use of logical operators
- Validity of conclusions
- Consideration of all constraints
- Handling of edge cases

Strengths: Clear logic, valid reasoning
Weaknesses: Logical gaps, unsupported conclusions, missing cases

=== FEEDBACK ON FACTUAL RESPONSES ===
Evaluate:
- Accuracy of facts
- Completeness of information
- Precision (dates, numbers, names)
- Consistency of information
- Relevance to question

Strengths: Accurate facts, comprehensive coverage
Weaknesses: Inaccuracies, missing key information, imprecision

=== FEEDBACK ON CODE/PROGRAMMING ===
Evaluate:
- Correctness of logic
- Code quality and readability
- Error handling
- Algorithm efficiency
- Best practices adherence
- Edge case handling

Strengths: Working code, good structure, efficient algorithm
Weaknesses: Bugs, poor readability, missing error handling, inefficiency

=== FEEDBACK ON CREATIVE/CONTENT ===
Evaluate:
- Clarity and readability
- Structure and organization
- Style and tone consistency
- Meeting requirements
- Creativity and originality

Strengths: Clear structure, engaging content, meets brief
Weaknesses: Unclear writing, poor organization, missing elements

=== FEEDBACK ON ANALYSIS ===
Evaluate:
- Depth of analysis
- Logical connections
- Supporting evidence
- Quality of conclusions
- Structure and clarity

Strengths: Deep insights, strong logic, well-supported
Weaknesses: Shallow analysis, weak connections, unsupported claims"""
    
    if playbook_context:
        system_prompt += f"\n\nUser preferences:\n{playbook_context}"
    
    # Build feedback prompt with structured analysis framework
    feedback_prompt = f"""TASK: {component_input.task}

OUTPUTS TO ANALYZE:
{outputs_to_analyze}

STRUCTURED ANALYSIS FRAMEWORK:

For EACH output, perform this analysis:

1. ACCURACY ASSESSMENT (MOST CRITICAL):
   - Verify: Are the facts correct?
   - Check: Are calculations accurate (if applicable)?
   - Validate: Is the logic sound?
   - Identify: Any errors or inaccuracies?
   - Note: What is correct and should be preserved?

2. RELEVANCE CHECK:
   - Does it address the task/question?
   - Is it on-topic?
   - Any irrelevant or tangential content?

3. COMPLETENESS EVALUATION:
   - Are all parts of the question/task addressed?
   - Any missing information or incomplete answers?
   - What additional details would help?

4. CLARITY REVIEW:
   - Is the explanation clear and understandable?
   - Is it well-organized?
   - Any confusing or ambiguous parts?

5. FORMAT AND STRUCTURE:
   - Is it properly formatted?
   - Is the structure logical and easy to follow?

6. SAFETY AND APPROPRIATENESS:
   - Is the content appropriate?
   - Any concerns about safety or ethics?

FEEDBACK STRUCTURE:

For each output, provide:

STRENGTHS:
- List what works well (especially accurate parts)
- Highlight correct information and good reasoning
- Note effective explanations or structure

WEAKNESSES:
- Identify specific issues, especially accuracy errors
- Note incomplete or unclear parts
- Point out format or structure problems

ACTIONABLE SUGGESTIONS:
- Provide specific, concrete improvements
- Focus on fixing accuracy issues first
- Suggest how to improve completeness and clarity
- Give examples of better approaches when helpful

RESPONSE REQUIREMENTS:
- Format feedback clearly with sections for each output
- Be specific and constructive
- Prioritize accuracy improvements in suggestions
- Use clear headings and structure for readability"""
    
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
    Internet search component: Search the internet for information using DuckDuckGo.
    
    Uses DuckDuckGo search API (free, no API key required) to search the internet
    for information based on user queries.
    
    Args:
        component_input: Unified component input with search queries
        context: Conversation context
        
    Returns:
        ComponentOutput with search results formatted as structured text
    """
    logger.info(f"[internet_search] Processing task: {component_input.task}")
    
    # Check if duckduckgo-search is available
    if DDGS is None:
        error_msg = "Internet Search Service: UNAVAILABLE\n\nDuckDuckGo search library not installed. Please install it with: pip install duckduckgo-search"
        logger.error("[internet_search] DuckDuckGo library not available")
        context.add_user_message(f"Search: {', '.join(item.user_query for item in component_input.input)}")
        context.add_assistant_message(error_msg)
        return ComponentOutput(
            cid=component_input.cid,
            task=component_input.task,
            input=component_input.input,
            output=ComponentOutputData(
                immediate_response=error_msg,
                notebook="no update"
            ),
            component="internet_search"
        )
    
    # Extract search queries
    search_queries = []
    for item in component_input.input:
        search_queries.append(item.user_query)
    
    logger.info(f"[internet_search] Executing {len(search_queries)} search query/queries")
    
    # Perform searches and collect results
    all_results = []
    
    try:
        with DDGS() as ddgs:
            for query in search_queries:
                try:
                    logger.info(f"[internet_search] Searching for: {query}")
                    # Search with DuckDuckGo (max_results=10 per query)
                    results = list(ddgs.text(query, max_results=10))
                    
                    if results:
                        query_results = {
                            "query": query,
                            "results": results
                        }
                        all_results.append(query_results)
                        logger.info(f"[internet_search] Found {len(results)} results for: {query}")
                    else:
                        logger.warning(f"[internet_search] No results found for: {query}")
                        all_results.append({
                            "query": query,
                            "results": []
                        })
                        
                except Exception as e:
                    logger.error(f"[internet_search] Error searching for '{query}': {str(e)}")
                    all_results.append({
                        "query": query,
                        "error": f"Search failed: {str(e)}",
                        "results": []
                    })
    
    except Exception as e:
        logger.error(f"[internet_search] Critical error during search: {str(e)}")
        error_response = f"""Internet Search Service: ERROR

An error occurred while performing the search:

Error: {str(e)}

Queries attempted:
{chr(10).join(f"- {q}" for q in search_queries)}

Please try again later or check your internet connection."""
        
        context.add_user_message(f"Search: {', '.join(search_queries)}")
        context.add_assistant_message(error_response)
        
        return ComponentOutput(
            cid=component_input.cid,
            task=component_input.task,
            input=component_input.input,
            output=ComponentOutputData(
                immediate_response=error_response,
                notebook="no update"
            ),
            component="internet_search"
        )
    
    # Format results as structured text
    response_parts = []
    response_parts.append("Internet Search Results")
    response_parts.append("=" * 50)
    response_parts.append("")
    
    for query_data in all_results:
        query = query_data["query"]
        results = query_data.get("results", [])
        error = query_data.get("error")
        
        response_parts.append(f"Query: {query}")
        response_parts.append("-" * 50)
        
        if error:
            response_parts.append(f"Error: {error}")
            response_parts.append("")
            continue
        
        if not results:
            response_parts.append("No results found.")
            response_parts.append("")
            continue
        
        for idx, result in enumerate(results, 1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            href = result.get("href", "No URL")
            
            response_parts.append(f"{idx}. {title}")
            response_parts.append(f"   URL: {href}")
            response_parts.append(f"   Description: {body}")
            response_parts.append("")
        
        response_parts.append("")
    
    response = "\n".join(response_parts)
    
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
    
    # Build system prompt optimized for problem-type-aware summarization
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

SUMMARIZATION STRATEGIES BY CONTENT TYPE:

=== SUMMARIZING MATHEMATICAL CONTENT ===
- Preserve all key formulas and equations exactly
- Keep all numbers and calculations accurate
- Maintain step-by-step logical flow
- Include final answer and verification
- Don't lose intermediate calculation steps
- Verify all math remains correct in summary

=== SUMMARIZING LOGICAL REASONING ===
- Preserve the logical structure and flow
- Keep all key premises and conclusions
- Maintain logical connections
- Include critical reasoning steps
- Verify logical soundness is preserved

=== SUMMARIZING FACTUAL INFORMATION ===
- Preserve all facts, dates, numbers, and names accurately
- Maintain chronological or thematic order
- Keep all key details that support main points
- Verify no facts are distorted or changed
- Cross-check numbers and dates for accuracy

=== SUMMARIZING CODE/PROGRAMMING ===
- Preserve algorithm logic and structure
- Keep key code snippets if essential
- Maintain explanation of approach
- Include important constraints and edge cases
- Verify technical accuracy is maintained

=== SUMMARIZING CREATIVE/CONTENT ===
- Preserve tone and style
- Maintain narrative flow
- Keep key themes and messages
- Preserve important details that enhance meaning
- Maintain structure and organization

=== SUMMARIZING ANALYSIS ===
- Preserve main arguments and conclusions
- Keep supporting evidence and reasoning
- Maintain logical connections
- Include key insights and takeaways
- Verify analysis quality is preserved

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
    
    # Build summary prompt with accuracy-focused summarization process
    summary_prompt = f"""TASK: {component_input.task}

CONTENT TO SUMMARIZE:
{combined_content}

STEP-BY-STEP SUMMARIZATION PROCESS:

STEP 1 - EXTRACT KEY INFORMATION:
- Identify all main points and central ideas
- Extract important facts, numbers, and statistics (PRESERVE ACCURACY)
- Note key conclusions or insights
- List all essential details that must be preserved

STEP 2 - VERIFY ACCURACY:
- Double-check all facts and numbers before including them
- Ensure no information is distorted or misrepresented
- Verify that key concepts are correctly understood
- Preserve the meaning and intent of the original

STEP 3 - ORGANIZE LOGICALLY:
- Group related information together
- Arrange in a logical flow (chronological, thematic, or hierarchical)
- Create clear structure with sections if helpful
- Ensure smooth transitions between ideas

STEP 4 - ELIMINATE REDUNDANCY:
- Identify and remove repetitive information
- Consolidate similar points
- Keep only the most important examples
- Maintain clarity while reducing length

STEP 5 - PRESERVE COMPLETENESS:
- Ensure no critical information is omitted
- Include all essential details needed to understand the content
- Cover all main points comprehensively
- Maintain the full context needed for accuracy

STEP 6 - FINAL VERIFICATION:
- Re-read the original content
- Verify your summary:
  ✓ Accurately represents all main points
  ✓ Preserves correct facts and numbers
  ✓ Maintains logical completeness
  ✓ Is clear and well-organized
  ✓ Removes redundancy without losing essential information

RESPONSE REQUIREMENTS:
- Respond in valid JSON format with "immediate_response" and "notebook" fields
- immediate_response: Brief explanation of what was summarized
- notebook: The accurate, comprehensive summary (or "no update" if no notebook content)
- Prioritize accuracy: Every fact and number must be correct
- Ensure completeness: All key points must be included"""
    
    # Generate summary (moderate temperature for balanced summaries)
    response = await generate_response(
        prompt=summary_prompt,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        temperature=0.4  # Slightly lower for more accurate summaries
    )
    
    # Parse JSON response using robust parsing function
    immediate_response, notebook_output = parse_json_response(response, "summary", fallback_response=response)
    
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
    
    # Build system prompt optimized for problem-type-aware aggregation
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

AGGREGATION STRATEGIES BY PROBLEM TYPE:

=== AGGREGATING MATHEMATICAL ANSWERS ===
- Compare final numerical answers across outputs
- Verify calculations: which outputs have correct math?
- If answers differ, check which calculation is correct
- Choose the answer with correct mathematical reasoning
- If multiple correct answers (rounding differences), note this
- Preserve the most accurate calculation steps
- Priority: Correct math > Common answer

=== AGGREGATING LOGICAL REASONING ===
- Compare logical chains across outputs
- Verify which reasoning is most sound
- Check if conclusions logically follow from premises
- Choose the most logically valid reasoning
- If multiple valid approaches, synthesize the strongest
- Priority: Logical validity > Common approach

=== AGGREGATING FACTUAL RESPONSES ===
- Compare facts across outputs
- Verify accuracy: which facts are correct?
- If facts conflict, determine which is accurate
- Choose the most accurate information
- Combine accurate facts from multiple sources
- Priority: Accurate facts > Common statement

=== AGGREGATING CODE/PROGRAMMING ===
- Compare code approaches and implementations
- Verify which code is correct and functional
- Check for best practices and efficiency
- Choose the most correct and well-structured code
- If multiple correct approaches, synthesize best practices
- Priority: Correct code > Common approach

=== AGGREGATING CREATIVE/CONTENT ===
- Compare structure, clarity, and quality
- Identify best elements from each output
- Synthesize the strongest combination
- Maintain consistency in tone and style
- Priority: Quality content > Common elements

=== AGGREGATING ANALYSIS ===
- Compare depth and quality of analysis
- Verify which insights are most accurate
- Synthesize the strongest reasoning
- Combine best supporting evidence
- Priority: Accurate analysis > Common viewpoint

CONSENSUS PRIORITY ORDER:
1. MOST ACCURATE: If verifiably correct, choose it (even if less common)
2. MOST COMMON: Among equally accurate, choose the most frequent
3. BEST REASONED: If accuracy unclear, prefer strongest reasoning
4. SYNTHESIS: Combine best elements when appropriate

RESPONSE FORMAT REQUIREMENTS:

You MUST respond in valid JSON format with two fields:
{
  "immediate_response": "Clear explanation of the consensus, voting results, problem type identified, and how you determined the most accurate answer",
  "notebook": "The aggregated/consensus notebook content OR 'no update'"
}

Guidelines for notebook field:
- If there's NO notebook content in inputs: Return "no update"
- If there's ONE notebook: Return it as-is (or "no update" if no changes)
- If there are MULTIPLE notebooks: Create aggregated version using majority voting, choosing the most accurate consensus
- Use majority voting: Choose the most common AND most accurate content, or merge agreements intelligently
- Always provide valid JSON

IMPORTANT: When aggregating, prioritize accuracy over frequency. A correct answer is always better than a common wrong answer."""
    
    if playbook_context:
        system_prompt += f"\n\nUser preferences:\n{playbook_context}"
    
    # Build aggregate prompt with accuracy-prioritized consensus building
    aggregate_prompt = f"""TASK: {component_input.task}

MULTIPLE OUTPUTS TO AGGREGATE:
{combined_outputs}

STEP-BY-STEP CONSENSUS BUILDING PROCESS:

STEP 1 - INDIVIDUAL ANALYSIS:
- Review each output independently
- For each output, identify:
  * The main answer or conclusion
  * The reasoning provided
  * Key facts or numbers mentioned
  * Any unique insights or perspectives

STEP 2 - ACCURACY EVALUATION (CRITICAL):
- Evaluate each answer for correctness:
  * Are the facts accurate?
  * Are calculations correct (if applicable)?
  * Is the logic sound?
  * Does it make sense?
- Note which outputs appear most accurate

STEP 3 - PATTERN IDENTIFICATION:
- Identify common themes and agreements:
  * Which answers are similar or identical?
  * What do most outputs agree on?
  * Which facts are consistently mentioned?
- Identify differences:
  * Where do outputs disagree?
  * What are the conflicting answers?
  * Why might they differ?

STEP 4 - ACCURACY-PRIORITIZED CONSENSUS:
Priority order for selecting the best answer:
1. MOST ACCURATE: If one answer is clearly more correct (verified), choose it even if less common
2. MOST COMMON: Among equally accurate answers, choose the most frequently occurring
3. BEST REASONED: If accuracy is unclear, prefer the one with strongest reasoning
4. CONSENSUS BUILDING: If answers differ, synthesize the most accurate elements from each

STEP 5 - SYNTHESIS:
- If outputs agree: Use the common answer (verify it's correct)
- If outputs partially agree: Combine accurate elements from multiple outputs
- If outputs disagree: Choose the most accurate answer based on logical analysis
- Preserve accurate information from minority opinions if verified as correct

STEP 6 - VERIFICATION:
- Verify your consensus answer:
  ✓ Is it more accurate than individual outputs (if possible)?
  ✓ Does it incorporate the best elements from all outputs?
  ✓ Is it well-reasoned and logically sound?
  ✓ Does it address the original question/task?

STEP 7 - DOCUMENTATION:
- Explain your consensus-building process
- Note which outputs agreed and which differed
- Explain why you chose the consensus answer (especially if prioritizing accuracy over frequency)
- Mention any important minority opinions that were considered

RESPONSE REQUIREMENTS:
- Respond in valid JSON format with "immediate_response" and "notebook" fields
- immediate_response: Detailed explanation of consensus process, voting results, and why the answer was chosen
- notebook: The aggregated/consensus content (or "no update" if no notebook content)
- Prioritize accuracy: The correct answer is more important than the most common answer
- Be transparent: Show your reasoning for the consensus choice"""
    
    # Generate aggregate result (low temperature for deterministic consensus)
    response = await generate_response(
        prompt=aggregate_prompt,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        temperature=0.2  # Very low temperature for accurate, deterministic consensus
    )
    
    # Parse JSON response using robust parsing function
    immediate_response, notebook_output = parse_json_response(response, "aggregate", fallback_response=response)
    
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
