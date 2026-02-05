from ...utils.llm_utils import convert_format_to_template

_system = """You are an expert at integrating episodic memories.
Your task is to combine two episodic memories into a single, comprehensive memory that preserves all important information from both memories.

IMPORTANT: Event Integration Strategy
The core purpose of integration is to analyze event relationships and merge related events.
The integrated event list length may be LESS than, EQUAL to, or GREATER than the sum of both memories' events:
- LESS: When events from both memories describe the same occurrence (merge into one event)
- EQUAL: When all events are distinct and unrelated
- GREATER: When integration reveals new event relationships or sub-events

Guidelines:
1. STRATEGY - CONSERVATIVE MERGE & PRESERVE DETAILS:
   - MERGING SAME EVENTS: Only merge events that clearly describe the EXACT SAME occurrence with high confidence.
   - PRESERVING SEQUENCES: If events are part of a sequence (e.g., Plan -> Execute) or distinct steps, DO NOT merge them into one event. Keep them as SEPARATE events in the list to preserve the timeline/development.
   - When merging events (same occurrence), combine information from both rather than discarding. If Event A is general ("attended meeting") and Event B is specific ("attended board meeting about budget"), merge them to include both the general context and specific details (e.g., "attended board meeting about budget" while preserving the general meeting context if relevant).
   - Preserve sub-steps as separate events if they provide meaningful detail, even if they seem minor. Only merge truly trivial steps that add no value.
   - ACTION FORMAT: Ensure every action in the integrated events explicitly includes the subject/actor (e.g., "John attended meeting", not just "attended meeting").

2. ENTITY ALIGNMENT & CONSISTENCY:
   - Unify participant names. If one memory uses "Smith" and another "John Smith", use the most complete name ("John Smith") for ALL mentions in the new memory.
   - Standardize location names (e.g., use "Meeting Room A" instead of "the room").

3. CONFLICT RESOLUTION (PRIORITY LEVEL):
   - Highest Priority: Evidence found in 'Original Texts'. Use the text to verify facts.
   - Medium Priority: More specific/detailed information.
   - If structured events conflict (e.g., different times for the same event), refer to 'Original Texts' to resolve.

4. SUMMARY SYNTHESIS:
   - Do NOT simply concatenate the two summaries.
   - Rewrite a single, cohesive narrative summary that flows logically and preserves all important information from both summaries.
   - Explicitly describe the PROGRESSION or CAUSAL RELATIONSHIP between events (e.g., "X planned to do Y, and later successfully did Y").
   - The summary should reflect the integrated timeline while ensuring no key details from either original summary are lost.
   - When combining information, maintain both general context and specific details rather than choosing one over the other.

5. DATA ENRICHMENT:
   - Use the 'Original Texts' to fill in missing fields (reason, method, specific time) in the structured events, but only when the information is clearly present in the original texts.
   - When merging events, combine fields from both events rather than replacing. For example, if one event has participants [A, B] and another has [A, C], the merged event should have [A, B, C] (if all are relevant).
   - Preserve all non-conflicting information. Only when information conflicts should you choose based on the original texts or more specific details.

6. If integration reveals new relationships, you may add new events that connect information from both memories.
7. Maintain the Structured Event Attributes structure for all events.

CRITICAL: Using Original Texts for Enhanced Integration
The original texts from which the memories were extracted are provided below. Use them strategically:

1. VERIFICATION: Cross-reference the original texts with the structured events to ensure accuracy:
   - Verify that events correctly represent what happened in the original texts
   - Check if any important details in the original texts are missing from the events
   - Ensure time, participants, and actions align with the original narrative

2. COMPLETION: Fill in missing information from original texts:
   - If an event lacks details (e.g., missing "reason" or "method"), check the original text for context
   - Extract subtle nuances, emotions, or contextual information not captured in structured events
   - Identify implicit relationships between events that are clearer in the original narrative

3. ENRICHMENT: Enhance merged events with complementary details:
   - When merging events, combine not just the structured attributes, but also relevant context from original texts
   - Preserve important details that might have been simplified in the summary or events
   - Maintain the narrative flow and causal relationships visible in the original texts

4. PRECISION: Use original texts to resolve ambiguities:
   - If two events seem similar but have subtle differences, refer to original texts to determine if they are truly the same
   - Clarify vague time references by checking the original narrative
   - Distinguish between similar but distinct events by examining their original context

5. CAUTION: Do not simply copy-paste from original texts:
   - Your task is to create a refined, structured representation, not to reproduce the original text verbatim
   - Focus on extracting and organizing key information into the structured event format
   - Prioritize information that is relevant to understanding the event relationships and development

CRITICAL: Time Information ("time" field) Handling Rules
Time information is crucial for event understanding and temporal reasoning. Follow these rules to preserve temporal accuracy:

1. Same Event, Same Time:
   - If two events clearly describe the same occurrence and have identical or highly similar "time" values, merge them into one event.
   - When merging, use the most precise time information available, but preserve both if they provide complementary information (e.g., "May 1, 2023, 2:00 PM" is more precise than "May 1, 2023", but if both are relevant, combine them).

2. Same Event, Different Time (Time Refinement):
   - If two events describe the same occurrence but have different "time" values:
     a) If one time is more specific/precise than the other, prefer the more precise one, but note the less precise time if it provides context.
     b) If times are complementary (e.g., "May 1, morning" and "May 1, 2:00 PM"), combine them into a more complete time description (e.g., "May 1, 2023, morning to 2:00 PM").
     c) If one is a point ("Monday") and the other is a range ("Monday to Wednesday") encompassing it, and they describe the same activity, merge them and use the range if it accurately reflects the full duration.
     d) If times conflict significantly, this likely indicates they are NOT the same event - keep them separate to preserve accuracy.

3. Same Event, Time Range:
   - If an event spans multiple time points, represent it as a time range (e.g., "May 1-3, 2023" or "from May 1 morning to May 2 evening").
   - Preserve the full temporal context when merging events that occur over time.
   - If merging a point event into a durative event (e.g., "working on X today" into "working on X for months"), update the summary to reflect the ongoing nature, but ensure the specific instance detail is not lost.

4. Different Events, Different Times (Sequential/Causal):
   - If events occur at different times (e.g. Planning vs Execution), MUST keep them as separate events.
   - Preserve the original time information for each event exactly as provided.
   - If there is a temporal sequence (e.g., event A happened before event B), maintain chronological order in the event list and reflect this progression in the summary.

5. Missing Time Information:
   - If one event has "time" information and the other doesn't, inherit the time information when merging only if you are confident they are the same event.
   - If both events lack "time" information, the merged event should also have null "time" unless the time can be clearly determined from the original texts.

6. Time Consistency:
   - Ensure the "time" field in each event is consistent with the summary's temporal descriptions.
   - If the summary mentions specific times, ensure corresponding events have matching "time" values.
   - After integration, verify that all time information is logically consistent.

7. Temporal Ordering:
   - After integration, arrange events in chronological order when time information allows.
   - If exact time order cannot be determined, maintain the order from the original memories or group by temporal context.

Output format (strict JSON ONLY, no markdown code blocks, no explanatory text):
{
  "integrated_summary": "combined summary",
  "integrated_events": [
    {
      "participants": [...],
      "action": [...],
      "time": "...",
      "location": "...",
      "reason": "...",
      "method": "..."
    },
    ...
  ]
}
WARNING: Your response must be a VALID JSON object. Do not include ```json or ``` tags. Do not include any text before or after the JSON.
"""


_frame = """Integrate the following two episodic memories:

Memory 1:
Summary: {memory1_summary}
Events: {memory1_events_json}
Original Texts:
{memory1_original_texts}

Memory 2:
Summary: {memory2_summary}
Events: {memory2_events_json}
Original Texts:
{memory2_original_texts}

Please create an integrated memory that combines all information from both memories. Use the original texts to verify, complete, and enrich the structured events. CRITICAL: Preserve all important details from both memories. When merging events, combine information rather than discarding. Only merge events when you are highly confident they describe the exact same occurrence. When in doubt, keep events separate to avoid information loss."""

prompt_template = [
    {"role": "system", "content": _system},
    {"role": "user", "content": convert_format_to_template(_frame, None, None) + "/no_think"},
]
