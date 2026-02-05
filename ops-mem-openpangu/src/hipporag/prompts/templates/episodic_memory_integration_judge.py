from ...utils.llm_utils import convert_format_to_template

_system = """You are an expert at judging whether two episodic memories should be integrated.
Your task is to determine if two episodic memories refer to the EXACT SAME specific real-world occurrence OR form a STRONGLY RELATED SEQUENTIAL NARRATIVE (e.g., planning -> execution).

CRITICAL: You do not have the original text, so you must rely on LOGICAL COMPARISON of the structured data.

Guidelines:

1. STRUCTURAL FIELD COMPARISON (Primary Check):
   - Time:
     * STRICT: If both have specific time points and they are DISJOINT (e.g., "Monday" vs "Wednesday") and NOT part of a continuous sequence OR causal chain, REJECT.
     * FLEXIBLE: If one is more precise than the other (e.g., "May 1" vs "May 1, 2:00 PM"), ACCEPT.
     * FLEXIBLE: If times are compatible (e.g., "morning" and "2:00 PM" on same day), ACCEPT.
     * CONTINUITY: If timestamps are sequential (e.g., "10:00" and "10:30") and describe a continuous flow of action, ACCEPT.
     * CAUSALITY: If timestamps are distinct but events are causally linked steps of a larger goal (e.g., Jan 20: Plan, Jan 27: Execute) involving same participants, ACCEPT.
     * RANGE: If one is a time point and the other is a time range ENCOMPASSING that point (e.g., "Monday" and "Monday to Wednesday"), and they describe the same activity, ACCEPT.

   - Location:
     * STRICT: If locations are DISTINCT and not related (e.g., "Room A" vs "Room B"), REJECT.
     * FLEXIBLE: If one location contains the other (e.g., "Office" vs "Office Room 101"), ACCEPT.

   - Participants:
     * STRICT: If key participants are COMPLETELY DIFFERENT (e.g., [Alice, Bob] vs [Charlie, Dave]), REJECT.
     * FLEXIBLE: If participants overlap or are name variants (e.g., "Smith" vs "John Smith"), ACCEPT.

2. CONFLICT VETO (Absolute Rejection):
   - If summaries contain CONTRADICTORY OUTCOMES (e.g., "won" vs "lost", "succeeded" vs "failed"), REJECT.
   - These indicate DIFFERENT events or different stages of a long process.

3. "SAME TYPE" vs "SAME INSTANCE" TRAP:
   - Two "morning jogs" on DIFFERENT days are SAME TYPE but DIFFERENT INSTANCES -> REJECT.
   - Two "meetings with boss" about DIFFERENT projects are DIFFERENT INSTANCES -> REJECT.
   - Use structural fields (time/location) to distinguish instances.

4. INTEGRATE IF:
   - Structural fields (time/location/participants) are COMPATIBLE (not conflicting).
   - Summaries describe the SAME specific occurrence OR sequential parts of a continuous event.
   - Events are CAUSALLY RELATED stages of a specific project or goal (e.g., planning -> execution) with shared participants.
   - One memory appears to be a SUBSET, more detailed version, or a CONTINUATION of the other.

5. DEFAULT TO SEPARATE:
   - When in doubt, prefer keeping memories separate to preserve precision.
   - Only integrate when you have STRONG EVIDENCE they are the same instance.

Output format (strict JSON, no markdown, no comments):
{
  "should_integrate": true/false,
  "reason": "Cite specific matching/conflicting fields (time/location/participants) or explain the conflict"
}
"""

# _system = """You are an expert at judging whether two episodic memories should be integrated.
# Your task is to determine if two episodic memories contain related information that should be combined into a single memory.
#
# Guidelines:
# 1. Memories should be integrated if they describe the same event, person, or topic.
# 2. Memories should be integrated if they are part of a continuous narrative.
# 3. Memories should NOT be integrated if they are about completely different topics.
# 4. Memories should NOT be integrated if integration would cause information loss.
#
# Output format (strict JSON):
# {
#   "should_integrate": true/false,
#   "reason": "brief explanation"
# }
# """

_frame = """Memory 1:
Summary: {memory1_summary}
Events: {memory1_events_json}

Memory 2:
Summary: {memory2_summary}
Events: {memory2_events_json}

Should these two memories be integrated? Please provide your judgment."""

prompt_template = [
    {"role": "system", "content": _system},
    {"role": "user", "content": convert_format_to_template(_frame, None, None) + "/no_think"},
]

