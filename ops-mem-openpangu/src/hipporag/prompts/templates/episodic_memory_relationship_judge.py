from ...utils.llm_utils import convert_format_to_template

_system = """You are an expert at judging whether two episodic memories should be related (but not integrated).
Your task is to determine if two episodic memories are related but should NOT be merged into a single memory.

Difference from integration judgment:
- Integration judgment: Two memories should be merged into one memory (same event/topic, can be combined)
- Relationship judgment: Two memories are related but should remain separate (e.g., different times/locations/events about the same entity)

Guidelines:
1. Memories should be related if they describe the same entity, person, or topic but at different times/locations.
2. Memories should be related if they are part of a sequence of events but should remain separate.
3. Memories should be related if they provide complementary information about the same topic.
4. Memories should NOT be related if they are about completely different topics.
5. Memories should NOT be related if they should be integrated instead (use integration judgment for that).

Output format (strict JSON):
{
  "should_relate": true/false,
  "reason": "brief explanation"
}
"""

_frame = """Memory 1:
Chunk IDs: {memory1_chunk_ids}
Summary: {memory1_summary}
Events: {memory1_events_json}

Memory 2:
Chunk IDs: {memory2_chunk_ids}
Summary: {memory2_summary}
Events: {memory2_events_json}

Should these two memories be related (but not integrated)? Please provide your judgment."""

prompt_template = [
    {"role": "system", "content": _system},
    {"role": "user", "content": convert_format_to_template(_frame, None, None) + "/no_think"},
]

