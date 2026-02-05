from ...utils.llm_utils import convert_format_to_template

_system = """You are an expert at selecting related chunks based on episodic memories.\nReturn strict JSON: {\n  \"related_chunk_ids\": [chunk_id, ...]\n}\nOnly include chunk_ids from candidates.\n"""

_frame = """Target chunk:\nchunk_id: {target_chunk_id}\nsummary: {target_summary}\nevents: {target_events_json}\n\nCandidates(JSONL):\n{candidate_chunks_jsonl}\n\nTop-K to select: {k_value}\n"""

prompt_template = [
    {"role": "system", "content": _system},
    {"role": "user", "content": convert_format_to_template(_frame, None, None) + "/no_think"},
]


