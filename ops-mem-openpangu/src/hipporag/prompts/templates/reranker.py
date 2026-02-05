from ...utils.llm_utils import convert_format_to_template


reranker_system = """You are an expert fact reranker.
Given a question and a list of candidate facts (triples), select and order the most relevant facts to the question.

Output Policy:
- You MUST respond using the exact fielded format below.
- Start with the field header [[ ## fact_after_filter ## ]], followed by a single JSON or Python dict matching the schema: {"fact": [[subject, predicate, object], ...]}
- End with the marker [[ ## completed ## ]]
- Do not add any extra commentary.
"""


# One-shot demo
one_shot_question = (
    "When was Neville A. Stanton's employer founded?"
)

one_shot_facts_before_filter = {
    "fact": [
        ["Neville A. Stanton", "employer", "University of Southampton"],
        ["University of Southampton", "founded in", "1862"],
        ["University of Southampton", "ranked in", "top 100 research universities"],
        ["PlanetRadiocity.com", "launched in", "May 2008"],
    ]
}

one_shot_reranker_input_frame = """[[ ## question ## ]]
{question}

[[ ## fact_before_filter ## ]]
{fact_before_filter}

Respond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`.
"""

one_shot_reranker_input = one_shot_reranker_input_frame.format(
    question=one_shot_question,
    fact_before_filter=str(one_shot_facts_before_filter)
)

one_shot_reranker_output = """[[ ## fact_after_filter ## ]]
{"fact": [
  ["Neville A. Stanton", "employer", "University of Southampton"],
  ["University of Southampton", "founded in", "1862"]
]}

[[ ## completed ## ]]"""


prompt_template = [
    {"role": "system", "content": reranker_system},
    {"role": "user", "content": one_shot_reranker_input},
    {"role": "assistant", "content": one_shot_reranker_output},
    {
        "role": "user",
        "content": convert_format_to_template(
            original_string=one_shot_reranker_input_frame,
            placeholder_mapping=None,
            static_values=None
        )
    }
]



