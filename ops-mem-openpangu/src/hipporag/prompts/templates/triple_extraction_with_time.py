from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """Your task is to extract relationship triples with temporal context from the given dialogue text and named entity lists.
The dialogue format is: [t=timestamp, speaker=speaker_name] dialogue_content

Important requirements:
1. Resolve pronouns (it, that, they, etc.) by referring to previous dialogue turns or context.
2. Extract temporal relationships (e.g., "last week", "yesterday", "21 January 2022").
3. Extract speaker relationships and actions (e.g. who said what, who did what).
4. Extract events with participants, time, and location.
5. Each entry should be a quadruple: [subject, predicate, object, time].
   - If a specific time is mentioned for the fact, include it.
   - If the fact happens at the current conversation time, use the timestamp provided in the input (e.g., "15 March 2023" from [t=...]).
   - If no specific time is applicable or extractable, use null.
6. Use the provided named entities as much as possible.

Respond with a JSON list of quadruples under the key "triples".
"""


ner_conditioned_re_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""


ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)


ner_conditioned_re_output = """{"triples": [
            ["Alex", "speaks to", "Sarah", "15 March 2023"],
            ["Alex", "mentions", "Modern Art Exhibition", "15 March 2023"],
            ["Modern Art Exhibition", "located in", "downtown", null],
            ["Sarah", "speaks to", "Alex", "15 March 2023"],
            ["Sarah", "saw", "poster", "yesterday"],
            ["poster", "promotes", "Modern Art Exhibition", null],
            ["Alex", "plans to visit", "Modern Art Exhibition", "this Saturday"],
            ["Alex", "invites", "Sarah", "15 March 2023"],
            ["visit", "scheduled for", "this Saturday", "this Saturday"],
            ["15 March 2023", "is time of", "dialogue", "15 March 2023"]
    ]
}
"""


prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None) + "/no_think"}
]

