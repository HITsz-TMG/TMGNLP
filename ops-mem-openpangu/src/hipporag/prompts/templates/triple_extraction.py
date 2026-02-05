from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template


ner_conditioned_re_system = """Your task is to extract relationship triples from the given dialogue text and named entity lists.
The dialogue format is: [t=timestamp, speaker=speaker_name] dialogue_content

Important requirements:
1. Resolve pronouns (it, that, they, etc.) by referring to previous dialogue turns or context.
2. Extract temporal relationships (e.g., "last week", "yesterday", "21 January 2022").
3. Extract speaker relationships and actions (e.g. who said what, who did what).
4. Extract events with participants, time, and location.
5. Each triple should be: [subject, predicate, object].
6. Use the provided named entities as much as possible.

Respond with a JSON list of triples.
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
            ["Alex", "speaks to", "Sarah"],
            ["Alex", "mentions", "Modern Art Exhibition"],
            ["Modern Art Exhibition", "located in", "downtown"],
            ["Sarah", "speaks to", "Alex"],
            ["Sarah", "saw", "poster"],
            ["poster", "promotes", "Modern Art Exhibition"],
            ["Alex", "plans to visit", "Modern Art Exhibition"],
            ["Alex", "invites", "Sarah"],
            ["visit", "scheduled for", "this Saturday"],
            ["15 March 2023", "is time of", "dialogue"]
    ]
}
"""


prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None) + "/no_think"}
]
