# ner_system = """Your task is to extract named entities from the given paragraph. 
# Respond with a JSON list of entities.
# """

# one_shot_ner_paragraph = """Radio City
# Radio City is India's first private FM radio station and was started on 3 July 2001.
# It plays Hindi, English and regional songs.
# Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""


# one_shot_ner_output = """{"named_entities":
#     ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
# }
# """

ner_system = """Your task is to extract named entities from the given dialogue text.
The dialogue format is: [t=timestamp, speaker=speaker_name] dialogue_content

Extract the following types of entities:
- Person names (including the speaker)
- Locations
- Dates and times (from both timestamp and dialogue content)
- Organizations, games, movies, books, etc.
- Events and activities
- Specific objects or items mentioned

Respond with a JSON list of entities.
"""

one_shot_ner_paragraph = """[t=10:15 am on 15 March, 2023, speaker=Alex] Hi Sarah! Did you hear about the new Modern Art Exhibition downtown?
[t=10:16 am on 15 March, 2023, speaker=Sarah] Hey Alex! Yes, I saw a poster for it yesterday. It looks amazing!
[t=10:17 am on 15 March, 2023, speaker=Alex] I'm planning to go this Saturday. Do you want to come along?"""

one_shot_ner_output = """{"named_entities":
    ["Alex", "Sarah", "Modern Art Exhibition", "downtown", "yesterday", "poster", "this Saturday", "15 March 2023", "10:15 am"]
}
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}/no_think"}
]
