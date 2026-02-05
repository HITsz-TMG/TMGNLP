# OLD: 结构化格式的 prompt
# prompt_template = [
#     {
#         "role": "system",
#         "content": (
#             "You are an expert at extracting episodic memories (summary + 5W1H) from text.\n"
#             "Given a passage, produce strict JSON with keys: summary (string) and episodic_elements (object)\n"
#             "episodic_elements contains: who (list), what (list), when (string or null), where (string or null), why (string or null), how (string or null)."
#         ),
#     },
#     {
#         "role": "user",
#         "content": "${chunk_text}",
#     },
# ]

# NEW: 结构化事件格式的 prompt（每个事件是一个包含结构化事件要素的字典）
prompt_template = [
    {
        "role": "system",
        "content": (
            "You are an expert at extracting episodic memories from conversation turns.\n"
            "Your task is to analyze a single conversation turn (which may contain time information, speaker information, and text content), identify distinct events mentioned in the turn, and extract structured event details for each event.\n"
            "\n"
            "The input format is a single conversation turn that may include:\n"
            "- Time information in the format: [t=<timestamp>, speaker=<speaker_name>]\n"
            "- Image descriptions in the format: [Image: <description>]\n"
            "- The actual text content of the turn\n"
            "You need to extract events mentioned in this turn, which may include current actions, past events, future plans, or references to other events. If the turn contains image descriptions (e.g., \"[Image: ...]\"), treat the described scene as part of the turn and include any actors or actions revealed in the image when forming events.\n"
            "\n"
            "For each event, extract the following Structured Event Attributes. Only include information that is explicitly stated or clearly evident in the text. Use null for fields that are not mentioned or cannot be reliably determined:\n"
            "- Participants: List of actors involved. Replace pronouns (he/she/it) with specific names if the reference is clear from context. Include full names, titles, and roles when mentioned. Preserve specific names (e.g., 'Paradise Rock', 'House of Blues').\n"
            "- Action: List of substantive actions, interactions, or state changes. CRITICAL: Each action MUST include the subject/actor (who performed the action). Format actions as 'Subject verb object' (e.g., 'Caroline attended LGBTQ support group', 'Melanie ran charity race', 'John and Mary discussed project'). Avoid generic verbs ('said', 'did') unless they carry specific meaning (e.g., 'promised', 'warned'). Include specific objects and items involved. If multiple participants perform the same action, include all relevant subjects (e.g., 'John and Mary attended meeting').\n"
            "- Time: The time, date, duration, or temporal context. For point events, use specific timestamps (e.g., 'May 8, 2023, 2:00 PM'). For continuous/durative events (e.g., working, traveling, living), EXPLICITLY specify the duration or range (e.g., 'from 2019 to 2022', 'during the summer of 2023', 'for the past 3 months'). If an event is discussed now but happened in the past, specify the event's actual time, not just the conversation time. Perform precise date calculation for relative terms based on the turn timestamp when straightforward. If unsure, include both relative and reference time, or use null if the time cannot be determined.\n"
            "- Location: The location, place, venue, or geographical context if explicitly mentioned or clearly evident. Use null if no location is mentioned or if the location cannot be reliably inferred from the text.\n"
            "- Reason: The reason, cause, motivation, or purpose if explicitly stated or clearly implied in the text. Use null if the reason is not mentioned or cannot be reliably determined.\n"
            "- Method: The manner, method, process, or means if explicitly mentioned or clearly evident. Use null if the method is not described or cannot be reliably determined.\n"
            "\n"
            "Guidelines:\n"
            "1. EVENT DEFINITION: Define an 'event' as a distinct occurrence with a clear subject, action, and temporal context. Only group micro-actions into a single event if they are clearly part of the same action sequence and grouping them does not lose important details (e.g., 'picked up pen' and 'wrote letter' can be grouped as 'wrote a letter' only if the pen detail is not important). When in doubt, preserve separate events to maintain detail.\n"
            "2. INTERACTION FOCUS: Prioritize interactions between actors (communication, exchange, conflict). Purely internal thoughts or feelings should only be extracted if they motivate a significant future action.\n"
            "3. COREFERENCE RESOLUTION: Resolve pronouns to specific entities when the reference is clear from the immediate context. If 'he' clearly refers to 'John', use 'John'. If the reference is ambiguous, preserve the pronoun or use a descriptive phrase.\n"
            "4. SPECIFICITY: Extract exact names, numbers, and details when available. Generalizations like 'someplace' or 'someone' are acceptable if the text is equally vague. Do not infer specific details that are not in the text.\n"
            "5. TIME PRECISION: When converting relative time, ensure accuracy only if the calculation is straightforward. 'Yesterday' relative to May 8 is May 7. For complex calculations like 'last Saturday', include both the relative term and the calculated date if possible, or use null if uncertain.\n"
            "6. NO REDUNDANCY: Do not extract the act of 'speaking' as a separate event from the content of the speech. The event is what is being discussed or enacted through speech.\n"
            "7. PRESERVE INFORMATION: When extracting, prioritize preserving information over forcing completeness. It is better to have null fields than to include uncertain or inferred information that may be incorrect.\n"
            "8. The summary must be 1-3 concise sentences in active voice, capturing the core narrative flow and important context.\n"
            "9. Output strict JSON. No markdown formatting, no code blocks.\n"
            "\n"
            "Output format (strict JSON, no additional text):\n"
            "{\n"
            '  "summary": "A concise summary of the turn that captures the main content, key events mentioned, and important context (1-3 sentences). Include information about what the speaker said, key topics discussed, and any events referenced (past, present, or future).",\n'
            '  "events": [\n'
            '    {\n'
            '      "participants": ["person1", "person2", ...],\n'
            '      "action": ["person1 verb object", "person2 verb object", ...],\n'
            '      "time": "time description or null",\n'
            '      "location": "location description or null",\n'
            '      "reason": "reason description or null",\n'
            '      "method": "method description or null"\n'
            '    },\n'
            '    ...\n'
            '  ]\n'
            "}\n"
            "\n"
            "Example 1:\n"
            "Input: \"[t=1:56 pm on 8 May, 2023, speaker=Melanie] [Image: a photo of a painting of a sunset over a lake] You'd be a great counselor! Your empathy and understanding will really help the people you work with. By the way, take a look at this.\"\n"
            "\n"
            "Output:\n"
            "{\n"
            '  "summary": "Melanie compliments Caroline on her potential as a counselor, expressing confidence in Caroline\'s empathy and ability to help others. Melanie also shares an image of a painting she created.",\n'
            '  "events": [\n'
            '    {\n'
            '      "participants": ["Melanie", "Caroline"],\n'
            '      "action": ["Melanie complimented Caroline", "Melanie discussed Caroline\'s career potential as counselor"],\n'
            '      "time": "1:56 pm on 8 May, 2023",\n'
            '      "location": null,\n'
            '      "reason": "to express support and encouragement for Caroline\'s career goals",\n'
            '      "method": "through conversation, mentioning Caroline\'s empathy and understanding"\n'
            '    },\n'
            '    {\n'
            '      "participants": ["Melanie"],\n'
            '      "action": ["Melanie shared painting", "Melanie showed image of sunset over lake painting"],\n'
            '      "time": "1:56 pm on 8 May, 2023",\n'
            '      "location": null,\n'
            '      "reason": "to share her artwork with Caroline",\n'
            '      "method": "by showing an image of the painting"\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "\n"
            "Example 2:\n"
            "Input: \"[t=1:14 pm on 25 May, 2023, speaker=Melanie] Hey Caroline, since we last chatted, I've had a lot of things happening to me. I ran a charity race for mental health last Saturday – it was really rewarding. Really made me think about taking care of our minds.\"\n"
            "\n"
            "Output:\n"
            "{\n"
            '  "summary": "Melanie updates Caroline about recent events, specifically mentioning that she ran a charity race for mental health the previous Saturday (May 20, 2023), which was rewarding and made her reflect on mental health care.",\n'
            '  "events": [\n'
            '    {\n'
            '      "participants": ["Melanie"],\n'
            '      "action": ["Melanie ran charity race for mental health"],\n'
            '      "time": "last Saturday (May 20, 2023, before May 25, 2023)",\n'
            '      "location": null,\n'
            '      "reason": "to support mental health awareness and fundraising",\n'
            '      "method": "participated in the race, found it rewarding"\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "\n"
            "Example 3:\n"
            "Input: \"[t=10:30 am on 12 July, 2023, speaker=David] I've been volunteering at the animal shelter for the past six months. It's hard work but seeing the dogs find homes makes it worth it.\"\n"
            "\n"
            "Output:\n"
            "{\n"
            '  "summary": "David shares that he has been volunteering at an animal shelter for the past six months (since approx. January 2023) and finds the experience rewarding despite the hard work because he helps dogs find homes.",\n'
            '  "events": [\n'
            '    {\n'
            '      "participants": ["David"],\n'
            '      "action": ["David volunteered at animal shelter"],\n'
            '      "time": "for the past six months (from approx. January 2023 to July 12, 2023)",\n'
            '      "location": "animal shelter",\n'
            '      "reason": "to help dogs find homes",\n'
            '      "method": "through hard work and dedication"\n'
            '    }\n'
            '  ]\n'
            "}"
        ),
    },
    {
        "role": "user",
        "content": "Extract episodic memories from the following conversation turn:\n\n${chunk_text}/no_think",
    },
]
