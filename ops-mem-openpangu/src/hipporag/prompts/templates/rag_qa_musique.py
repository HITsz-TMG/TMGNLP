one_shot_rag_qa_docs = (
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
)


one_shot_ircot_demo = (
    f'{one_shot_rag_qa_docs}'
    '\n\nQuestion: '
    f"When was Neville A. Stanton's employer founded?"
    '\nThought: '
    f"The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
    f"\nAnswer: 1862"
    '\n\n'
)


rag_qa_system = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
    'Carefully read through the provided passages to extract the information needed to answer the question accurately. '
    'Use the passage content comprehensively to understand key events, entities, temporal relationships, and causal connections. '
    '\n\n'
    'Answer Format Requirements:\n'
    '1. Your response must start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions.\n'
    '2. Conclude with "Answer: " followed by ONLY the answer itself - no additional explanations, no citations, no references to passages, no punctuation unless part of the answer (e.g., dates, names with punctuation). After providing the answer, STOP immediately.\n'
    '3. The answer should be concise and direct - extract only the essential information needed to answer the question.\n'
    '4. For date questions, provide the year or date as it appears in the context (e.g., "1862" or "May 30, 2003").\n'
    '5. For person/entity questions, provide the complete name as it appears in the context.\n'
    '6. For location questions, provide the specific location name.\n'
    '7. Ensure your answer contains all key information words from the context that are relevant to the question - completeness is crucial for accurate evaluation.\n'
    '8. Avoid adding words like "the", "a", "an" unless they are essential parts of the answer (e.g., "The Beatles" should include "The").\n'
    '9. Do not include phrases like "the answer is", "according to", "Passage X states", or similar explanatory text in your answer - only the factual answer itself.\n'
    '10. If the question cannot be answered based on the provided context, output exactly "Not mentioned in the conversation".'
)

one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nQuestion: "
    "When was Neville A. Stanton's employer founded?"
    '\nThought: '
)


one_shot_rag_qa_output = (
    "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
    "\nAnswer: 1862"
)


prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
    {"role": "user", "content": "${prompt_user}"}
]
