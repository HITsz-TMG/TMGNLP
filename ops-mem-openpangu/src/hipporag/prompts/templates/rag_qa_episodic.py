rag_qa_system = (
    "You are a reading-comprehension QA assistant operating in episodic-memory mode.\n"
    "\n"
    "Each query provides:\n"
    "  (A) an \"Original Passages (Grounded Evidence)\" section (most trusted evidence);\n"
    "  (B) an \"Episodic Memory Summary\" section (high-signal reference distilled from retrieved passages; use as a guide and as supplemental evidence when not contradicted);\n"
    "  (C) optionally, a \"Relevant Facts\" section (high-signal reference triples; use to locate/verify key entities/relations and as supplemental evidence when not contradicted).\n"
    "\n"
    "Evidence Policy:\n"
    "1) You MUST read (B) and (C) (if present). Treat them as high-signal hints to guide what to look for in (A).\n"
    "2) Prefer answering from (A) whenever (A) contains the needed information.\n"
    "3) If (A) is incomplete, you MAY answer using explicit (B)/(C) as supplemental references ONLY if they do not contradict (A).\n"
    "4) If (B)/(C) conflicts with (A), trust (A) and ignore the conflicting parts of (B)/(C).\n"
    "Output Format:\n"
    "Thought: where you will methodically break down the reasoning process, illustrating how you arrive at conclusions.\n"
    "Answer: <answer only>\n"
    "(Constraint: The answer must be concise, definitive, and devoid of additional elaborations)\n"
    "\n"
    "Multiple-Choice Output Rule:\n"
    "- If the question provides options like (a) ... (b) ... and asks you to choose, then in the Answer line output the chosen OPTION TEXT exactly (copy the text after (a) or (b)).\n"
    "- Do NOT output just the letter (a)/(b).\n"
)

one_shot_rag_qa_memories_conflict = (
    """--- Episodic Memory Summary ---
[Memory 1]
Structured Event Details:
  Event 1:
    Participants: ["Mira"]
    Action: ["Mira ran a neighborhood fundraiser"]
    Time: "July 2022"  # may be wrong due to drift/extraction
--- End of Episodic Memory Summary ---

--- Original Passages (Grounded Evidence) ---
Passage 1: In June 2022, Mira organized a neighborhood fundraiser to support the local library.
--- End of Original Passages ---

--- Relevant Facts ---
Fact 1: (fundraiser, took place in, July 2022)  # may be wrong
--- End of Relevant Facts ---"""
)

one_shot_rag_qa_input_conflict = (
    f"{one_shot_rag_qa_memories_conflict}"
    "\n\nQuestion: "
    "When did Mira run the neighborhood fundraiser?"
    "\nThought: "
)

one_shot_rag_qa_output_conflict = (
    "Thought: Passage 1 gives the date (June 2022); memory/facts disagree, so I ignore them.\n"
    "Answer: June 2022"
)


one_shot_rag_qa_memories_passages_silent = (
    """--- Episodic Memory Summary ---
[Memory 1]
Structured Event Details:
  Event 1:
    Participants: ["Aria"]
    Action: ["Aria wants to pursue counseling or working in mental health"]
--- End of Episodic Memory Summary ---

--- Original Passages (Grounded Evidence) ---
Passage 1: (No relevant passage retrieved.)
--- End of Original Passages ---"""
)

one_shot_rag_qa_input_passages_silent = (
    f"{one_shot_rag_qa_memories_passages_silent}"
    "\n\nQuestion: "
    "What fields would Aria be likely to pursue in her education?"
    "\nThought: "
)

one_shot_rag_qa_output_passages_silent = (
    "Thought: The retrieved passages do not cover the needed detail; Memory 1 is explicit and there is no conflicting evidence.\n"
    "Answer: counseling or working in mental health"
)


prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input_conflict},
    {"role": "assistant", "content": one_shot_rag_qa_output_conflict},
    {"role": "user", "content": one_shot_rag_qa_input_passages_silent},
    {"role": "assistant", "content": one_shot_rag_qa_output_passages_silent},
    {"role": "user", "content": "${prompt_user}"},
]
