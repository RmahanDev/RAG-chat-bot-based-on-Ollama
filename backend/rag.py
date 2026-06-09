from langchain_chroma import Chroma
from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings
)

from config import *

import time
import re


# ==========================================
# VECTOR DATABASE
# ==========================================

embedding = OllamaEmbeddings(
    model=EMBED_MODEL
)

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding
)

retriever = db.as_retriever(
    search_kwargs={
        "k": 1
    }
)

# ==========================================
# LLM
# ==========================================

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)

# ==========================================
# TOKENIZER / TERM EXTRACTION
# ==========================================

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "do", "does", "did", "for", "from", "had", "has", "have", "he",
    "her", "his", "i", "if", "in", "into", "is", "it", "its", "me",
    "my", "not", "of", "on", "or", "our", "she", "so", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "to", "too", "us", "was", "we", "were", "what", "when",
    "where", "which", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "can", "could", "shall", "should", "may",
    "might", "must", "just", "also", "only", "still", "very", "more",
    "most", "some", "any", "each", "few", "many", "much", "other",
    "such", "no", "yes", "here", "there", "up", "down", "out", "over",
    "under", "again", "once", "about", "after", "before", "between",
    "during", "without", "within", "across", "through", "per"
}

GENERIC_FILLER_WORDS = {
    "answer", "answers", "answered", "answering",
    "assistant", "context", "contexts",
    "detail", "details",
    "document", "documents",
    "explicit", "explicitly",
    "explain", "explains", "explained", "explaining",
    "find", "finding",
    "information", "infos",
    "mentioned", "mention", "mentions",
    "missing",
    "part", "parts",
    "provide", "provided", "provides", "providing",
    "question", "questions",
    "relevant", "irrelevant",
    "respond", "responds", "responded", "responding",
    "response", "responses",
    "said", "says", "say", "saying",
    "state", "states", "stated", "stating",
    "support", "supported", "supporting",
    "term", "terms",
    "use", "used", "uses", "using",
    "user", "users",
    "able", "enough",
    "allow", "allows", "allowed", "allowing",
    "according", "based",
    "better", "clearly", "exactly",
    "important", "insufficient",
    "need", "needs", "needed", "needing",
    "please", "possible",
    "show", "shown", "showing",
    "tell", "telling",
    "thing", "things",
    "word", "words",
    "wording",
    "data",
    "facts",
    "fact",
    "claim", "claims",
    "concept", "concepts",
    "entity", "entities",
    "name", "names",
    "person", "people",
    "company", "companies",
    "product", "products",
    "location", "locations",
    "project", "projects",
    "mission",
    "team",
    "teams"
}

TOKEN_ALIASES = {
    "build": "founding",
    "built": "founding",
    "builder": "founding",
    "builders": "founding",
    "creator": "founding",
    "creators": "founding",
    "created": "founding",
    "creation": "founding",
    "establish": "founding",
    "established": "founding",
    "founder": "founding",
    "founders": "founding",
    "founded": "founding",
    "founding": "founding",
    "launch": "launched",
    "launches": "launched",
    "launched": "launched",
    "launching": "launched",
    "located": "location",
    "locate": "location",
    "location": "location",
    "headquarter": "headquarters",
    "headquarters": "headquarters",
    "hq": "headquarters",
    "ceo": "ceo",
    "ai": "ai",
    "ml": "ml",
    "nlp": "nlp",
    "cv": "cv",
    "ux": "ux",
    "ui": "ui",
    "qa": "qa",
    "db": "db",
    "sql": "sql",
    "api": "api",
    "rag": "rag",
    "llm": "llm",
    "go": "go"
}

SHORT_KEEP = {
    "ai", "ml", "nlp", "cv", "ux", "ui", "qa", "db",
    "sql", "api", "rag", "llm", "go"
}


def normalize_term(term):

    term = term.lower().strip()
    term = term.replace("&", " and ")
    term = re.sub(r"[^\w\s]", " ", term)
    term = re.sub(r"\s+", " ", term).strip()

    if not term:
        return ""

    return TOKEN_ALIASES.get(term, term)


def tokenize(text):

    raw_tokens = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9'\-]*",
        text.lower()
    )

    tokens = set()

    for token in raw_tokens:

        cleaned = token.strip("_'\"-")

        if not cleaned:
            continue

        cleaned = TOKEN_ALIASES.get(cleaned, cleaned)

        if cleaned in STOP_WORDS:
            continue

        if cleaned in GENERIC_FILLER_WORDS:
            continue

        if len(cleaned) < 3 and cleaned not in SHORT_KEEP and not cleaned.isdigit():
            continue

        tokens.add(cleaned)

    return tokens


def extract_number_terms(text):

    patterns = [
        r"\b\d{4}\b",
        r"\b\d+(?:\.\d+)?\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4}\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?\b",
        r"\b\d+(?:\.\d+)?\s*(?:%|percent|million|billion|thousand|m|bn|k)\b"
    ]

    found = set()

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            term = normalize_term(match.group(0))
            if term:
                found.add(term)

    return found


def extract_entity_terms(text):

    candidates = set()

    patterns = [
        r"\b(?:[A-Z][\w&.'-]*\s+){1,6}[A-Z][\w&.'-]*\b",
        r"\b[A-Z]{2,}(?:[A-Z0-9]+)?\b",
        r"\b[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]+)+\b"
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            phrase = normalize_term(match.group(0))
            if not phrase:
                continue

            parts = phrase.split()

            if len(parts) == 1:
                token = parts[0]

                if token in STOP_WORDS or token in GENERIC_FILLER_WORDS:
                    continue

                if token.isalpha() and len(token) < 3 and token not in SHORT_KEEP:
                    continue

                candidates.add(token)
                continue

            if parts[0] in STOP_WORDS or parts[0] in GENERIC_FILLER_WORDS:
                continue

            if all(
                part in STOP_WORDS or part in GENERIC_FILLER_WORDS
                for part in parts
            ):
                continue

            candidates.add(phrase)

    return candidates


def build_profile(text):

    numbers = extract_number_terms(text)
    entities = extract_entity_terms(text)

    entity_token_parts = set()
    for entity in entities:
        entity_token_parts.update(entity.split())

    keywords = tokenize(text) - entity_token_parts - numbers

    terms = entities | numbers | keywords

    weights = {}

    for term in terms:
        if term in numbers:
            weights[term] = 3.5
        elif term in entities:
            weights[term] = 2.75 if " " in term else 2.5
        else:
            weights[term] = 1.0

    return {
        "terms": terms,
        "entities": entities,
        "numbers": numbers,
        "keywords": keywords,
        "weights": weights
    }


def profile_total_weight(profile):

    return sum(profile["weights"].values())


def profile_supported_weight(profile, allowed_terms):

    return sum(
        weight
        for term, weight in profile["weights"].items()
        if term in allowed_terms
    )


# ==========================================
# SCORING
# ==========================================

def grounding_score(
    context,
    question,
    answer
):

    context_profile = build_profile(context)
    question_profile = build_profile(question)
    answer_profile = build_profile(answer)

    if not answer_profile["terms"]:
        return 0.0

    allowed_terms = context_profile["terms"] | question_profile["terms"]

    supported_weight = profile_supported_weight(
        answer_profile,
        allowed_terms
    )

    total_weight = profile_total_weight(answer_profile)

    if total_weight == 0:
        return 0.0

    precision = supported_weight / total_weight

    context_total = profile_total_weight(context_profile)
    if context_total <= 0:
        context_total = 1.0

    coverage = supported_weight / context_total

    grounding = (
        (precision * 0.85) +
        (coverage * 0.15)
    ) * 4

    return min(
        grounding,
        4.0
    )


def hallucination_risk(
    context,
    question,
    answer
):

    context_profile = build_profile(context)
    question_profile = build_profile(question)
    answer_profile = build_profile(answer)

    if not answer_profile["terms"]:
        return 100.0

    allowed_terms = context_profile["terms"] | question_profile["terms"]

    supported_weight = profile_supported_weight(
        answer_profile,
        allowed_terms
    )

    total_weight = profile_total_weight(answer_profile)

    if total_weight == 0:
        return 100.0

    precision = supported_weight / total_weight

    risk = (1 - precision) * 100

    return round(
        max(0.0, risk),
        4
    )


def relevance_score(
    question,
    context,
    answer
):

    question_profile = build_profile(question)
    context_profile = build_profile(context)
    answer_profile = build_profile(answer)

    if not answer_profile["terms"]:
        return 0.0

    answer_total = profile_total_weight(answer_profile)
    if answer_total <= 0:
        return 0.0

    question_total = profile_total_weight(question_profile)

    if question_total > 0:
        question_supported = profile_supported_weight(
            answer_profile,
            question_profile["terms"]
        )
        question_precision = question_supported / question_total
    else:
        question_precision = 0.0

    context_supported = profile_supported_weight(
        answer_profile,
        context_profile["terms"]
    )
    context_precision = context_supported / answer_total

    if question_total > 0:
        relevance = (
            (question_precision * 0.60) +
            (context_precision * 0.40)
        ) * 3
    else:
        relevance = context_precision * 3

    return min(
        relevance,
        3.0
    )


def speed_score(
    elapsed
):

    if elapsed < 2:
        return 2

    elif elapsed < 5:
        return 2

    elif elapsed < 10:
        return 1.8

    elif elapsed < 20:
        return 1.8

    elif elapsed < 30:
        return 1.8

    elif elapsed < 40:
        return 1.5

    elif elapsed < 60:
        return 1

    elif elapsed < 90:
        return 0.8

    elif elapsed < 100:
        return 0.5

    return 0.25


def conciseness_score(
    answer
):

    length = len(answer)

    if 100 <= length <= 800:
        return 1

    elif 50 <= length <= 1200:
        return 0.75

    return 0.5


# ==========================================
# MAIN FUNCTION
# ==========================================

def ask(question):

    total_start = time.time()

    print("\n==============================")
    print("QUESTION:", question)
    print("==============================")

    # --------------------------------------
    # RETRIEVAL
    # --------------------------------------

    retrieve_start = time.time()

    docs = retriever.invoke(question)

    retrieve_time = round(
        time.time() - retrieve_start,
        4
    )

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )

    # محدود کردن Context

    context = context[:1500]

    context_length = len(context)

    chunk_count = len(docs)

    print(
        f"Retrieved Chunks: {chunk_count}"
    )

    print(
        f"Context Length: {context_length}"
    )

    print(
        "\nContext Preview:\n"
    )

    print(
        context[:250]
    )

    # --------------------------------------
    # PROMPT
    # --------------------------------------

    prompt = f"""
You are a strict RAG assistant.

IMPORTANT RULES:

- When user tryes to intract with saying hi orhow are you just intract back to them with saying hi how are you today and how can i assist you today ? and still if user intract with any other intraction word try to respond friendly.

- Use users qustion context words in your respond.

- Answr faster

- dont answer only with one word.

- Adjust the response length based on the question. For simple questions, provide concise answers. For complex questions, provide detailed explanations while staying strictly grounded in the provided context.

- Use more of the documents words and data.

- Use ONLY the provided context.

- Use terminology and wording similar to the user's question whenever possible.

- You may reuse relevant words and phrases from the user's question even if those exact words do not appear in the context.

- If the user consistently refers to concepts using alternative but related terminology, you may mirror that terminology in your response.

- However, all facts, names, dates, events, numbers and claims must still come exclusively from the provided context.

- Adapting the user's wording must never be used as a reason to invent information that is not supported by the context.

- Pay attention to names, organizations, products and locations mentioned in the user's question.

- If a specific name, person, company, product or concept mentioned by the user does not appear in the provided context, explicitly state that it was not found in the documents.

- When possible, mention which term or entity could not be found.

- Do not invent information about missing entities.

- If only part of the question can be answered from the context, answer the supported part and clearly indicate which parts are not present in the documents.

- Never invent names.

- Never invent dates.

- Never invent companies.

- Never invent facts.

- Every statement must be supported by the context.

- If information is missing, ignore it.

- Do not expand beyond the context.

- If the context is insufficient, say:

I could not find enough information in the documents.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""

    # --------------------------------------
    # GENERATION
    # --------------------------------------

    print("\nGenerating answer...")

    llm_start = time.time()

    response = llm.invoke(prompt)

    llm_time = round(
        time.time() - llm_start,
        4
    )

    total_time = round(
        time.time() - total_start,
        4
    )

    answer = response.content.strip()

    # --------------------------------------
    # METRICS
    # --------------------------------------

    grounding = grounding_score(
        context,
        question,
        answer
    )

    relevance = relevance_score(
        question,
        context,
        answer
    )

    speed = speed_score(
        total_time
    )

    concise = conciseness_score(
        answer
    )

    hallucination = hallucination_risk(
        context,
        question,
        answer
    )

    final_score = round(
        grounding +
        relevance +
        speed +
        concise,
        10
    )

    # --------------------------------------
    # LOGGING
    # --------------------------------------

    print("\n========== METRICS ==========")

    print(
        f"Retriever: {retrieve_time}s"
    )

    print(
        f"LLM: {llm_time}s"
    )

    print(
        f"Total: {total_time}s"
    )

    print()

    print(
        f"Grounding: {grounding:.10f}/4"
    )

    print(
        f"Relevance: {relevance:.10f}/3"
    )

    print(
        f"Speed: {speed:.10f}/2"
    )

    print(
        f"Conciseness: {concise:.10f}/1"
    )

    print()

    print(
        f"Hallucination Risk: {hallucination:.4f}%"
    )

    print(
        f"Final Score: {final_score:.10f}/10"
    )

    print("=============================")

    # --------------------------------------
    # RETURN
    # --------------------------------------

    return {

        "answer": answer,

        "time": total_time,

        "score": final_score,

        "context_length": context_length,

        "chunk_count": chunk_count,

        "retriever_time": retrieve_time,

        "llm_time": llm_time,

        "hallucination_risk": round(
            hallucination,
            4
        ),

        "metrics": {

            "grounding":
                round(
                    grounding,
                    10
                ),

            "relevance":
                round(
                    relevance,
                    10
                ),

            "speed":
                round(
                    speed,
                    10
                ),

            "conciseness":
                round(
                    concise,
                    10
                )
        }
    }