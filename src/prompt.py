def build_messages(text, categories):
    system_message = (
        "You are a banking intent classification assistant. "
        "Classify the customer's message into exactly one of the "
        "available intents. Return only the exact intent name, "
        "with no explanation."
    )

    intent_list = "\n".join(f"- {intent}" for intent in categories)

    user_message = (
        f"Available intents:\n{intent_list}\n\n"
        f"Customer message:\n{text}\n\n"
        "Return exactly one intent name from the list."
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def build_training_messages(text, categories, answer):
    messages = build_messages(text, categories)
    return messages + [
        {"role": "assistant", "content": answer}
    ]
