def select_lowest(records):
    selected = None
    selected_score = None
    for record in records:
        score = record["score"]
        if isinstance(score, bool) or not isinstance(score, int):
            raise TypeError("score must be an integer")
        if selected is None or score <= selected_score:
            selected, selected_score = record, score
    if selected is None:
        raise ValueError("records must not be empty")
    return selected
