# Offline public contracts

All public functions are exported by notehub. There is no file, network or database IO. Checked-in tests are small API smoke checks, not proof of every edge case.

- parse_tags(value): text only, TypeError otherwise. Split commas, strip whitespace, skip blanks, normalize with casefold, remove duplicate normalized tags and preserve first encounter order. Return a fresh list; empty text returns [].
- page(items, offset=0, limit=20): nonnegative integers excluding bool. TypeError for nonintegers/bools; ValueError for negatives. Return a fresh list of at most limit elements starting at offset. Do not mutate input. Tuple input is supported; zero limit and out-of-range offsets return [].
- add_label(note, label): text only, TypeError otherwise; trim and reject blank labels with ValueError. Both returned mapping and labels list are fresh, including duplicate-label paths. Never mutate input labels. Missing labels means []; present labels must be a list. Preserve insertion order and identity of unrelated values.
- escape_text(text): text only, TypeError otherwise. Escape original backslashes as two backslashes, original pipes as backslash-pipe, and original LF as backslash-n, exactly once. render_line emits id, literal pipe delimiter, escaped title. Missing fields retain KeyError.
- next_identifier(notes): one pass over iterable mappings with positive integer id, excluding bool. Highest id plus one, or 1 if empty. Gaps and duplicates are allowed. Missing id retains KeyError; invalid type raises TypeError; nonpositive id raises ValueError.
- choose_handler(action, handlers): nonempty text action; TypeError for other types, ValueError for empty text. Callable default is required even for known actions. Present handlers must be callable. Registered callable is returned by identity even if false in boolean context; absent actions use default. Missing default retains KeyError.
- create_note(title, tags, notes): notes is a list. Trim/validate title, parse tags, select id, then return (new_note_dict, new_notes_list), without mutating the caller list. It does not persist anything.

These are desired contracts. Starting code intentionally contains small edge-case defects. Inspection tasks ask about current implementation.
