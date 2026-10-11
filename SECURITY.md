# Security and data handling

CorePilot is a local single-user research tool. Bind the interface to loopback. The model endpoint must also be loopback. It does not implement public hosting, account authentication or multi-tenant isolation.

Data ingestion only opens paths explicitly chosen by the user. Outputs are local, ignored by Git, and created with restrictive permissions on supported systems. Redaction is heuristic; inspect data before sharing it. Hidden reasoning is not imported. CorePilot does not upload histories, train data or weights.

Task edits occur in a separate snapshot or Git worktree. Applying a reviewed diff is an explicit action with conflict checks. Commands use macOS sandbox-exec with network disabled and restricted writes. On systems without this supported sandbox, command execution is disabled; file tools and history preparation remain available. A worktree is not itself a security sandbox.

Models, repository files and historical messages are untrusted. Do not give a model access to sensitive repositories or credentials solely because a test passes. No secret filter can guarantee that all sensitive data is found.

Report vulnerabilities through GitHub private vulnerability reporting when available. Otherwise open a minimal issue asking for a private contact without including exploit payloads, secrets or personal data.
