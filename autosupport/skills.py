"""Loads skills/*.md by name for graph nodes. Skills are never all concatenated into
one system prompt — each LLM node loads its own fixed skill plus any optional skills
`triage` added to `active_skills` (graph-design.md §3). Built at CP4."""
