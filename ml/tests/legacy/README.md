# Legacy tests

These tests target superseded experiment APIs or require local model/data
artifacts that are not part of a fresh checkout. They are retained as history
and are not collected by the default pytest configuration.

Run a legacy test explicitly only after confirming that its referenced model,
dataset, and source API still exist.
