# Spec removal: {{id}}

`{{id}}` — {{title}} — was implemented earlier from the specification, and that section has since been DELETED from the spec. The spec is the master: the code follows it, so the implementation and its tests go too.

{{previous}}

## What the spec used to say

```
{{old_text}}
```

## Do

1. Find what implements it. The test that named it was `{{covered_by}}`; the implementation is whatever that test exercises. `search_text` for the id `{{id}}`, for its test name, and for the names of the functions or types the test calls.
2. Delete that implementation and its tests, including fixtures and sample data that exist only for it.
3. Keep everything else working. Code the removed item shares with items that are still in the spec STAYS: a helper two other items call is not part of this removal. When in doubt about a shared piece, leave it and say so.
4. Remove what is now dead because of the deletion: an import that no longer resolves, a menu entry pointing at a screen that is gone, a settings field nothing reads.
5. Run `{{test_cmd}}` from `{{out_dir}}/`. The whole suite must pass, with no test mentioning `{{id}}` left in it.
6. Call `respond` with what you deleted and what you deliberately kept.

## Do not

- Do not delete anything the spec still describes. If removing the item would break an item that is still specified, stop and say so in `respond` instead of cutting deeper.
- Do not rewrite unrelated code while you are in there. This round removes one item and nothing else.
