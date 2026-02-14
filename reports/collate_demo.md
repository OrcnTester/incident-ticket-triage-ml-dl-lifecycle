# Collate / Batch Consistency Demo

- label_col: `priority`  text_col: `text`
- tokenizer mode: `subword` (lowercase=True)
- subword n-grams: `3..5`
- vocab_size: `1771`  OOV rate: `0.0054`

## Collate config
- pad_to: `fixed`
- max_len: `128`
- truncation: `True` (side=head)
- batch_size: `32`

## Example batch shapes
- batch 0 idx[0:32] input_ids=[32, 128] mask=[32, 128] labels=[32] (seq_len=128, lengths=52..128)
- batch 1 idx[32:64] input_ids=[32, 128] mask=[32, 128] labels=[32] (seq_len=128, lengths=43..128)

## Notes
- `attention_mask=1` marks real tokens; `0` marks padding.
- Use **the same vocab/tokenizer artifacts across splits** to avoid train/inference mismatch.