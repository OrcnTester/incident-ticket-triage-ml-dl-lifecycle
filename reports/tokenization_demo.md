# Tokenization Demo Report

- mode: `subword`  lowercase: `True`
- subword n-grams: `3..5`
- vocab_size: `1771`  (min_freq=2, max_size=20000)
- OOV rate: `0.0054`  (unk=1121 / tokens=207007)

## Sequence length stats (tokens per text)
- min: 25
- median: 97.0
- p95: 187.0
- max: 265

## Samples (text → tokens → ids)
### Sample 0

**Text**
`[auth] p95 high (E500). alert triggered from montioring.`

**Tokens (first 50)**
```
['<au', 'aut', 'uth', 'th>', '<aut', 'auth', 'uth>', '<auth', 'auth>', '<p9', 'p95', '95>', '<p95', 'p95>', '<p95>', '<hi', 'hig', 'igh', 'gh>', '<hig', 'high', 'igh>', '<high', 'high>', '<e5', 'e50', '500', '00>', '<e50', 'e500', '500>', '<e500', 'e500>', '<.>', '<al', 'ale', 'ler', 'ert', 'rt>', '<ale', 'aler', 'lert', 'ert>', '<aler', 'alert', 'lert>', '<tr', 'tri', 'rig', 'igg']
```
**IDs (first 50)**
```
[340, 175, 225, 343, 344, 218, 349, 345, 348, 1199, 1202, 1198, 1200, 1203, 1201, 1195, 1192, 74, 1190, 1196, 1193, 1191, 1197, 1194, 8, 10, 325, 324, 9, 435, 326, 434, 436, 2, 148, 72, 208, 203, 197, 173, 200, 209, 219, 198, 201, 221, 217, 224, 23, 183]
```
### Sample 1

**Text**
`[payments] p9 5high (E403). alert triggered from monitoring.`

**Tokens (first 50)**
```
['<pa', 'pay', 'aym', 'yme', 'men', 'ent', 'nts', 'ts>', '<pay', 'paym', 'ayme', 'ymen', 'ment', 'ents', 'nts>', '<paym', 'payme', 'aymen', 'yment', 'ments', 'ents>', '<p9', 'p9>', '<p9>', '<5h', '5hi', 'hig', 'igh', 'gh>', '<5hi', '5hig', 'high', 'igh>', '<5hig', '5high', 'high>', '<e4', 'e40', '403', '03>', '<e40', 'e403', '403>', '<e403', 'e403>', '<.>', '<al', 'ale', 'ler', 'ert']
```
**IDs (first 50)**
```
[100, 108, 112, 14, 13, 3, 313, 233, 106, 113, 114, 16, 15, 310, 314, 111, 116, 115, 17, 320, 311, 1199, 1, 1, 1, 1, 1192, 74, 1190, 1, 1, 1193, 1191, 1, 1, 1194, 11, 27, 423, 76, 26, 432, 424, 429, 433, 2, 148, 72, 208, 203]
```
### Sample 2

**Text**
`[inventory] login fails (OOM). seen by customers`

**Tokens (first 50)**
```
['<in', 'inv', 'nve', 'ven', 'ent', 'nto', 'tor', 'ory', 'ry>', '<inv', 'inve', 'nven', 'vent', 'ento', 'ntor', 'tory', 'ory>', '<inve', 'inven', 'nvent', 'vento', 'entor', 'ntory', 'tory>', '<lo', 'log', 'ogi', 'gin', 'in>', '<log', 'logi', 'ogin', 'gin>', '<logi', 'login', 'ogin>', '<fa', 'fai', 'ail', 'ils', 'ls>', '<fai', 'fail', 'ails', 'ils>', '<fail', 'fails', 'ails>', '<oo', 'oom']
```
**IDs (first 50)**
```
[19, 62, 122, 317, 3, 295, 24, 301, 304, 65, 121, 315, 318, 299, 308, 322, 302, 120, 312, 316, 319, 309, 321, 323, 1018, 1045, 1050, 1041, 989, 1019, 1046, 1051, 1042, 1020, 1047, 1052, 297, 294, 117, 1094, 495, 298, 300, 1080, 1095, 307, 1092, 1081, 383, 388]
```