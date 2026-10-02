import sys
sys.path.insert(0, r"C:\02_QUILLAN\09 - Projects\projects\oni")
from quillan_tokenizer_unified import UnifiedQuillanTokenizer
tok = UnifiedQuillanTokenizer()
for s, want in [("<|start|>", 50257), ("<|user|>", 50258),
                ("<|assistant|>", 50259), ("<|im_start|>", 50260),
                ("<|im_end|>", 50261)]:
    ids = tok.encode(s)
    ok = "SINGLE-OK" if ids == [want] else "FRAGMENT-FAIL"
    print(f"{s} -> {ids} expect [{want}] {ok}", flush=True)
