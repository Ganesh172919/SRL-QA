from full_analysis_pipeline import exact_match, token_prf, bleu
print({"exact_match": exact_match("a","a"), "token_prf": token_prf("to office","office"), "bleu": bleu("to office","office")})
