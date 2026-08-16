import pandas as pd
import json

def generate_g2p_lexicon(csv_path):
    # Read CSV with pipeline separator
    df = pd.read_csv(csv_path, sep='|', header=0)
    
    # Clean column headers
    df.columns = [c.strip() for c in df.columns]

    g2p_rules = {
        "pali_terms": [],
        "sinhala_overrides": [],
        "english_lexicon": {}
    }

    for _, row in df.iterrows():
        lang = str(row['lang']).strip()
        base = str(row['baseTerm']).strip()
        has_plural = str(row['hasPlural']).strip() if pd.notna(row['hasPlural']) else ""
        has_poss = str(row['hasPossessive']).strip() if pd.notna(row['hasPossessive']) else ""

        # Construct variants (e.g. jeta, jeta's)
        variants = [base]
        if has_plural:
            variants.append(base + has_plural)
        if has_poss:
            variants.append(base + has_poss)

        for term in variants:
            if not term:
                continue
            if lang == "PI":
                g2p_rules["pali_terms"].append(term)
            elif lang == "2P":
                g2p_rules["sinhala_overrides"].append(term)

    # Sort terms by length descending so regex matcher matches "Sāvatthī's" before "Sāvatthī"
    g2p_rules["pali_terms"] = sorted(list(set(g2p_rules["pali_terms"])), key=len, reverse=True)
    g2p_rules["sinhala_overrides"] = sorted(list(set(g2p_rules["sinhala_overrides"])), key=len, reverse=True)

    with open("g2p_lexicon.json", "w", encoding="utf-8") as f:
        json.dump(g2p_rules, f, ensure_ascii=False, indent=2)

    print(f"G2P Lexicon generated: {len(g2p_rules['pali_terms'])} Pali terms, {len(g2p_rules['sinhala_overrides'])} 2P terms.")

if __name__ == "__main__":
    generate_g2p_lexicon("sutta-words-freq-list.csv")