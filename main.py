def rebuild_segments_smart(result, max_chars_per_line=38, max_gap_seconds=0.6):
    """
    Riorganizza i sottotitoli a livello di singola parola per:
    1. Limitare RIGOROSAMENTE il sottotitolo a MAX 2 RIGHE totali (circa 76-80 caratteri max per blocco).
    2. Spezzare IMMEDIATAMENTE quando cambia lo speaker (max 1 riga per speaker se ravvicinati).
    3. Tagliare prima dei limiti di riga basandosi su punteggiatura e pause.
    """
    all_words = []
    
    # Estrae tutte le parole allineate con i relativi timestamp e speaker
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            if "start" in word and "end" in word and "word" in word:
                all_words.append({
                    "word": word["word"].strip(),
                    "start": word["start"],
                    "end": word["end"],
                    "speaker": word.get("speaker", segment.get("speaker", "UNKNOWN"))
                })

    if not all_words:
        return result.get("segments", [])

    new_segments = []
    current_words = []
    current_speaker = all_words[0]["speaker"]
    current_start = all_words[0]["start"]
    last_end = all_words[0]["end"]

    # Limite massimo di caratteri per un blocco da 2 righe
    MAX_BLOCK_CHARS = max_chars_per_line * 2 

    def commit_segment(words, start, end, speaker):
        if not words:
            return
        text = " ".join([w["word"] for w in words])
        
        # Se il testo supera la lunghezza di 1 riga, inseriamo un andata a capo '\n' bilanciata
        if len(text) > max_chars_per_line:
            mid_point = len(words) // 2
            line1 = " ".join([w["word"] for w in words[:mid_point]])
            line2 = " ".join([w["word"] for w in words[mid_point:]])
            formatted_text = f"{line1}\n{line2}"
        else:
            formatted_text = text

        new_segments.append({
            "start": start,
            "end": end,
            "text": formatted_text,
            "speaker": speaker,
            "words": words
        })

    for i, w in enumerate(all_words):
        word_text = w["word"]
        speaker_changed = (w["speaker"] != current_speaker)
        time_gap = (w["start"] - last_end) > max_gap_seconds
        
        # Calcola la lunghezza potenziale del testo nel blocco corrente
        current_text_len = sum(len(x["word"]) + 1 for x in current_words) + len(word_text)
        
        # Controlla punteggiatura sulla parola precedente
        prev_word_has_punctuation = False
        if current_words:
            last_word_str = current_words[-1]["word"]
            prev_word_has_punctuation = any(last_word_str.endswith(p) for p in [".", "?", "!", ",", ";"])

        # CONDITTIONAL SPLIT (Forza la chiusura del sottotitolo):
        # 1. Cambio dello speaker (garantisce 1 riga/blocco per speaker ed evita sovrapposizioni)
        # 2. Pausa di silenzio > 0.6s
        # 3. Superamento del limite fisico di 2 RIGHE (MAX_BLOCK_CHARS)
        # 4. Superamento della 1ª riga SE siamo su una punteggiatura naturale
        must_split = (
            speaker_changed 
            or time_gap 
            or (current_text_len >= MAX_BLOCK_CHARS)
            or (current_text_len > max_chars_per_line and prev_word_has_punctuation)
        )

        if must_split and current_words:
            commit_segment(current_words, current_start, last_end, current_speaker)
            current_words = []
            current_speaker = w["speaker"]
            current_start = w["start"]

        current_words.append(w)
        last_end = w["end"]

        # Se la parola attuale ha un punto forte (. ? !), chiude il blocco per non trascinare il testo oltre
        if any(word_text.endswith(p) for p in [".", "?", "!"]):
            commit_segment(current_words, current_start, last_end, current_speaker)
            current_words = []
            if i + 1 < len(all_words):
                next_w = all_words[i + 1]
                current_speaker = next_w["speaker"]
                current_start = next_w["start"]

    # Salva le ultime parole rimanenti
    if current_words:
        commit_segment(current_words, current_start, last_end, current_speaker)

    return new_segments
