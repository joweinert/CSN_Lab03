LANG_DICT = {
    "en": "english",
    "ar": "arabic",
    "cs": "czech",
    "de": "german",
    "es": "spanish",
    "fi": "finnish",
    "fr": "french",
    "gl": "galician",
    "hi": "hindi",
    "id": "indonesian",
    "is": "icelandic",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sv": "swedish",
    "th": "thai",
    "tr": "turkish",
    "zh": "chinese",
}


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def fmt_simulation_results(lang, results):
    out, first = "", True
    for null_model, (
        p_val,
        avg_orig_closeness,
        avg_null_closeness,
        std_null_closeness,
    ) in results.items():
        if first:
            out += f"{LANG_DICT[lang]:<15}{null_model:<15}{p_val:<15.6f}{avg_orig_closeness:<25.6f}{avg_null_closeness:<25.6f}{std_null_closeness:<25.6f}\n"
            first = False
        else:
            out += f"{'':<15}{null_model:<15}{p_val:<15.6f}{'':<25}{avg_null_closeness:<25.6f}{std_null_closeness:<25.6f}\n"
    return out
