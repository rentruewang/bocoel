import logging
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import fire
import structlog
from tqdm import tqdm

from bocoel import (
    BigBenchChoiceType,
    BigBenchMatchType,
    BigBenchMultipleChoice,
    BigBenchQuestionAnswer,
    ComposedCorpus,
    ConcatStorage,
    DatasetsStorage,
    HuggingfaceGenerativeLM,
    HuggingfaceLogitsLM,
)

from . import common
from .common import CorpusEvaluatorRegistry

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

LOGGER = structlog.get_logger()


def main(
    *,
    ds_path: str = "bigbench",
    ds_names: Sequence[str] | Literal["all"] = ("abstract_narrative_understanding",),
    ds_split: str = "default",
    index_name: Literal["hnswlib", "polar", "whitening"] = "hnswlib",
    sbert_model: str = "all-mpnet-base-v2",
    llm_model: str = "distilgpt2",
    batch_size: int = 16,
    device: str = "cpu",
    sobol_steps: int = 20,
    index_threads: int = 8,
    optimizer: Literal["ax", "kmeans", "kmedoids", "random", "brute"],
    optimizer_steps: int = 30,
    reduced_dim: int = 16,
    metric: str,
    task: str = "EXPLORE",
    acqf: str = "ENTROPY",
    embedders: str = "textattack/bert-base-uncased-SST-2,textattack/distilbert-base-cased-SST-2",
    corpus_cache_path: str | Path = "./cache/",
) -> None:
    # The corpus part
    if ds_names == "all":
        ds_names = _ALL_BIG_BENCH

    for dataset_id in ds_names:
        assert dataset_id in _ALL_BIG_BENCH

    LOGGER.info("Loading datasets...", ds_list=ds_names)

    dataset_list: list[DatasetsStorage] = []
    for ds_name in tqdm(ds_names):
        LOGGER.debug("Loading...", path=ds_path, name=ds_name, split=ds_split)
        dataset = DatasetsStorage(path=ds_path, name=ds_name, split=ds_split)
        dataset_list.append(dataset)

    storage = ConcatStorage.join(dataset_list)

    embedder = common.ensemble_embedder(
        embedders=embedders.split(","), batch_size=batch_size
    )

    LOGGER.info(
        "Creating corpus with storage and embedder",
        storage=storage,
        embedder=embedder,
        device=device,
    )

    corpus_cache_path = Path(corpus_cache_path)
    corpus: ComposedCorpus
    if corpus_cache_path.exists():
        LOGGER.info("Loading corpus from cache", path=corpus_cache_path)
        with open(corpus_cache_path, "rb") as f:
            corpus = pickle.load(f)
    else:
        corpus = common.composed_corpus(
            batch_size=batch_size,
            storage=storage,
            device=device,
            index_name=index_name,
            index_threads=index_threads,
            reduced=reduced_dim,
            sentence="inputs",
            embedder=embedder,
        )
        with open(corpus_cache_path, "wb") as f:
            pickle.dump(corpus, f)

    # ------------------------
    # The model part

    LOGGER.info(
        "Creating adaptor with arguments",
        metric=metric,
    )
    adaptor = bigbench_adaptor(
        llm_model=llm_model,
        metric=metric,
        device=device,
        batch_size=batch_size,
    )

    # ------------------------
    # The optimizer part.

    LOGGER.info(
        "Creating optimizer with arguments",
        corpus=corpus,
        lm=llm_model,
        adaptor=adaptor,
        sobol_steps=sobol_steps,
        device=device,
        acqf=acqf,
    )

    registry = CorpusEvaluatorRegistry()
    optim, optimizer_steps = common.optimizer_and_steps(
        optimizer=optimizer,
        optimizer_steps=optimizer_steps,
        corpus=corpus,
        adaptor=adaptor,
        sobol_steps=sobol_steps,
        device=device,
        task=task,
        acqf=acqf,
        batch_size=batch_size,
        corpus_evals=registry,
    )
    for i in tqdm(range(optimizer_steps)):
        try:
            state = optim.step()
        except StopIteration:
            break
        LOGGER.info("iteration {i}: {state}", i=i, state=state)


def bigbench_adaptor(
    llm_model: str, metric: str, device: str, batch_size: int
) -> BigBenchMultipleChoice | BigBenchQuestionAnswer:
    LOGGER.info("Creating LM with model", model=llm_model, device=device)

    if metric not in _QUESTION_ANSWER + _MULTIPLE_CHOICE:
        raise ValueError(
            f"Metric must be one of {_QUESTION_ANSWER + _MULTIPLE_CHOICE}, got {metric}."
        )

    multiple_choice = metric in _MULTIPLE_CHOICE
    if multiple_choice:
        LOGGER.info("Creating multiple choice adaptor")
        logits_lm = HuggingfaceLogitsLM(
            model_path=llm_model,
            device=device,
            batch_size=batch_size,
            choices=[str(i) for i in range(1, 100 + 1)],
        )

        return BigBenchMultipleChoice(
            lm=logits_lm, choice_type=BigBenchChoiceType.lookup(metric)
        )
    else:
        LOGGER.info("Creating question answer adaptor")
        generative_lm = HuggingfaceGenerativeLM(
            model_path=llm_model, device=device, batch_size=batch_size
        )
        LOGGER.info("Creating LM with model", model=llm_model, device=device)

        return BigBenchQuestionAnswer(
            lm=generative_lm, matching_type=BigBenchMatchType.lookup(metric)
        )


_QUESTION_ANSWER = [
    "EXACT",
    "NLTK_BLEU",
    "SACRE_BLEU",
    "ROUGE",
    "ROUGE_1",
    "ROUGE_2",
    "ROUGE_L",
]
_MULTIPLE_CHOICE = [
    "SUM_OF_SCORES",
    "LIST_OF_ANSWERS",
]

_ALL_BIG_BENCH = """
abstract_narrative_understanding anachronisms analogical_similarity analytic_entailment arithmetic ascii_word_recognition
authorship_verification auto_categorization auto_debugging bbq_lite_json bridging_anaphora_resolution_barqa causal_judgment
cause_and_effect checkmate_in_one chess_state_tracking chinese_remainder_theorem cifar10_classification code_line_description
codenames color common_morpheme conceptual_combinations conlang_translation contextual_parametric_knowledge_conflicts crash_blossom
crass_ai cryobiology_spanish cryptonite cs_algorithms dark_humor_detection date_understanding disambiguation_qa
discourse_marker_prediction disfl_qa dyck_languages elementary_math_qa emoji_movie emojis_emotion_prediction empirical_judgments
english_proverbs english_russian_proverbs entailed_polarity entailed_polarity_hindi epistemic_reasoning
evaluating_information_essentiality fact_checker fantasy_reasoning few_shot_nlg figure_of_speech_detection
formal_fallacies_syllogisms_negation gem gender_inclusive_sentences_german general_knowledge geometric_shapes goal_step_wikihow
gre_reading_comprehension hhh_alignment hindi_question_answering hindu_knowledge hinglish_toxicity human_organs_senses hyperbaton
identify_math_theorems identify_odd_metaphor implicatures implicit_relations intent_recognition international_phonetic_alphabet_nli
international_phonetic_alphabet_transliterate intersect_geometry irony_identification kanji_ascii kannada key_value_maps
known_unknowns language_games language_identification linguistic_mappings linguistics_puzzles list_functions logic_grid_puzzle
logical_args logical_deduction logical_fallacy_detection logical_sequence mathematical_induction matrixshapes metaphor_boolean
metaphor_understanding minute_mysteries_qa misconceptions misconceptions_russian mnist_ascii modified_arithmetic
moral_permissibility movie_dialog_same_or_different movie_recommendation mult_data_wrangling multiemo natural_instructions navigate
nonsense_words_grammar novel_concepts object_counting odd_one_out operators paragraph_segmentation parsinlu_qa
parsinlu_reading_comprehension penguins_in_a_table periodic_elements persian_idioms phrase_relatedness physical_intuition physics
physics_questions play_dialog_same_or_different polish_sequence_labeling presuppositions_as_nli qa_wikidata question_selection
real_or_fake_text reasoning_about_colored_objects repeat_copy_logic rephrase riddle_sense ruin_names
salient_translation_error_detection scientific_press_release semantic_parsing_in_context_sparc semantic_parsing_spider
sentence_ambiguity similarities_abstraction simp_turing_concept simple_arithmetic_json simple_arithmetic_json_multiple_choice
simple_arithmetic_json_subtasks simple_arithmetic_multiple_targets_json simple_ethical_questions simple_text_editing snarks
social_iqa social_support sports_understanding strange_stories strategyqa sufficient_information suicide_risk
swahili_english_proverbs swedish_to_german_proverbs symbol_interpretation temporal_sequences tense timedial topical_chat
tracking_shuffled_objects understanding_fables undo_permutation unit_conversion unit_interpretation unnatural_in_context_learning
vitaminc_fact_verification what_is_the_tao which_wiki_edit winowhy word_sorting word_unscrambling""".strip().split()

if __name__ == "__main__":
    fire.Fire(main)
