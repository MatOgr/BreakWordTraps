from readability.text import Analyzer
from readability.text.analyzer import AnalyzerStatistics
from readability.scorers.flesch import Flesch
from readability.scorers.gunning_fog import GunningFog

from break_word_traps.schemas import Readability


def calculate_scores(text: str) -> Readability:
    analyzer = Analyzer()
    analyzer._dale_chall_set = analyzer._load_dale_chall()
    analyzer._spache_set = analyzer._load_spache()
    stats = analyzer._statistics(text)
    analyzer.sentences = stats["sentences"]
    og_num_words = stats["num_words"]
    # Mock number of words when creating scorers
    stats["num_words"] = stats["num_words"] if stats["num_words"] >= 100 else 100
    analyzer_statistics = AnalyzerStatistics(stats)
    flesch = Flesch(analyzer_statistics)
    gunning_fog = GunningFog(analyzer_statistics)
    # Restore original value
    stats["num_words"] = og_num_words

    flesch_score = flesch.score()
    gunning_fog_score = gunning_fog.score()

    return Readability(
        fleschScore=flesch_score.score,
        fleschGrade=flesch_score.ease,
        gunningFogScore=gunning_fog_score.score,
        gunningFogGrade=gunning_fog_score.grade_level,
    )
