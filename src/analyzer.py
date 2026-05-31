"""
AI思维克隆器 - 风格分析框架
从对话记录中提取语言特征和思维模式
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jieba
import jieba.analyse


@dataclass
class StyleProfile:
    """风格画像数据结构"""
    # 语言特征
    avg_sentence_length: float = 0.0
    sentence_length_std: float = 0.0
    exclamation_freq: float = 0.0
    question_freq: float = 0.0

    # 词汇特征
    top_keywords: list = field(default_factory=list)
    high_freq_words: list = field(default_factory=list)
    unique_vocab_size: int = 0

    # 句式特征
    avg_clause_count: float = 0.0
    parallelism_freq: float = 0.0
    metaphor_freq: float = 0.0

    # 认知模式
    reasoning_style: str = ""  # deductive/inductive/abductive
    argument_depth: float = 0.0
    example_usage_freq: float = 0.0

    # 情感特征
    emotion_words: list = field(default_factory=list)
    sentiment_score: float = 0.0

    # 元数据
    sample_count: int = 0
    total_characters: int = 0


class StyleAnalyzer:
    """风格分析器"""

    def __init__(self, data_path: str, config: Optional[dict] = None):
        """
        初始化分析器

        Args:
            data_path: 对话数据路径
            config: 配置参数
        """
        self.data_path = Path(data_path)
        self.config = config or {
            'min_samples': 50,
            'top_keywords_count': 30,
            'stopwords': {'的', '了', '是', '在', '我', '有', '和', '就'}
        }
        self.dialogues: list = []
        self.profile = StyleProfile()

    def load_data(self) -> None:
        """加载对话数据"""
        if self.data_path.is_file():
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.dialogues = json.load(f)
        elif self.data_path.is_dir():
            self.dialogues = []
            for file_path in self.data_path.glob('*.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.dialogues.extend(json.load(f))

        if len(self.dialogues) < self.config['min_samples']:
            raise ValueError(f"样本数量不足，至少需要 {self.config['min_samples']} 条对话")

        self.profile.sample_count = len(self.dialogues)

    def analyze_sentence_structure(self) -> None:
        """分析句子结构"""
        sentences = []
        total_length = 0

        for dialogue in self.dialogues:
            content = dialogue.get('content', '')
            # 分句
            parts = re.split(r'[。！？；\n]', content)
            sentences.extend([s.strip() for s in parts if s.strip()])
            total_length += len(content)

        self.profile.total_characters = total_length

        if not sentences:
            return

        # 计算句长统计
        lengths = [len(s) for s in sentences]
        self.profile.avg_sentence_length = sum(lengths) / len(lengths)

        # 计算标准差
        mean = self.profile.avg_sentence_length
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        self.profile.sentence_length_std = variance ** 0.5

    def analyze_punctuation(self) -> None:
        """分析标点符号使用"""
        total_chars = self.profile.total_characters
        if total_chars == 0:
            return

        exclamation_count = 0
        question_count = 0

        for dialogue in self.dialogues:
            content = dialogue.get('content', '')
            exclamation_count += content.count('！') + content.count('!')
            question_count += content.count('？') + content.count('?')

        self.profile.exclamation_freq = exclamation_count / total_chars * 100
        self.profile.question_freq = question_count / total_chars * 100

    def analyze_vocabulary(self) -> None:
        """分析词汇使用"""
        all_text = ' '.join(d.get('content', '') for d in self.dialogues)

        # 分词
        words = jieba.lcut(all_text)
        words = [w for w in words if len(w) > 1 and w not in self.config['stopwords']]

        # 词频统计
        word_freq = Counter(words)
        self.profile.unique_vocab_size = len(word_freq)
        self.profile.high_freq_words = word_freq.most_common(20)

        # 关键词提取
        keywords = jieba.analyse.extract_tags(all_text, topK=self.config['top_keywords_count'])
        self.profile.top_keywords = keywords

    def analyze_rhetoric(self) -> None:
        """分析修辞手法"""
        parallelism_patterns = [
            r'(.{2,})[，,]\1',  # 排比模式
        ]

        metaphor_keywords = ['像', '如同', '好比', '仿佛', '似']

        parallelism_count = 0
        metaphor_count = 0
        total_sentences = 0

        for dialogue in self.dialogues:
            content = dialogue.get('content', '')
            sentences = re.split(r'[。！？\n]', content)
            total_sentences += len([s for s in sentences if s.strip()])

            for pattern in parallelism_patterns:
                parallelism_count += len(re.findall(pattern, content))

            for keyword in metaphor_keywords:
                metaphor_count += content.count(keyword)

        if total_sentences > 0:
            self.profile.parallelism_freq = parallelism_count / total_sentences
            self.profile.metaphor_freq = metaphor_count / total_sentences

    def analyze_cognitive_pattern(self) -> None:
        """分析认知模式"""
        # 因果词统计
        causal_words = ['因为', '所以', '因此', '导致', '由于', '使得']
        inductive_words = ['例如', '比如', '举例', '案例', '例子']
        deductive_words = ['因此', '所以', '推出', '必然', '一定']

        causal_count = 0
        inductive_count = 0
        deductive_count = 0

        for dialogue in self.dialogues:
            content = dialogue.get('content', '')

            for word in causal_words:
                causal_count += content.count(word)
            for word in inductive_words:
                inductive_count += content.count(word)
            for word in deductive_words:
                deductive_count += content.count(word)

        # 判断推理风格
        if inductive_count > deductive_count:
            self.profile.reasoning_style = 'inductive'
        elif deductive_count > inductive_count:
            self.profile.reasoning_style = 'deductive'
        else:
            self.profile.reasoning_style = 'abductive'

        self.profile.example_usage_freq = inductive_count / max(self.profile.sample_count, 1)

    def analyze_emotion(self) -> None:
        """分析情感特征"""
        positive_words = ['喜欢', '开心', '高兴', '满意', '期待', '希望', '爱', '好', '棒']
        negative_words = ['讨厌', '难过', '伤心', '失望', '担心', '害怕', '恨', '差', '糟']

        positive_count = 0
        negative_count = 0

        for dialogue in self.dialogues:
            content = dialogue.get('content', '')

            for word in positive_words:
                positive_count += content.count(word)
            for word in negative_words:
                negative_count += content.count(word)

        total = positive_count + negative_count
        if total > 0:
            self.profile.sentiment_score = (positive_count - negative_count) / total

        self.profile.emotion_words = [
            word for word in positive_words + negative_words
            if any(word in d.get('content', '') for d in self.dialogues)
        ]

    def analyze(self) -> StyleProfile:
        """执行完整分析"""
        self.load_data()
        self.analyze_sentence_structure()
        self.analyze_punctuation()
        self.analyze_vocabulary()
        self.analyze_rhetoric()
        self.analyze_cognitive_pattern()
        self.analyze_emotion()

        return self.profile

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            '语言特征': {
                '平均句长': round(self.profile.avg_sentence_length, 2),
                '句长标准差': round(self.profile.sentence_length_std, 2),
                '感叹号频率': f"{self.profile.exclamation_freq:.2f}%",
                '问号频率': f"{self.profile.question_freq:.2f}%"
            },
            '词汇特征': {
                '核心关键词': self.profile.top_keywords[:10],
                '高频词汇': self.profile.high_freq_words[:10],
                '词汇丰富度': self.profile.unique_vocab_size
            },
            '句式特征': {
                '排比使用频率': f"{self.profile.parallelism_freq:.2%}",
                '比喻使用频率': f"{self.profile.metaphor_freq:.2%}"
            },
            '认知模式': {
                '推理风格': self.profile.reasoning_style,
                '举例频率': f"{self.profile.example_usage_freq:.2f}"
            },
            '情感特征': {
                '情感倾向分数': round(self.profile.sentiment_score, 3),
                '常用情感词': self.profile.emotion_words
            },
            '统计信息': {
                '样本数量': self.profile.sample_count,
                '总字符数': self.profile.total_characters
            }
        }

    def save_profile(self, output_path: str) -> None:
        """保存分析结果"""
        result = self.to_dict()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"风格画像已保存至: {output_path}")


if __name__ == '__main__':
    # 示例用法
    analyzer = StyleAnalyzer('data/dialogues.json')
    profile = analyzer.analyze()
    analyzer.save_profile('output/style_profile.json')
    print(json.dumps(analyzer.to_dict(), ensure_ascii=False, indent=2))