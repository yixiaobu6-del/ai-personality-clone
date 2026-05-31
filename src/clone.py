"""
AI思维克隆器 - 克隆生成框架
基于风格画像生成可部署的AI分身
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .analyzer import StyleProfile


@dataclass
class CloneConfig:
    """克隆配置"""
    model_type: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    include_examples: bool = True
    example_count: int = 5


@dataclass
class PromptTemplate:
    """提示词模板"""
    system_prompt: str = ""
    user_prompt_template: str = ""
    examples: list = field(default_factory=list)
    style_instructions: list = field(default_factory=list)


class CloneGenerator:
    """克隆生成器"""

    def __init__(self, profile: StyleProfile, config: Optional[CloneConfig] = None):
        """
        初始化克隆生成器

        Args:
            profile: 风格画像
            config: 生成配置
        """
        self.profile = profile
        self.config = config or CloneConfig()
        self.template = PromptTemplate()

    def generate_system_prompt(self) -> str:
        """生成系统提示词"""
        style_hints = []

        # 语言风格指导
        if self.profile.avg_sentence_length > 0:
            avg_len = self.profile.avg_sentence_length
            if avg_len < 15:
                style_hints.append("使用简短有力的句子，保持表达的精炼性")
            elif avg_len > 30:
                style_hints.append("可以使用较长的句子进行深入阐述，注意逻辑的连贯性")
            else:
                style_hints.append("句子长度适中，保持表达的流畅与清晰")

        # 标点使用指导
        if self.profile.exclamation_freq > 1:
            style_hints.append("适度使用感叹号表达情感，增强语言的感染力")
        if self.profile.question_freq > 0.5:
            style_hints.append("善于使用设问和反问，引导思考")

        # 推理风格指导
        style_map = {
            'inductive': "倾向于通过具体案例和例子来说明观点，注重从实践中总结经验",
            'deductive': "善于从原则和理论出发，进行逻辑推导，注重论证的严密性",
            'abductive': "善于提出假设和猜想，从现象出发寻找最可能的解释"
        }
        style_hints.append(style_map.get(self.profile.reasoning_style, "保持思维的开放性"))

        # 情感表达指导
        if self.profile.sentiment_score > 0.3:
            style_hints.append("保持积极乐观的表达态度，传递正能量")
        elif self.profile.sentiment_score < -0.3:
            style_hints.append("可以适度表达忧虑，但注意保持理性客观")

        # 关键词融入
        if self.profile.top_keywords:
            keywords_str = '、'.join(self.profile.top_keywords[:8])
            style_hints.append(f"对话中自然融入这些主题词汇：{keywords_str}")

        system_prompt = f"""你是一个基于真实对话数据训练的AI助手，需要模仿特定用户的表达方式和思维模式。

## 核心身份
你是一个经过个性化训练的AI分身，代表着用户的思维风格和表达习惯。

## 语言风格要求
{chr(10).join(f"- {hint}" for hint in style_hints)}

## 交互原则
1. 保持与用户一致的思维视角
2. 用相似的方式分析问题和提出见解
3. 在专业领域展现相应的知识深度
4. 保持对话的自然流畅，避免机械式回应

## 注意事项
- 不要刻意模仿到失真的程度
- 保持回答的实用性和价值
- 适时展示个性，但不过分强调"""

        return system_prompt

    def generate_style_instructions(self) -> list:
        """生成风格指令列表"""
        instructions = []

        # 句式风格
        instructions.append({
            'category': '句式结构',
            'instruction': f"平均句长约{round(self.profile.avg_sentence_length)}字",
            'example': "保持这个长度节奏进行表达"
        })

        # 词汇使用
        if self.profile.high_freq_words:
            freq_words = [w[0] for w in self.profile.high_freq_words[:5]]
            instructions.append({
                'category': '高频词汇',
                'instruction': f"偏好使用词汇：{'、'.join(freq_words)}",
                'example': "在适当语境中自然使用这些词汇"
            })

        # 修辞偏好
        if self.profile.metaphor_freq > 0.1:
            instructions.append({
                'category': '修辞手法',
                'instruction': '善于使用比喻和类比',
                'example': "用形象化的比喻帮助理解复杂概念"
            })

        # 论证方式
        if self.profile.example_usage_freq > 0.5:
            instructions.append({
                'category': '论证方式',
                'instruction': '习惯用实例支撑观点',
                'example': "解释时提供具体案例"
            })

        return instructions

    def select_examples(self, dialogues: list) -> list:
        """选择代表性对话作为示例"""
        if not self.config.include_examples:
            return []

        selected = []
        for dialogue in dialogues[:self.config.example_count]:
            if 'question' in dialogue and 'answer' in dialogue:
                selected.append({
                    'user': dialogue['question'],
                    'assistant': dialogue['answer']
                })

        return selected

    def generate_user_prompt_template(self) -> str:
        """生成用户提示词模板"""
        template = """请基于以下风格特征回答问题：

{style_context}

用户问题：{question}

请用上述风格特征回答问题，保持语言的自然流畅。"""

        return template

    def build_prompt_template(self, dialogues: Optional[list] = None) -> PromptTemplate:
        """构建完整的提示词模板"""
        self.template.system_prompt = self.generate_system_prompt()
        self.template.style_instructions = self.generate_style_instructions()
        self.template.user_prompt_template = self.generate_user_prompt_template()

        if dialogues:
            self.template.examples = self.select_examples(dialogues)

        return self.template

    def generate(self, dialogues: Optional[list] = None) -> dict:
        """生成完整的克隆配置"""
        template = self.build_prompt_template(dialogues)

        clone_data = {
            'config': {
                'model_type': self.config.model_type,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens
            },
            'prompt_template': {
                'system_prompt': template.system_prompt,
                'user_prompt_template': template.user_prompt_template,
                'examples': template.examples
            },
            'style_instructions': template.style_instructions,
            'profile_summary': {
                'avg_sentence_length': round(self.profile.avg_sentence_length, 2),
                'reasoning_style': self.profile.reasoning_style,
                'sentiment_score': round(self.profile.sentiment_score, 3),
                'top_keywords': self.profile.top_keywords[:10]
            }
        }

        return clone_data

    def save(self, output_path: str) -> None:
        """保存克隆配置"""
        clone_data = self.generate()
        clone_data['file_info'] = {
            'version': '1.0',
            'description': 'AI思维克隆配置文件'
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clone_data, f, ensure_ascii=False, indent=2)

        print(f"克隆配置已保存至: {output_path}")

    def export_for_openai(self, output_path: str) -> None:
        """导出为OpenAI API可用格式"""
        clone_data = self.generate()

        messages = [{"role": "system", "content": clone_data['prompt_template']['system_prompt']}]

        for example in clone_data['prompt_template']['examples']:
            messages.append({"role": "user", "content": example['user']})
            messages.append({"role": "assistant", "content": example['assistant']})

        export_data = {
            'messages': messages,
            'config': {
                'model': clone_data['config']['model_type'],
                'temperature': clone_data['config']['temperature'],
                'max_tokens': clone_data['config']['max_tokens']
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        print(f"OpenAI格式配置已保存至: {output_path}")


class CloneValidator:
    """克隆效果验证器"""

    def __init__(self, original_profile: StyleProfile, clone_profile: StyleProfile):
        """
        初始化验证器

        Args:
            original_profile: 原始风格画像
            clone_profile: 克隆输出风格画像
        """
        self.original = original_profile
        self.clone = clone_profile

    def calculate_similarity(self) -> dict:
        """计算风格相似度"""
        metrics = {}

        # 句长相似度
        len_diff = abs(self.original.avg_sentence_length - self.clone.avg_sentence_length)
        max_len = max(self.original.avg_sentence_length, self.clone.avg_sentence_length, 1)
        metrics['sentence_length_similarity'] = 1 - (len_diff / max_len)

        # 情感倾向相似度
        sent_diff = abs(self.original.sentiment_score - self.clone.sentiment_score)
        metrics['sentiment_similarity'] = 1 - (sent_diff / 2)

        # 关键词重合度
        orig_keywords = set(self.original.top_keywords)
        clone_keywords = set(self.clone.top_keywords)
        if orig_keywords or clone_keywords:
            overlap = len(orig_keywords & clone_keywords)
            union = len(orig_keywords | clone_keywords)
            metrics['keyword_similarity'] = overlap / union if union > 0 else 0
        else:
            metrics['keyword_similarity'] = 0

        # 推理风格一致性
        metrics['reasoning_consistency'] = 1 if self.original.reasoning_style == self.clone.reasoning_style else 0

        # 综合评分
        weights = {
            'sentence_length_similarity': 0.2,
            'sentiment_similarity': 0.2,
            'keyword_similarity': 0.4,
            'reasoning_consistency': 0.2
        }

        metrics['overall_score'] = sum(metrics[k] * weights[k] for k in weights)

        return metrics

    def generate_report(self) -> str:
        """生成验证报告"""
        metrics = self.calculate_similarity()

        report = f"""## 克隆效果验证报告

### 相似度指标

| 维度 | 相似度 |
|------|--------|
| 句长风格 | {metrics['sentence_length_similarity']:.2%} |
| 情感倾向 | {metrics['sentiment_similarity']:.2%} |
| 关键词使用 | {metrics['keyword_similarity']:.2%} |
| 推理风格 | {metrics['reasoning_consistency']:.2%} |

### 综合评分

**{metrics['overall_score']:.2%}**

### 改进建议
"""

        if metrics['overall_score'] < 0.7:
            report += "- 建议增加训练样本数量以提高克隆效果\n"
            report += "- 检查对话数据的质量和代表性\n"
        else:
            report += "- 当前克隆效果良好，可进行实际应用测试\n"

        return report


if __name__ == '__main__':
    # 示例用法
    from .analyzer import StyleAnalyzer

    # 分析原始数据
    analyzer = StyleAnalyzer('data/dialogues.json')
    profile = analyzer.analyze()

    # 生成克隆
    generator = CloneGenerator(profile)
    generator.save('output/my_clone.json')