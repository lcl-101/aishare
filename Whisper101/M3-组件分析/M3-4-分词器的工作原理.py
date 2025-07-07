"""
M3-4: 分词器 (Tokenizer) 的工作原理

本脚本演示 Whisper 工作流的第四步：分词器的作用。
分词器负责将文本字符串与数字ID序列进行相互转换，是模型能够
理解和生成文本的基础。

依赖安装:
- pip install -U openai-whisper
"""

import os
import whisper

def demonstrate_tokenizer(language="zh"):
    """
    演示 Whisper 的多语言分词器如何进行编码和解码。

    Args:
        language (str): 要使用的分词器的语言代码 (e.g., "zh", "en", "ja").
    """
    print(f"--- 演示 Whisper 多语言分词器 (语言: {language}) ---")

    # 1. 获取对应语言的分词器
    # Whisper 的分词器是多语言的，需要指定语言才能正确处理特殊 token
    try:
        # `get_tokenizer` 会根据 multilingual 和 language 参数返回一个配置好的分词器实例
        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=language)
        print(f"成功获取 '{language}' 语言的分词器。")
        
        # 获取词汇表大小 - 使用 encoding 对象的 n_vocab 属性
        vocab_size = tokenizer.encoding.n_vocab
        print(f"词汇表大小 (Vocabulary Size): {vocab_size}")
    except Exception as e:
        print(f"错误: 获取分词器失败 -> {e}")
        return

    # 2. 演示编码 (Encode): 文本 -> Token ID
    print("\n--- 1. 编码 (Text -> Token IDs) ---")
    text_to_encode = "你好，世界。这是 OpenAI Whisper。"
    print(f"原始文本: '{text_to_encode}'")
    
    encoded_ids = tokenizer.encode(text_to_encode)
    print(f"编码后的 Token ID 序列: \n{encoded_ids}")
    
    # 3. 演示解码 (Decode): Token ID -> 文本
    print("\n--- 2. 解码 (Token IDs -> Text) ---")
    # 我们使用上面编码得到的 ID 序列进行解码
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"从ID序列解码回的文本: '{decoded_text}'")

    # 4. 演示特殊 Token
    print("\n--- 3. 了解特殊 Tokens ---")
    print("这些是模型用来控制转录流程的特殊'指令'。")
    
    # 开始转录 Token (Start of Transcript)
    sot_token_id = tokenizer.sot
    sot_token_text = tokenizer.decode([sot_token_id])
    print(f"- SOT (Start of Transcript): ID={sot_token_id},  文本表示='{sot_token_text}'")

    # 结束转录 Token (End of Transcript)
    eot_token_id = tokenizer.eot
    eot_token_text = tokenizer.decode([eot_token_id])
    print(f"- EOT (End of Transcript):  ID={eot_token_id},  文本表示='{eot_token_text}'")

    # 无语音 Token (No Speech)
    no_speech_token_id = tokenizer.no_speech
    no_speech_token_text = tokenizer.decode([no_speech_token_id])
    print(f"- No Speech:              ID={no_speech_token_id}, 文本表示='{no_speech_token_text}'")

    # 语言 Token (例如，中文)
    lang_token_id = tokenizer.language_token
    lang_token_text = tokenizer.decode([lang_token_id])
    print(f"- '{language}' 语言 Token:     ID={lang_token_id}, 文本表示='{lang_token_text}'")


def main():
    """
    脚本主入口函数。
    """
    # 演示中文分词器
    demonstrate_tokenizer(language="zh")
    
    # 你也可以取消下面的注释，看看英文分词器有什么不同
    # print("\n" + "="*50 + "\n")
    # demonstrate_tokenizer(language="en")


if __name__ == "__main__":
    main()