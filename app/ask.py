"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 対話モード

ツールを持たない「素の」呼び出しだが、対話モードでは会話履歴を保持する。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from typing import Any

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, history: list[dict[str, Any]] | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    contents: str | list[dict[str, Any]] = question
    if history is not None:
        contents = [
            *history,
            {"role": "user", "parts": [{"text": question}]},
        ]
    response = client.models.generate_content(model=MODEL, contents=contents)

    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[usage] input_tokens={usage.prompt_token_count}, "
            f"output_tokens={usage.candidates_token_count}"
        )

    answer = response.text
    if history is not None:
        history.extend(
            [
                {"role": "user", "parts": [{"text": question}]},
                {"role": "model", "parts": [{"text": answer}]},
            ]
        )
    return answer


def interactive() -> None:
    """質問を繰り返し、同じ会話履歴を使って回答する。"""
    history: list[dict[str, Any]] = []
    print("質問を入力してください。終了するには exit または quit を入力してください。")
    while True:
        try:
            question = input("\n[質問] ")
        except EOFError:
            print()
            break
        if question.strip().lower() in {"exit", "quit"}:
            break
        if not question.strip():
            continue
        try:
            print(ask(question, history))
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", help="質問（省略すると対話モード）")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.question is None:
        interactive()
        return

    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
