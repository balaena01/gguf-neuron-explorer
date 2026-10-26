# チャットテンプレートの上書き

GGUF の `tokenizer.chat_template` が壊れている・入っていないモデル向け(設計書 §5.7 の 4)。
`<model_id>.yaml`(`gne info` に出る)か `<arch>.yaml` を置くと、こちらを使う(model_id が優先)。

```yaml
chat_template: |
  {% for m in messages %}<|im_start|>{{ m.role }}
  {{ m.content }}<|im_end|>
  {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
  {% endif %}
```
