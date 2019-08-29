from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from pytorch_pretrained_bert import BertTokenizer
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

model = torch.load('output')
model.eval()
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-chinese', do_lower_case=True)


@app.route("/predict", methods=['POST'])
def predict():
    text = request.json["text"]
    label = request.json["label"]
    try:
        tokenized_texts = tokenizer.tokenize(text)
        b_input_ids = torch.LongTensor(
            [tokenizer.convert_tokens_to_ids(tokenized_texts)])
        b_input_mask = torch.LongTensor(
            [[float(i > 0) for i in b_input_ids[0]]])
        with torch.no_grad():
            logits = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        return jsonify({
            "result":
            g_cnns[torch.argmax(logits, dim=1)[0].item()][1],
            "label":
            label
        })
    except Exception as e:
        print(e)
        return jsonify({"result": "Model Failed"})


g_cnns = {
    0: [100, '民生 故事', 'news_story'],
    1: [101, '文化 文化', 'news_culture'],
    2: [102, '娱乐 娱乐', 'news_entertainment'],
    3: [103, '体育 体育', 'news_sports'],
    4: [104, '财经 财经', 'news_finance'],
    5: [106, '房产 房产', 'news_house'],
    6: [107, '汽车 汽车', 'news_car'],
    7: [108, '教育 教育', 'news_edu'],
    8: [109, '科技 科技', 'news_tech'],
    9: [110, '军事 军事', 'news_military'],
    10: [112, '旅游 旅游', 'news_travel'],
    11: [113, '国际 国际', 'news_world'],
    12: [114, '证券 股票', 'stock'],
    13: [115, '农业 三农', 'news_agriculture'],
    14: [116, '电竞 游戏', 'news_game']
}

if __name__ == "__main__":
    app.run('0.0.0.0', port=8000)
