from datetime import datetime
from bson import ObjectId
from langchain_core.messages import HumanMessage
from pytz import timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TransformerModel:
    def __init__(self, mongodb, mariadb, device):
        self.EMOTION_TO_SCORE_0_5 = {
            "기쁨": 5,
            "설렘": 4,
            "평범함": 3,
            "불쾌함": 2,
            "슬픔": 2,
            "놀라움": 1,
            "두려움": 0,
            "분노": 0,
        }
        self.SCORE_TO_WEATHER = {
            0: "토네이도",
            1: "번개",
            2: "비",
            3: "구름",
            4: "맑음",
            5: "최고",
        }
        self.seoul_tz = timezone("Asia/Seoul")
        model_path = "LimYeri/HowRU-KoELECTRA-Emotion-Classifier"
        self.mongodb = mongodb
        self.mariadb = mariadb
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.emotion_model.eval()
        self.id2label = self.emotion_model.config.id2label

    def _score_to_weather(self, score: float) -> str:
        s = int(round(score))
        s = max(0, min(5, s))
        return self.SCORE_TO_WEATHER[s]
    def _clamp_0_5(self, x: float) -> float:
        return max(0.0, min(5.0, x))

    def _analyze_emotion_score(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.emotion_model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        probs_list = probabilities.tolist()

        pred_id = int(torch.argmax(probabilities).item())
        predicted_label = self.id2label[pred_id]

        score_0_5 = self._clamp_0_5(float(self.EMOTION_TO_SCORE_0_5.get(predicted_label, 3)))
        scores = {self.id2label[i]: round(p * 100, 2) for i, p in enumerate(probs_list)}

        return {
            "예측": predicted_label,
            "확률": scores,
            "척도값": score_0_5,
        }
    def update_db(self, user_code, final_score, overall_emotion_label):
        target_date = datetime.now(self.seoul_tz).date()
        create_at = datetime.now(self.seoul_tz).replace(tzinfo=None)
        latest = self.mariadb.get_latest_by_user_and_date(user_code=user_code, target_date=target_date)
        if latest and latest.get("analysis_code"):
            summary_keep = None if latest.get("summary") is None else latest.get("summary")
            self.mariadb.update(
                analysis_code=int(latest["analysis_code"]),
                emotion_score=final_score,
                emotion_name=overall_emotion_label,
                summary=summary_keep,
                create_at=create_at,
            )
        else:

            self.mariadb.insert(
                user_code=user_code,
                emotion_score=final_score,
                emotion_name=overall_emotion_label,
                summary=None,
                create_at=create_at,
            )

    def inference(self, user_code: int, conv_id: ObjectId):
        messages = self.mongodb.get_chat_history(user_code, conv_id)
        user_utterances = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
        if not user_utterances:
            return None, None
        analysis_results = []
        total_scale_score = 0.0
        for text in user_utterances:
            analysis = self._analyze_emotion_score(text)
            analysis_results.append(
                {
                    "text": text,
                    "prediction": analysis["예측"],
                    "score_scale": analysis["척도값"],
                    "probs": analysis["확률"],
                }
            )
            total_scale_score += float(analysis["척도값"])
            avg_scale_score = self._clamp_0_5(total_scale_score / len(user_utterances))
            final_score = round(avg_scale_score, 2)
            overall_emotion_label = self._score_to_weather(final_score)
            return [final_score, overall_emotion_label]



