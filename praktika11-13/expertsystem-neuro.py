import json
import datetime


class ExpertSystem:
    def __init__(self):
        # База знаний с правилами
        self.knowledge_base = [
            {
                "id": 1,
                "conditions": ["компьютер не включается", "нет индикации"],
                "conclusion": "Проблема с питанием",
                "explanation": "Проверьте подключение кабеля питания, блок питания или розетку.",
                "confidence": 0.9
            },
            {
                "id": 2,
                "conditions": ["самопроизвольное выключение", "зависания", "шум вентилятора"],
                "conclusion": "Перегрев процессора",
                "explanation": "Очистите систему охлаждения от пыли и замените термопасту.",
                "confidence": 0.85
            },
            {
                "id": 3,
                "conditions": ["синий экран", "компьютер пищит"],
                "conclusion": "Ошибка оперативной памяти",
                "explanation": "Попробуйте переустановить планки памяти в слотах или протестируйте их программой MemTest86.",
                "confidence": 0.8
            },
            {
                "id": 4,
                "conditions": ["артефакты на экране", "вылеты драйвера"],
                "conclusion": "Неисправность видеокарты",
                "explanation": "Проверьте температуру видеокарты и обновите драйверы. Если не поможет — возможен отвал чипа.",
                "confidence": 0.75
            },
            {
                "id": 5,
                "conditions": ["долгая загрузка", "пропадание файлов"],
                "conclusion": "Проблема с накопителем (HDD/SSD)",
                "explanation": "Проверьте состояние диска через S.M.A.R.T. и сделайте резервную копию данных.",
                "confidence": 0.7
            },
            {
                "id": 6,
                "conditions": ["циклическая перезагрузка", "ошибки реестра"],
                "conclusion": "Ошибка операционной системы",
                "explanation": "Попробуйте восстановить систему из точки восстановления или переустановить ОС.",
                "confidence": 0.65
            }
        ]
        self.history_file = "diagnostic_history.json"

    def infer(self, symptoms):
        """Механизм прямого вывода"""
        results = []
        for rule in self.knowledge_base:
            # Считаем количество совпавших симптомов
            matched_conditions = sum(1 for cond in rule["conditions"] if cond in symptoms)

            # Если совпало более 50% условий правила
            if matched_conditions >= len(rule["conditions"]) * 0.5:
                rule_result = rule.copy()
                rule_result["match_ratio"] = matched_conditions / len(rule["conditions"])
                results.append(rule_result)

        # Сортируем результаты по точности совпадения
        return sorted(results, key=lambda x: x["match_ratio"], reverse=True)

    def explain_decision(self, result):
        """Модуль объяснения решений"""
        print(f"\n--- Диагноз: {result['conclusion']} ---")
        print(f"Уверенность системы: {result['confidence'] * 100}%")
        print(f"Рекомендация: {result['explanation']}")

    def save_history(self, symptoms, conclusion):
        """Сохранение истории сеанса в JSON[cite: 3]"""
        entry = {
            "timestamp": str(datetime.datetime.now()),
            "input_symptoms": symptoms,
            "conclusion": conclusion
        }
        try:
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except FileNotFoundError:
                data = []

            data.append(entry)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Ошибка сохранения истории: {e}")


# Основной цикл программы
def main():
    es = ExpertSystem()
    print("Экспертная система диагностики ПК v2.0")
    print("Введите симптомы через запятую (например: синий экран, компьютер пищит):")

    user_input = input("> ").lower()
    symptoms = [s.strip() for s in user_input.split(',')]

    matches = es.infer(symptoms)

    if matches:
        top_match = matches[0]
        es.explain_decision(top_match)

        # Обратная связь[cite: 3]
        feedback = input("\nПомог ли вам этот диагноз? (да/нет): ").lower()
        if feedback == 'да':
            es.save_history(symptoms, top_match['conclusion'])
            print("Результат сохранен в историю.")
    else:
        print("К сожалению, система не смогла определить неисправность по данным симптомам.")


if __name__ == "__main__":
    main()